import os
import json
import sys

import h5py
import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from typing import Dict, Any
from ray.tune.utils import pin_in_object_store, get_pinned_object
from typing import List, Dict, Tuple, Union, Any

from layers.datasets import GraphDataset
import layers.vnfmodel as vnfmodel
import dataprep.utils as dutils


METRIC_KEY = 'val_loss'
if torch.cuda.is_available():
    DEV = torch.device("cuda")
else:
    DEV = torch.device("cpu")


class GatTrainable(tune.Trainable):
    """
    Trainable for shortest path models.
    """
    def setup(self, config: Dict[str, Any]):
        """
        Create a single trainable model.

        Args:
            config: Dict that contains all hyper parameters. Contains the
                keys `model_config` with the configuration options for the
                model and key `optimizer` with parameters for the optimizer
                to use.
        """
        if 'best_trial_dir' in config:
            log_dir, _ = os.path.split(config['best_trial_dir'].rstrip('/'))
            with open(os.path.join(log_dir, 'params.json'), 'r') as fh:
                loaded_config = json.load(fh)
            for k, v in config.items():
                loaded_config[k] = v
            config = self._extend_config(loaded_config)
            self.model = vnfmodel.DenseGatPerfPlayerModel(
                vnfmodel.GatModelConfig.from_dict(config['model_config'])
            )
            # self.model.load_state_dict(
            #     torch.load(os.path.join(config['best_trial_dir'], 'model.pth')),
            #     strict=False
            # )
        else:
            config = self._extend_config(config)
            self.model = vnfmodel.DenseGatPerfPlayerModel(
                vnfmodel.GatModelConfig.from_dict(config['model_config'])
            )
        self.model = self.model.to(DEV)
        self.batch_size = config['batch_size']
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay']
        )

    def _extend_config(self, config: Dict[str, Any]):
        cfgs = []
        for _ in range(config['model_config']['num_attn_modules']):
            cfg = {k: v for k, v in config['model_config']['attn_module_config'].items()}
            cfg['dim_in'] = config['model_config']['dim_initial']
            cfg['dim_linear'] = config['model_config']['dim_initial']
            cfgs.append(cfg)
        config['model_config']['attention_configs'] = cfgs
        return config

    def _loss_batch(self, sample, opt=None) -> Tuple[np.array, int]:
        """
        Perform forward and backward pass for one minibatch. In case of training,
        the optimizer changes the parameter. In case of validation, this does
        not happen.

        Args:
            loss_fct:
            sample:
            opt:

        Returns:

        """
        pred = self.model(
            node_features=sample['node_features'].to(DEV),
            query_idxs=sample['query_idx'].to(DEV),
            masks=sample['masks'].to(DEV),
            adj=sample['adj'].to(DEV)
        )
        if self.model.config.target in ['soft_overload', 'hard_overload']:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                input=pred,
                target=sample['target'].to(DEV),
                weight=torch.unsqueeze(sample['weight'], dim=-1).to(DEV)
            )
        else:
            loss = torch.nn.functional.mse_loss(
                input=pred,
                target=sample['target'].to(DEV)
            )
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.cpu().item(), sample['target'].shape[0]

    def _get_data_set(self, store_id):
        """
        Create a dataset from a memory pinned dictionary containing the
        relevant data.

        Args:
            store_id:

        Returns:

        """
        dataset_dict = get_pinned_object(store_id)
        dataset = GraphDataset(target=self.model.config.target, **dataset_dict)
        return dataset

    def step(self):
        """
        Train the model for the specified number of epochs.

        Returns:

        """
        dset_train = self._get_data_set(TRAINING_DATA_ID)
        dset_val = self._get_data_set(VALIDATION_DATA_ID)
        loader_train = DataLoader(dset_train, batch_size=self.batch_size, shuffle=True)
        loader_val = DataLoader(dset_val, batch_size=2 * self.batch_size)

        self.model.train()
        for batch, sample in enumerate(loader_train):
            self._loss_batch(sample, self.optimizer)

        self.model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[self._loss_batch(sample) for sample in loader_val]
            )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        return {METRIC_KEY: val_loss}

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, checkpoint):
        checkpoint_path = os.path.join(checkpoint, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))

    def reset_config(self, new_config):
        if "lr" in new_config:
            self.optimizer.param_groups[0]['lr'] = new_config['lr']
        if "batch_size" in new_config:
            self.batch_size = new_config['batch_size']
        self.config = new_config
        return True


def sample_config() -> vnfmodel.GatModelConfig:
    return vnfmodel.GatModelConfig(
        attention_configs=[
            vnfmodel.AttnModuleConfig(
                num_heads=3,
                attention_class='Gat',
                dim_in=20,
                dim_hidden=20,
                dim_out=20,
                dim_linear=20
            ),
            vnfmodel.AttnModuleConfig(
                num_heads=3,
                attention_class='Gat',
                dim_in=20,
                dim_hidden=20,
                dim_out=20,
                dim_linear=20
            )
        ],
        dim_initial=20,
        dim_hiddens=[50, 50],
        dim_out=2,
        dim_in=21,
        nn_scale=1.
    )


def run_pbt(num_fts: int, target: str, class_weight: float = None):
    scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        metric=METRIC_KEY,
        mode='min',
        hyperparam_mutations={
            'lr': lambda: np.random.uniform(0.000001, 0.001)
        },
        perturbation_interval=5
    )

    # For dim_out_fcn lists of length 1-3 are sampled, i.e., the dense part
    # of the model has one to three hidden layers of varying size.
    search_space = {
        "seed": seed,
        "batch_size": 294,
        "model_config": {
            "target": target,
            "dim_initial": 16,
            "num_attn_modules": 3,
            "attn_module_config": {
                "num_heads": 5,
                "dim_hidden": 24,
                "dim_out": 21,
                "dim_in": None,
                "dim_linear": None,
                "attention_class": "Gat"
            },
            "dim_hiddens": [60, 41],
            "dim_out": 1 if target in ['vnf_throughput', 'sfc_throughput', 'sfc_latency'] else 2,
            "dim_in": num_fts,
            "nn_scale": {
                'soft_overload': 1.,
                'hard_overload': 1.,
                'vnf_throughput': 2349759.0518103624,
                'sfc_throughput': 2344295.7255441803,
                'sfc_latency': 2.751976938915141  # 15.673587
            }[target]
        },
        "optimizer": {
            "lr": 1.591408722500197e-05,
            "weight_decay": 0
        }
    }
    analysis = tune.run(
        GatTrainable,
        resources_per_trial={'cpu': 1, 'gpu': 0.24},
        num_samples=10,
        stop={"training_iteration": 160},
        scheduler=scheduler,
        config=search_space,
        name=f'DenseGatNewSpacePbt{"" if class_weight is None else f"W{int(class_weight)}"}-{target}',
        checkpoint_at_end=True,
        checkpoint_freq=10,
        checkpoint_score_attr='min-' + METRIC_KEY,
        local_dir='/opt/project/data/tune',
        sync_on_checkpoint=False,
        trial_dirname_creator=lambda trial: 'GatTrainable_{:s}'.format(trial.trial_id)
    )
    print("Best config: ", analysis.get_best_config(metric=METRIC_KEY))


def run_asha(num_fts: int, target: str):
    asha = ASHAScheduler(
        time_attr='training_iteration',
        metric=METRIC_KEY,
        mode='min',
        max_t=320,
        grace_period=10,
        reduction_factor=2
    )

    # For dim_out_fcn lists of length 1-3 are sampled, i.e., the dense part
    # of the model has one to three hidden layers of varying size.
    search_space = {
        "seed": seed,
        "batch_size": tune.sample_from(lambda _: int(random.randint(256, 513))),
        "model_config": {
            "target": target,
            "dim_initial": tune.sample_from(lambda _: int(random.randint(10, 31))),
            "num_attn_modules": tune.sample_from(lambda _: int(random.randint(1, 6))),
            "attn_module_config": {
                "num_heads": tune.sample_from(lambda _: int(random.randint(1, 6))),
                "dim_hidden": tune.sample_from(lambda _: int(random.randint(10, 33))),
                "dim_out": tune.sample_from(lambda _: int(random.randint(10, 33))),
                "dim_in": None,
                "dim_linear": None,
                "attention_class": "Gat"
            },
            "dim_hiddens": tune.sample_from(lambda _: random.choice(
                np.arange(20, 100),
                replace=True,
                size=random.choice([1, 2])).tolist()),
            "dim_out": 1 if target in ['vnf_throughput', 'sfc_throughput', 'sfc_latency'] else 2,
            "dim_in": num_fts,
            "nn_scale": {
                'soft_overload': 1.,
                'hard_overload': 1.,
                'vnf_throughput': 2349759.0518103624,
                'sfc_throughput': 2344295.7255441803,
                'sfc_latency': 2.751976938915141 #15.673587
            }[target]
        },
        "optimizer": {
            "lr": tune.sample_from(lambda _: float(10. ** (-1 * random.uniform(3, 6)))),
            "weight_decay": 0
        }
    }

    analysis = tune.run(
        GatTrainable,
        resources_per_trial={'cpu': 1, 'gpu': 0.32},
        num_samples=1000,
        scheduler=asha,
        config=search_space,
        name=f'DenseGatNewSpace-{target}',
        checkpoint_at_end=True,
        checkpoint_freq=10,
        checkpoint_score_attr='min-' + METRIC_KEY,
        local_dir='/opt/project/data/tune',
        sync_on_checkpoint=False,
        trial_dirname_creator=lambda trial: 'GatTrainable_{:s}'.format(trial.trial_id)
    )
    print("Best config: ", analysis.get_best_config(metric=METRIC_KEY))


def check_model_executing():
    dset = GraphDataset.from_hdf5('/opt/project/data/nas/graphs-ma-diederich/val-set.h5')
    loader = DataLoader(dset, batch_size=10, shuffle=True)
    model = vnfmodel.DenseGatPerfPlayerModel(sample_config())
    for sample in loader:
        logits = model(
            node_features=sample['node_features'],
            query_idxs=sample['query_idx'],
            masks=sample['masks'],
            adj=sample['adj']
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            sample['target'],
            weight=torch.unsqueeze(sample['weight'], dim=-1)
        )
        print(loss)
        break
    print("Done")


if __name__ == '__main__':
    ray.init()
    seed = 1
    target = '' if len(sys.argv) == 1 else sys.argv[1]
    choice = '\n\t'.join([f'{i + 1} {v}' for i, v in enumerate(GraphDataset.possible_targets)])
    while target not in GraphDataset.possible_targets:
        choice = int(input(f"Choose target:\n\t{choice}\nYour Choice: ")) - 1
        target = GraphDataset.possible_targets[choice]
    print(f"Your choice: {target}")

    class_weight = None if len(sys.argv) < 3 else float(sys.argv[2])
    random = np.random.RandomState(seed=seed)
    # pos-numa-graph-dset-train.h5
    # dataset = GraphDataset.from_hdf5('/opt/project/data/nas/graphs-ma-diederich/train-set.h5')
    dataset = GraphDataset.from_hdf5(
        '/opt/project/data/nas/graphs-golden-samples/train-set.h5',
        target=target,
        cpu_overload_class_weight=class_weight
    )
    TRAINING_DATA_ID = pin_in_object_store({
        k: v if type(v) == float else v.copy() for k, v in dataset.todict().items() if type(v) != str
    })
    # dataset = GraphDataset.from_hdf5('/opt/project/data/nas/graphs-ma-diederich/val-set.h5')
    dataset = GraphDataset.from_hdf5(
        '/opt/project/data/nas/graphs-golden-samples/val-set.h5',
        target=target,
        cpu_overload_class_weight = class_weight
    )
    VALIDATION_DATA_ID = pin_in_object_store({
        k: v if type(v) == float else v.copy() for k, v in dataset.todict().items() if type(v) != str
    })
    num_ft = dataset.node_features.shape[-1]
    del dataset
    # run_asha(num_ft, target)
    run_pbt(num_ft, target, class_weight)
    # run_pbt()

