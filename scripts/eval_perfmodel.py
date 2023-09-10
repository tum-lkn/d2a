import os
import json
from typing import List, Dict, Union, Tuple, Any

import matplotlib.pyplot as plt
import torch
from ray.tune import Analysis
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as skm

import layers.vnfmodel as vnfmodel
from layers.datasets import GraphDataset
import evaluation.plotutils as plutils


def load_tune_result(exp_dir: str) -> Analysis:
    ana = Analysis(exp_dir)
    return ana


def _extend_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfgs = []
    for _ in range(config['model_config']['num_attn_modules']):
        cfg = {k: v for k, v in config['model_config']['attn_module_config'].items()}
        cfg['dim_in'] = config['model_config']['dim_initial']
        cfg['dim_linear'] = config['model_config']['dim_initial']
        cfgs.append(cfg)
    config['model_config']['attention_configs'] = cfgs
    return config


def load_model(checkpoint_dir: str) -> vnfmodel.DenseGatPerfPlayerModel:
    trial_dir = os.path.split(checkpoint_dir.rstrip('/'))[0]
    print("Load: ", os.path.join(trial_dir, 'params.json'))
    with open(os.path.join(trial_dir, 'params.json'), 'r') as fh:
        params = json.load(fh)
    state_dict = torch.load(os.path.join(checkpoint_dir, 'model.pth'), map_location='cpu')
    params = _extend_config(params)
    if 'nn_scale' not in params['model_config']:
        params['model_config']['nn_scale'] = 1999853.335557038
    config = vnfmodel.GatModelConfig.from_dict(params['model_config'])
    model = vnfmodel.DenseGatPerfPlayerModel(config)
    model.load_state_dict(state_dict)
    return model


def load_best_model(ana: Analysis) -> vnfmodel.DenseGatPerfPlayerModel:
    best_logdir = ana.get_best_logdir(metric='val_loss', mode='min')
    best_checkpoint_dir = ana.get_best_checkpoint(best_logdir, 'val_loss', 'min')
    return load_model(best_checkpoint_dir)


def get_json_result_data(trial_dir) -> List[Dict[str, Any]]:
    with open(os.path.join(trial_dir, 'result.json'), 'r') as fh:
        lines = [json.loads(l) for l in fh.readlines()]
    return lines


if __name__ == '__main__':
    # tmp = '/opt/project/data/nas/tune_perfmodel/GoldenSamplesDenseGatSoftFailure/GatTrainable_4b1f4_00218'
    # a = load_tune_result('/opt/project/data/nas/tune_perfmodel/GoldenSamplesDenseGatSoftFailure')
    a = load_tune_result('/opt/project/data/tune/GoldenSamplesDenseGatSoftFailure-new-space')
    ldir = str(a.get_best_logdir(metric='val_loss', mode='min'))
    exp_name = os.path.split(os.path.split(ldir)[0])[1]
    fig_dir = os.path.join('/opt/project/Graphs', exp_name)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    step_results = get_json_result_data(ldir)
    fig, ax = plutils.get_fig(1)
    ax.plot([a['val_loss'] for a in step_results], c=plutils.COLORS[0],
            marker=plutils.MARKER[0], markevery=30, markeredgecolor='black',
            markerfacecolor='white')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'validation-loss.pdf'))
    plt.close(fig)

    model = load_best_model(a)
    dset = GraphDataset.from_hdf5('/opt/project/data/nas/graphs-golden-samples/val-set.h5')
    loader = DataLoader(dset, batch_size=200, shuffle=True)
    y = np.array([])
    z = np.array([])
    for i, sample in enumerate(loader):
        print(f"Batch {i}")
        DEV = 'cpu'
        with torch.no_grad():
            prediction = model(
                node_features=sample['node_features'].to(DEV),
                query_idxs=sample['query_idx'].to(DEV),
                masks=sample['masks'].to(DEV),
                adj=sample['adj'].to(DEV)
            )
            prediction = torch.softmax(prediction, dim=-1)
            y = np.concatenate([y, torch.argmax(prediction, dim=-1).numpy()])
            z = np.concatenate([z, torch.argmax(sample['target'], dim=-1).numpy()])
    acc = skm.accuracy_score(y, z)
    rec = skm.recall_score(y, z)
    prc = skm.precision_score(y, z)
    msg = json.dumps(
        {
            'Logdir': ldir,
            'Accuracy': acc,
            'Recall': rec,
            'Precision': prc
        },
        indent=1
    )
    print(msg)
    with open(os.path.join(fig_dir, 'stats.txt'), 'w') as fh:
        fh.write(msg)
