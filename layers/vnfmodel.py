import h5py
import torch
import numpy as np
from typing import Union, List, Tuple, Any, Dict
import os

import layers.attn as attn
DEV = 'cuda'


class AttnModuleConfig(object):
    @classmethod
    def from_dict(cls, d):
        return cls(
            num_heads=d['num_heads'],
            attention_class=d['attention_class'],
            dim_hidden=d['dim_hidden'],
            dim_out=d['dim_out'],
            dim_in=d['dim_in'],
            dim_linear=d['dim_linear']
        )

    def __init__(self, num_heads: int, attention_class: str,
                 dim_hidden: int, dim_out: int, dim_in: int,
                 dim_linear: int):
        self.num_heads = num_heads
        self.attention_class = attention_class
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.dim_in = dim_in
        self.dim_linear = dim_linear


class GatModelConfig(object):
    @classmethod
    def from_dict(cls, d):
        return cls(
            attention_configs=[AttnModuleConfig.from_dict(dd) for dd in d['attention_configs']],
            dim_initial=d['dim_initial'],
            dim_hiddens=d['dim_hiddens'],
            dim_out=d['dim_out'],
            dim_in=d['dim_in'],
            nn_scale=d.get('nn_scale', 1999853.335557038),
            target=d.get('target', 'soft_overload')
        )

    def __init__(self, attention_configs: List[AttnModuleConfig],
                 dim_initial: int, dim_hiddens: List[int], dim_out: int,
                 dim_in: int, nn_scale: float, target:str = 'soft_overload'):
        self.attention_configs = attention_configs
        self.dim_hiddens = dim_hiddens
        self.dim_out = dim_out
        self.dim_initial = dim_initial
        self.dim_in = dim_in
        self.nn_scale = nn_scale
        self.target = target


class PerfModel(torch.nn.Module):
    def __init__(self):
        super(PerfModel, self).__init__()
        self.attn_module = attn.MultiHeadAttentionLayer(
            num_heads=5,
            attention_class='SelfAttentionLayer',
            dim_in=-1,
            dim_hidden=10,
            dim_out=10,
            dim_q=48,
            dim_k=32,
            dim_v=32
        )
        self.attn_linear = torch.nn.Linear(50, 50)
        self.nonlinear1 = torch.nn.Linear(50, 50)
        self.out = torch.nn.Linear(50, 2)
        self.act_fct = torch.nn.ELU()

    def forward(self, X_kv, X_q):
        out, weights = self.attn_module(
            keys=X_kv,
            values=X_kv,
            queries=X_q
        )
        self.attn_weights = weights
        out = torch.squeeze(out, dim=-2)
        out = self.act_fct(self.attn_linear(out))
        out = self.act_fct(self.nonlinear1(out))
        out = self.out(out)
        return out


class PerfPlayerModel(torch.nn.Module):

    def __init__(self):
        super(PerfPlayerModel, self).__init__()
        self.attn_module = attn.MultiHeadAttentionLayer(
            num_heads=5,
            attention_class='SelfAttentionLayer',
            dim_in=-1,
            dim_hidden=10,
            dim_out=10,
            dim_q=36,
            dim_k=36,
            dim_v=36
        )
        self.attn_linear = torch.nn.Linear(50, 50)
        self.nonlinear1 = torch.nn.Linear(50, 50)
        self.out = torch.nn.Linear(50, 1)
        self.out2 = torch.nn.Linear(50, 1)
        self.act_fct = torch.nn.ELU()
        self.attn_weights = None

    def forward(self, obs: torch.Tensor, query_idxs: torch.Tensor, masks: torch.Tensor,
                dev: str) -> torch.Tensor:
        queries = obs.reshape([obs.shape[0] * obs.shape[1], obs.shape[2]])
        queries = queries[query_idxs + torch.arange(obs.shape[0], device=dev) * obs.shape[1]].unsqueeze(1)
        out, weights = self.attn_module(
            keys=obs,
            values=obs,
            queries=queries,
            attention_mask=masks
        )
        self.weights = weights
        out = torch.squeeze(out, dim=-2)
        out = self.act_fct(self.attn_linear(out))
        out = self.act_fct(self.nonlinear1(out))
        out1 = self.out(out)
        out2 = self.out2(out)
        return out1, out2


class StackedPerfPlayerModel(torch.nn.Module):

    def __init__(self):
        super(StackedPerfPlayerModel, self).__init__()
        self.attn_module_one = attn.MultiHeadAttentionLayer(
            num_heads=5,
            attention_class='SelfAttentionLayer',
            dim_in=27,
            dim_hidden=10,
            dim_out=10
        )
        self.attn_module_two = attn.MultiHeadAttentionLayer(
            num_heads=5,
            attention_class='SelfAttentionLayer',
            dim_in=27,
            dim_hidden=10,
            dim_out=10
        )
        self.attn_module_three = attn.MultiHeadAttentionLayer(
            num_heads=5,
            attention_class='SelfAttentionLayer',
            dim_in=27,
            dim_hidden=10,
            dim_out=10,

        )
        self.attn_module_four = attn.MultiHeadAttentionLayer(
            num_heads=5,
            attention_class='SelfAttentionLayer',
            dim_in=27,
            dim_hidden=10,
            dim_out=10,

        )
        self.attn_linear_one = torch.nn.Linear(50, 27)
        self.attn_linear_two = torch.nn.Linear(50, 27)
        self.attn_linear_three = torch.nn.Linear(50, 27)
        self.attn_linear_four = torch.nn.Linear(50, 27)
        self.layer_norm_one = torch.nn.LayerNorm(27)
        self.layer_norm_two = torch.nn.LayerNorm(27)
        self.layer_norm_three = torch.nn.LayerNorm(27)
        self.nonlinear1 = torch.nn.Linear(27, 27)
        self.out = torch.nn.Linear(27, 2)
        self.act_fct = torch.nn.ELU()
        self.attn_weights_one = None
        self.attn_weights_two = None
        self.attn_weights_three = None
        self.attn_weights_four = None

    def forward(self, obs: torch.Tensor, query_idxs: torch.Tensor, masks: torch.Tensor,
                dev: str) -> torch.Tensor:
        out, weights = self.attn_module_one(
            keys=obs,
            values=obs,
            queries=obs,
            attention_mask=masks
        )
        self.attn_weights_one = weights
        out = self.act_fct(self.attn_linear_one(out))
        out = out + obs
        obs = self.layer_norm_one(out)

        out, weights = self.attn_module_two(
            keys=obs,
            values=obs,
            queries=obs,
            attention_mask=masks
        )
        self.attn_weights_two = weights
        out = self.act_fct(self.attn_linear_two(out))
        out = out + obs
        obs = self.layer_norm_one(out)

        queries = obs.reshape([obs.shape[0] * obs.shape[1], obs.shape[2]])
        queries = queries[query_idxs + torch.arange(obs.shape[0], device=dev) * obs.shape[1]].unsqueeze(1)

        out, weights = self.attn_module_three(
            keys=obs,
            values=obs,
            queries=queries,
            attention_mask=masks
        )
        self.attn_weights_three = weights
        queries = self.layer_norm_three(queries + self.act_fct(self.attn_linear_three(out)))

        out, weights = self.attn_module_four(
            keys=obs,
            values=obs,
            queries=queries,
            attention_mask=masks
        )
        self.attn_weights_four = weights
        out = self.act_fct(self.attn_linear_four(out))
        out = torch.squeeze(out, dim=-2)

        out = self.act_fct(self.nonlinear1(out))
        out = self.out(out)
        return out


class AttnModule(torch.nn.Module):
    def __init__(self, attn_module_config: AttnModuleConfig):
        super(AttnModule, self).__init__()
        self.attn_module = attn.MultiHeadAttentionLayer(
            num_heads=attn_module_config.num_heads,
            attention_class=attn_module_config.attention_class,
            dim_in=attn_module_config.dim_in,
            dim_hidden=attn_module_config.dim_hidden,
            dim_out=attn_module_config.dim_out,
            linear_class='Linear'
        )
        self.linear = torch.nn.Linear(
            attn_module_config.num_heads * attn_module_config.dim_out,
            attn_module_config.dim_linear
        )
        self.activation = torch.nn.ELU()
        self.attn_weights = None

    def forward(self, node_features: torch.Tensor, masks: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
            out, weights = self.attn_module(
                node_features=node_features,
                adj=adj,
                mask=masks
            )
            self.attn_weights = weights
            return self.activation(self.linear(out))


class StackedGatPerfPlayerModel(torch.nn.Module):

    def old_setting(self, linear_class):
        self.linear_class = linear_class
        self.attn_module_one = attn.MultiHeadAttentionLayer(
            num_heads=5,
            attention_class='Gat',
            dim_in=20,
            dim_hidden=10,
            dim_out=10,
            linear_class=linear_class
        )
        self.attn_module_two = attn.MultiHeadAttentionLayer(
            num_heads=5,
            attention_class='Gat',
            dim_in=20,
            dim_hidden=10,
            dim_out=10,
            linear_class=linear_class
        )
        self.attn_module_three = attn.MultiHeadAttentionLayer(
            num_heads=5,
            attention_class='Gat',
            dim_in=20,
            dim_hidden=10,
            dim_out=10,
            linear_class=linear_class

        )
        linear_cls = attn.LINEAR_FACTORY[linear_class]
        self.initial_transform = linear_cls(7, 20)
        self.attn_linear_one = linear_cls(50, 20)
        self.attn_linear_two = linear_cls(50, 20)
        self.attn_linear_three = linear_cls(50, 20)
        # self.attn_linear_four = linear_cls(50, 50)
        if linear_class == 'Linear':
            self.layer_norm_one = torch.nn.LayerNorm(20)
            self.layer_norm_two = torch.nn.LayerNorm(20)
            self.layer_norm_three = torch.nn.LayerNorm(20)
        self.nonlinear1 = linear_cls(20, 20)
        self.out = linear_cls(20, 3)
        self.act_fct = torch.nn.ELU()
        self.attn_weights_one = None
        self.attn_weights_two = None
        self.attn_weights_three = None
        self.attn_weights_four = None

    def __init__(self, gat_model_config: GatModelConfig):
        super(StackedGatPerfPlayerModel, self).__init__()
        self.config = gat_model_config
        self.attn_modules = torch.nn.ModuleList(
            [AttnModule(cfg) for cfg in gat_model_config.attention_configs]
        )
        self.layer_norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(cfg.dim_linear) for cfg in gat_model_config.attention_configs]
        )
        self.initial_transform = torch.nn.Linear(gat_model_config.dim_in, gat_model_config.dim_initial)
        dims_out = [x for x in gat_model_config.dim_hiddens]
        dims_out.append(gat_model_config.dim_out)
        dims_out.insert(0, gat_model_config.attention_configs[-1].dim_linear)
        layers = []
        for dim_in, dim_out in zip(dims_out[:-1], dims_out[1:]):
            layers.append(torch.nn.Linear(dim_in, dim_out))
        self.final_transforms = torch.nn.ModuleList(layers)
        self.activation = torch.nn.ELU()

    def forward(self, node_features: torch.Tensor, query_idxs: torch.Tensor, masks: torch.Tensor,
                adj: torch.Tensor, sim_results: torch.Tensor) -> torch.Tensor:
        node_features = self.activation(self.initial_transform(node_features))
        for attn_mod, layer_norm in zip(self.attn_modules, self.layer_norms):
            out = attn_mod(node_features, masks, adj)
            out = out + node_features
            node_features = layer_norm(out)

        query_idxs = torch.unsqueeze(torch.unsqueeze(query_idxs, -1), -1)
        query_idxs = query_idxs.repeat_interleave(node_features.shape[-1], -1)
        vnf_ft = torch.squeeze(torch.gather(node_features, 1, query_idxs), 1)
        self.vnf_ft = vnf_ft
        for linear in self.final_transforms:
            vnf_ft = self.activation(linear(vnf_ft))
        return vnf_ft * self.config.nn_scale

    def forward_old(self, node_features: torch.Tensor, query_idxs: torch.Tensor, masks: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
             node_features: (BS, num_nodes, num_features)
             query_idxs: (BS)
             masks: (BS, num_nodes, max_degree, 1)
             adj: (BS, num_nodes, max_degree)
             vnf_idx: (BS, max_num_vnfs)
             dev: str
        """
        node_features = self.act_fct(self.initial_transform(node_features))
        out, weights = self.attn_module_one(
            node_features=node_features,
            adj=adj,
            mask=masks
        )
        self.attn_weights_one = weights
        out = self.act_fct(self.attn_linear_one(out))
        out = out + node_features
        if self.linear_class == 'Linear':
            node_features = self.layer_norm_one(out)
        else:
            node_features = out

        out, weights = self.attn_module_two(
            node_features=node_features,
            adj=adj,
            mask=masks
        )
        self.attn_weights_two = weights
        out = self.act_fct(self.attn_linear_two(out))
        out = out + node_features
        if self.linear_class == 'Linear':
            node_features = self.layer_norm_three(out)
        else:
            node_features = out

        out, weights = self.attn_module_three(
            node_features=node_features,
            adj=adj,
            mask=masks
        )
        self.attn_weights_three = weights
        out = self.act_fct(self.attn_linear_three(out))
        out = out + node_features
        if self.linear_class == 'Linear':
            node_features = self.layer_norm_three(out)
        else:
            node_features = out

        query_idxs = torch.unsqueeze(torch.unsqueeze(query_idxs, -1), -1)
        query_idxs = query_idxs.repeat_interleave(node_features.shape[-1], -1)
        vnf_ft = torch.squeeze(torch.gather(node_features, 1, query_idxs), 1)

        out = self.act_fct(self.nonlinear1(vnf_ft))
        out = self.out(out)
        return out


class DenseGatPerfPlayerModel(torch.nn.Module):
    def __init__(self, gat_model_config: GatModelConfig):
        super(DenseGatPerfPlayerModel, self).__init__()
        self.config = gat_model_config
        self.attn_modules = torch.nn.ModuleList(
            [AttnModule(cfg) for cfg in gat_model_config.attention_configs]
        )
        self.initial_transform = torch.nn.Linear(gat_model_config.dim_in, gat_model_config.dim_initial)
        dims_out = [x for x in gat_model_config.dim_hiddens]
        dims_out.append(gat_model_config.dim_out)
        dims_out.insert(0, int(np.sum([ac.dim_linear for ac in gat_model_config.attention_configs])))
        layers = []
        for dim_in, dim_out in zip(dims_out[:-1], dims_out[1:]):
            layers.append(torch.nn.Linear(dim_in, dim_out))
        self.final_transforms = torch.nn.ModuleList(layers)
        self.activation = torch.nn.ELU()

    def forward(self, node_features: torch.Tensor, query_idxs: torch.Tensor, masks: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        node_features = self.activation(self.initial_transform(node_features))
        outs = []
        for attn_mod in self.attn_modules:
            node_features = attn_mod(node_features, masks, adj)
            outs.append(node_features)
            # out = out + node_features
            # node_features = layer_norm(out)
        node_features = torch.cat(outs, dim=-1)
        query_idxs = torch.unsqueeze(torch.unsqueeze(query_idxs, -1), -1)
        query_idxs = query_idxs.repeat_interleave(node_features.shape[-1], -1)
        vnf_ft = torch.squeeze(torch.gather(node_features, 1, query_idxs), 1)
        self.vnf_ft = vnf_ft
        for linear in self.final_transforms:
            vnf_ft = self.activation(linear(vnf_ft))
        return vnf_ft * self.config.nn_scale


class StackedGatPerfSimModel(StackedGatPerfPlayerModel):

    def __init__(self, gat_model_config: GatModelConfig):
        super(StackedGatPerfSimModel, self).__init__(gat_model_config)
        self.linear = torch.nn.Linear(gat_model_config.attention_configs[-1].dim_linear, 2)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, node_features: torch.Tensor, query_idxs: torch.Tensor, masks: torch.Tensor,
                adj: torch.Tensor, sim_results: torch.Tensor) -> torch.Tensor:
        nn_prediction = super(StackedGatPerfSimModel, self).forward(
                node_features, query_idxs, masks, adj, sim_results
        )
        weights = self.softmax(self.linear(self.vnf_ft))
        nn_sim = torch.cat([nn_prediction, sim_results], dim=-1)
        self.weights = weights
        self.nn_sim = nn_sim
        combined = torch.unsqueeze(torch.sum(weights * nn_sim, dim=-1), -1)
        return combined


class PerfPlayerEsModel(torch.nn.Module, attn.ModuleMutationMixin):

    def __init__(self):
        super(PerfPlayerEsModel, self).__init__()
        self.attn_module = attn.MultiHeadAttentionLayer(
            num_heads=5,
            attention_class='SelfAttentionLayer',
            dim_in=-1,
            dim_hidden=10,
            dim_out=10,
            dim_q=1,
            dim_k=2,
            dim_v=2,
            linear_class='MutateLinear'
        )
        self.attn_linear = attn.MutateLinear(50, 50)
        self.nonlinear1 = attn.MutateLinear(58, 50)
        self.out = attn.MutateLinear(50, 2)
        self.act_fct = torch.nn.ELU()
        self.attn_weights = None

    def forward(self, kv: torch.Tensor, mask: torch.Tensor, q: torch.Tensor,
                loads: torch.Tensor, aux: torch.Tensor, dev: str) -> torch.Tensor:
        out, weights = self.attn_module(
            keys=kv,
            values=kv,
            queries=q,
            attention_mask=mask
        )
        self.weights = weights
        # out = self.act_fct(self.attn_linear(torch.squeeze(out, dim=-2)))
        out = self.act_fct(self.attn_linear(out))
        out = torch.cat([out, loads, aux], dim=-1)
        out = self.act_fct(self.nonlinear1(out))
        out = self.out(out)
        return out


def load_data_set() -> Dict[str, np.array]:
    f = h5py.File('/opt/projects/vnf-perf-prediction/data/vnf-player-data-set.h5', 'r')
    d = {k: f[k][()] for k in f.keys()}
    f.close()
    return d


def load_data_set_(p) -> Dict[str, np.array]:
    f = h5py.File(p, 'r')
    d = {k: f[k][()] for k in f.keys()}
    f.close()
    return d


def train_val(data: Dict[str, np.array]) -> Tuple[Dict[str, np.array], Dict[str, np.array]]:
    """
    Create a training and a validation set. The validation set contains for
    each algorithm type samples from 20 experiments conducted with the
    corresponding assignment algorithm.
    """
    def get_num_samples(idx: int, algo_names: np.array) -> int:
        count = 1
        while algo_names[idx + count] == algo_names[idx] and idx + count < algo_names.size - 1:
            count += 1
        return count

    if os.path.exists('/opt/projects/vnf-perf-prediction/data/player-training-set.h5'):
        train = load_data_set_('/opt/projects/vnf-perf-prediction/data/player-training-set.h5')
        val = load_data_set_('/opt/projects/vnf-perf-prediction/data/player-validation-set.h5')
        train['X_q'] = train['X_q'].astype(np.int)
        val['X_q'] = val['X_q'].astype(np.int)
        return train, val

    train = {k: None for k in data.keys()}
    val = {k: None for k in data.keys()}
    val_sample_count = {}
    algo_names = data['algo_names']
    val_algo_names = []
    idx = 0
    while idx < algo_names.size - 5:
        num_samples = get_num_samples(idx, algo_names)
        if algo_names[idx] not in val_sample_count:
            val_sample_count[algo_names[idx]] = 0
        if val_sample_count[algo_names[idx]] >= 20:
            to_update = train
        else:
            val_sample_count[algo_names[idx]] += 1
            val_algo_names.append(algo_names[idx])
            to_update = val
        for k, v in data.items():
            slice = data[k][idx:idx + num_samples]
            to_update[k] = slice if to_update[k] is None else np.concatenate([to_update[k], slice])
        idx += num_samples
    train['X_q'] = train['X_q'].astype(np.int)
    val['X_q'] = val['X_q'].astype(np.int)

    f = h5py.File("/opt/projects/vnf-perf-prediction/data/player-training-set.h5", "w")
    for k, v in train.items():
        f.create_dataset(name=k, data=v)
    f.close()

    f = h5py.File("/opt/projects/vnf-perf-prediction/data/player-validation-set.h5", "w")
    for k, v in train.items():
        f.create_dataset(name=k, data=v)
    f.close()

    return train, val


def log_prob(Y, Z):
    lp = torch.distributions.Normal(Y[:, 0], torch.square(Y[:, 1])).log_prob(Z)
    reg = lp - 10 * torch.clamp_min(torch.square(Y[:, 1]), 1)
    return reg


def squared_loss(Y, Z):
    dif = Y[:, 0] - Z
    squares = torch.square(dif)
    return squares


def step(optimizer, net, batch):
    optimizer.zero_grad()

    out = net(
        node_features=batch['node_features'],
        query_idxs=batch['vnf_nodes'],
        masks=batch['mask'],
        adj=batch['adj']
    )
    loss = torch.mean(torch.square((batch['targets'] - out) * batch['target_weights']))

    # out1, out2 = net(batch['X_kv'], batch['X_q'], batch['masks'], DEV)
    # w = torch.unsqueeze(batch['weights'], dim=-1)
    # # lp = log_prob(Y=outputs, Z=torch.squeeze(batch['targets'] * 1e6, dim=-1))
    # # loss = -1. * torch.mean(w * lp)

    # loss1 = torch.mean(w * squared_loss(out1, torch.squeeze(batch['targets'] * 1e6, dim=-1)))
    # loss2 = torch.mean(w * squared_loss(out2, torch.squeeze(batch['estimated_targets'] * 1e6, dim=-1)))
    # loss = loss1 + loss2

    loss.backward()
    optimizer.step()
    return loss.item()


def train(t_set: Dict[str, torch.Tensor], v_set: Dict[str, torch.Tensor],
          net: torch.nn.Module, n_epochs=1000, batch_size=500):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    train_losses = [0.0]
    val_losses = [0.0]
    squared_losses = [0.0]
    best_val_loss = 1e21
    best_iter = 0
    for i in range(n_epochs):
        idx = 0
        count = 0
        loss = torch.tensor(0., device=DEV)
        indices = torch.arange(v_set['targets'].shape[0], device=DEV)
        sql1 = torch.tensor(0., device=DEV)
        sql2 = torch.tensor(0., device=DEV)
        with torch.no_grad():
            while idx + batch_size < indices.shape[0]:
                batch_idx = indices[idx:idx + batch_size]
                batch = {k: v[batch_idx].to(DEV) for k, v in v_set.items()}
                # out1, out2 = net(batch['X_kv'], batch['X_q'], batch['masks'], DEV)
                out = net(
                    node_features=batch['node_features'],
                    query_idxs=batch['vnf_nodes'],
                    masks=batch['mask'],
                    adj=batch['adj']
                )
                loss += torch.sum(torch.square((batch['targets'] - out) * batch['target_weights']))
                # individual_losses = log_prob(out, torch.squeeze(batch['targets'] * 1e6, dim=-1))
                # loss += -1. * torch.sum(individual_losses)
                # sql += torch.sum(squared_loss(out, torch.squeeze(batch['targets'] * 1e6, dim=-1)))
                # loss1 = squared_loss(out1, torch.squeeze(batch['targets'] * 1e6, dim=-1))
                # loss2 = squared_loss(out2, torch.squeeze(batch['estimated_targets'] * 1e6, dim=-1))
                # loss += torch.mean(loss1) + torch.mean(loss2)
                # sql1 += torch.sum(torch.sqrt(loss1))
                # sql2 += torch.sum(torch.sqrt(loss2))
                idx += batch_size
                count += 1
        if torch.isnan(loss):
            print("Val Loss is NaN, abort")
            break
        val_losses.append(loss.item() / count)
        # squared_losses.append((sql1 / idx, sql2 / idx))
        if val_losses[-1] < best_val_loss:
            # print(f"{i:4d}\t{train_losses[-1]:12.4f}\t{val_losses[-1]:12.4f}\t{squared_losses[-1][0]:12.4f}\t{squared_losses[-1][1]:12.4f}")
            print(f"{i:4d}\t{train_losses[-1]:12.4f}\t{val_losses[-1]:12.4f}")
            best_val_loss = val_losses[-1]
            torch.save(
                net.cpu().state_dict(),
                '/opt/projects/vnf-perf-prediction/models/gat-model-2.pkl'
            )
            best_iter = i
            net = net.cuda()
        if i - best_iter > 500:
            print("Reached limit")
            break

        indices = torch.randperm(n=t_set['targets'].shape[0], device=DEV)
        idx = 0
        loss = torch.tensor(0., device=DEV)
        n_batches = 0
        while idx + batch_size < indices.shape[0]:
            n_batches += 1
            batch_idx = indices[idx:idx + batch_size]
            loss += step(optimizer, net, {k: v[batch_idx].to(DEV) for k, v in t_set.items()})
            idx += batch_size
        if torch.isnan(loss):
            print("Train Loss is NaN, abort")
            break
        train_losses.append(loss / n_batches)

    return net, train_losses, val_losses


if __name__ == '__main__':
    f = h5py.File('/opt/projects/vnf-perf-prediction/data/graph-dset-train.h5', 'r')
    t_set = {k: torch.tensor(f[k][()], device='cpu') for k in f.keys()}
    # t_set['target_weights'] = torch.ones(t_set['targets'].shape[0], 3, dtype=torch.float32, device=DEV)
    f.close()
    f = h5py.File('/opt/projects/vnf-perf-prediction/data/graph-dset-val.h5', 'r')
    v_set = {k: torch.tensor(f[k][()], device='cpu') for k in f.keys()}
    # v_set['target_weights'] = torch.ones(t_set['targets'].shape[0], 3, dtype=torch.float32, device=DEV)
    f.close()

    net = StackedGatPerfPlayerModel().to(DEV)
    net, train_losses, val_losses = train(t_set, v_set, net, n_epochs=500000, batch_size=1000)
    # out = model(
    #     node_features=torch.tensor(d['node_features']),
    #     query_idxs=torch.tensor(d['vnf_nodes']),
    #     masks=torch.tensor(d['mask']),
    #     adj=torch.tensor(d['adj'])
    # )


    # data = load_data_set()
    # print("Load dataset")
    # t_set, v_set = train_val(data)
    # # t_set, v_set = train_val([])
    # t_set = {k: torch.tensor(v, device=DEV) for k, v in t_set.items()}
    # v_set = {k: torch.tensor(v, device=DEV) for k, v in v_set.items()}
    # print("Build Model")
    # net = PerfPlayerModel().cuda()
    # print("Train network")
    # net, train_losses, val_losses = train(t_set, v_set, net, n_epochs=500000, batch_size=10000)
