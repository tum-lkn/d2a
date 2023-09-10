import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any

import dataprep.utils as dutils


class GraphDataset(Dataset):
    """
        A dataset that hides the complex structure of the underlying graphs.
        The dataset has four types of attributes. Attributes that describe the
        graphs, i.e., `adj`, `mask` and `node_features`. Attributes that describe
        the target features, all attributes starting with `targets_*`. Attributes
        that return the node indices in the respective graph `nodes_*`. Attributes
        that index the graphs `indices_*`.
        The relation is as follows. The graph attributes hold the features of
        a graph, i.e., have a first dimension of `num_graphs`. For each graph,
        different targets, such as the throughput of VNFs and SFCs, the latency
        of SFCs and the overload of CPU cores exists. Since each problem has a
        varying amount of those targets, the corresponding attributes have a varying
        size. Each target is a 1D array containing the corresponding value.
        The `indices_*` attributes return for an entry in `targets_*` the index
        to the graph to which the target measure belongs, i.e., `indices_vnf_throughput[sample_idx]`
        returns the graph in which `targets_vnf_throughput[sample_idx]` is observed.

        The attributes nodes_* finally resolve the node for a specific target in
        the corresponding graph. That is, with `nodes_vnf[sample_idx]` the dataset
        obtains the node index of the vnf in the graph `indices_vnf_throughput[sample_idx]`
        for which the target value `targets_vnf_throughput[sample_idx]` belongs to.
    """

    possible_targets: List[str] = ['soft_overload', 'hard_overload', 'vnf_throughput',
                                   'sfc_throughput', 'sfc_latency']

    @classmethod
    def from_hdf5(cls, path: str, target: str, cpu_overload_class_weight: float=None):
        return cls(target=target, cpu_overload_class_weight=cpu_overload_class_weight,
                   **dutils.load_dset(path))

    def __init__(self, nodes_vnf: np.array, nodes_cpu: np.array, nodes_tx: np.array,
                 nodes_sfc: np.array, adj: np.array, mask: np.array, node_features: np.array,
                 weights_hard_overload: np.array, weights_soft_overload: np.array,
                 targets_throughput: np.array, targets_soft_overload: np.array,
                 targets_hard_overload: np.array, targets_sfc_throughput: np.array,
                 targets_sfc_latency: np.array, indices_vnf_throughput: np.array,
                 indices_tx_throughput: np.array, indices_any_cpu_overload: np.array,
                 indices_sfc_latency: np.array, indices_sfc_throughput: np.array,
                 target: str, cpu_overload_class_weight: float=None, **kwargs):
        super(GraphDataset, self).__init__()
        assert target in self.possible_targets, \
            f"Target {target} not in [{', '.join(self.possible_targets)}]"
        assert cpu_overload_class_weight is None or cpu_overload_class_weight >= 1, \
            f"cpu_overload_class_weight must be None or larger one, is {cpu_overload_class_weight}"
        self.adj = adj.astype(np.int64)
        self.mask = mask.astype(np.float32)
        self.node_features = node_features.astype(np.float32)
        self.target = target
        self.cpu_overload_class_weight = cpu_overload_class_weight

        self.nodes_vnf = nodes_vnf.astype(np.int64)
        self.nodes_cpu = nodes_cpu.astype(np.int64)
        self.nodes_tx = nodes_tx.astype(np.int64)
        self.nodes_sfc = nodes_sfc.astype(np.int64)

        if cpu_overload_class_weight is None:
            self.weights_hard_overload = weights_hard_overload.astype(np.float32)
            self.weights_soft_overload = weights_soft_overload.astype(np.float32)
        else:
            print(f"Use weights of 1 for non-tip, and {cpu_overload_class_weight} for tip classes.")
            factor = np.array([[1., cpu_overload_class_weight]])
            self.weights_soft_overload = np.sum(factor * targets_soft_overload, axis=1).astype(np.float32)
            self.weights_hard_overload = np.sum(factor * targets_hard_overload, axis=1).astype(np.float32)

        self.targets_throughput = targets_throughput.astype(np.float32)
        self.targets_soft_overload = np.clip(targets_soft_overload, 0.05, 0.95).astype(np.float32)
        self.targets_hard_overload = np.clip(targets_hard_overload, 0.05, 0.95).astype(np.float32)
        self.targets_sfc_throughput = targets_sfc_throughput.astype(np.float32)
        self.targets_sfc_latency = targets_sfc_latency.astype(np.float32)

        self.indices_vnf_throughput = indices_vnf_throughput.astype(np.int64)
        self.indices_tx_throughput = indices_tx_throughput.astype(np.int64)
        self.indices_any_cpu_overload = indices_any_cpu_overload.astype(np.int64)
        self.indices_sfc_latency = indices_sfc_latency.astype(np.int64)
        self.indices_sfc_throughput = indices_sfc_throughput.astype(np.int64)

    def todict(self):
        return {
            "adj": self.adj,
            "mask": self.mask,
            "node_features": self.node_features,
            "target": self.target,
            "nodes_vnf": self.nodes_vnf,
            "nodes_cpu": self.nodes_cpu,
            "nodes_tx": self.nodes_tx,
            "nodes_sfc": self.nodes_sfc,
            "weights_hard_overload": self.weights_hard_overload,
            "weights_soft_overload": self.weights_soft_overload,
            "targets_throughput": self.targets_throughput,
            "targets_soft_overload": self.targets_soft_overload,
            "targets_hard_overload": self.targets_hard_overload,
            "targets_sfc_throughput": self.targets_sfc_throughput,
            "targets_sfc_latency": self.targets_sfc_latency,
            "indices_vnf_throughput": self.indices_vnf_throughput,
            "indices_tx_throughput": self.indices_tx_throughput,
            "indices_any_cpu_overload": self.indices_any_cpu_overload,
            "indices_sfc_latency": self.indices_sfc_latency,
            "indices_sfc_throughput": self.indices_sfc_throughput,
            "cpu_overload_class_weight": self.cpu_overload_class_weight
        }

    def __len__(self) -> int:
        return {
            "soft_overload": self.targets_soft_overload.shape[0],
            "hard_overload": self.targets_hard_overload.shape[0],
            "vnf_throughput": self.targets_throughput.shape[0],
            "sfc_throughput": self.targets_sfc_throughput.shape[0],
            "sfc_latency": self.targets_sfc_latency.shape[0]
        }[self.target]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        graph_idx = {
            'soft_overload': self.indices_any_cpu_overload,
            'hard_overload': self.indices_any_cpu_overload,
            'vnf_throughput': self.indices_vnf_throughput,
            'sfc_throughput': self.indices_sfc_throughput,
            'sfc_latency': self.indices_sfc_latency
        }[self.target][idx]
        target = {
            'soft_overload': self.targets_soft_overload,
            'hard_overload': self.targets_hard_overload,
            'vnf_throughput': self.targets_throughput,
            'sfc_throughput': self.targets_sfc_throughput,
            'sfc_latency': self.targets_sfc_latency
        }[self.target][idx]
        if self.target == 'sfc_latency':
            target = np.log(target)
        weight = {
            'soft_overload': lambda idx: self.weights_soft_overload[idx],
            'hard_overload': lambda idx: self.weights_hard_overload[idx],
            'vnf_throughput': lambda idx: np.array([1.], dtype=np.float32)[0],
            'sfc_throughput': lambda idx: np.array([1.], dtype=np.float32)[0],
            'sfc_latency': lambda idx: np.array([1.], dtype=np.float32)[0]
        }[self.target](idx)
        query_idx = {
            'soft_overload': self.nodes_cpu,
            'hard_overload': self.nodes_cpu,
            'vnf_throughput': self.nodes_vnf,
            'sfc_throughput': self.nodes_sfc,
            'sfc_latency': self.nodes_sfc
        }[self.target][idx]
        return {
            'node_features': self.node_features[graph_idx],
            'query_idx': query_idx,
            'masks': self.mask[graph_idx],
            'adj': self.adj[graph_idx],
            'target': target,
            'weight': weight
        }

