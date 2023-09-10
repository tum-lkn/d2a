from __future__ import annotations

import multiprocessing
import os

import pandas as pd
import numpy as np
import json
import networkx as nx
from typing import List, Tuple, Dict, Any
import logging

import dataprep.utils as dutils


logger = logging.getLogger('graph.py')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler('/opt/project/graph.log', mode='w'))


class GraphHelper(object):

    def __init__(self, sfcs: List[dutils.Sfc], machines: List[dutils.Machine], num_tx: int):
        """
        Helper class for the environments. Constructs a base graph from the initial
        problem data.

        Args:
            sfcs: The SFCs returned from the problem generator.
            machines: List of machines, the all_machines attribute of a environment
                class.
            num_tx: The number of tx threads.
        """
        self.sfcs = sfcs
        self.machines = machines[num_tx:]
        self.txs = [dutils.TxThread(i) for i in range(num_tx)]
        self.job2machine = {}
        self.job2tx = {}

        first_ids = []
        tmp = []
        weights = []
        total_rate = 0
        idx = 2
        for i, sfc in enumerate(sfcs):
            total_rate += sfc.jobs[0].rate()
            weights.append(sfc.weight)
            first_ids.append(idx)
            for j, job in enumerate(sfc.jobs):
                tmp.append((i, j, job))
                idx += 1

        self.gi = dutils.GeneralInfo(1, 10, '', '', weights, first_ids, total_rate, 64, num_tx)

        tx_load = np.zeros(len(self.txs))
        tmp.sort(key=lambda x: x[2].rate(), reverse=True)
        for _, _, job in tmp:
            idx = np.argmin(tx_load)
            tx_load[idx] += job.rate()
            self.job2tx[job] = self.txs[idx]
        tmp.sort(key=lambda x: x[2].demand)
        self.job_queue = tmp
        self.graph = _make_graph(self.gi, self.machines, self.sfcs,
                                 self.job2machine, self.job2tx, self.txs, None)
        self.name2idx = {d['name']: k for k, d in self.graph.nodes(data=True)}

    def add_edge(self, step: int, action: int):
        sfc_num, job_num, job = self.job_queue[step]
        cpu_name = f'cpu_{action + len(self.txs) + 1 + 1}'
        vnf_name = f'vnf_{sfc_num}_{job_num}'
        cpu_idx = self.name2idx[cpu_name]
        vnf_idx = self.name2idx[vnf_name]
        self.graph.add_edge(cpu_idx, vnf_idx)
        _add_cpu_metadata(self.graph, cpu_idx)
        _add_ratios(self.graph, cpu_idx)

    def get_features(self, machine_physical_id: int) -> Dict[str, int | np.array]:
        cpu_idx = self.name2idx[f'cpu_{machine_physical_id}']
        features = convert_graph(self.graph)
        features['cpu_idx'] = cpu_idx
        return features


def _make_vnf2name(sfcs: List[dutils.Sfc]) -> Dict[dutils.Job, str]:
    vnf2name = {}
    for sfc_num, sfc in enumerate(sfcs):
        for vnf_num, job in enumerate(sfc.jobs):
            vnf_name = f'vnf_{sfc_num}_{vnf_num}'
            vnf2name[job] = vnf_name
    return vnf2name


def _add_object(graph: nx.Graph, num_elements: int, idx_offset: int,
                node_type: str) -> nx.Graph:
    for i in range(num_elements):
        graph.add_node(
            i + idx_offset,
            node_type=node_type,
            index=i + idx_offset,
            name=f'{node_type}_{i:d}'
        )
    return graph


def add_socket_nodes(graph: nx.Graph, num_sockets: int, idx_offset: int=0) -> nx.Graph:
    return _add_object(
        graph=graph,
        num_elements=num_sockets,
        idx_offset=idx_offset,
        node_type='socket'
    )


def add_cpu_nodes(graph: nx.Graph, num_cores: int, idx_offset: int=0) -> nx.Graph:
    return _add_object(
        graph=graph,
        num_elements=num_cores,
        idx_offset=idx_offset,
        node_type='cpu'
    )


def add_tx_nodes(graph: nx.Graph, num_tx_threads: int, idx_offset: int=0) -> nx.Graph:
    return _add_object(
        graph=graph,
        num_elements=num_tx_threads,
        idx_offset=idx_offset,
        node_type='tx_thread'
    )


def add_divider(graph: nx.Graph) -> nx.Graph:
    graph.add_node(
        graph.number_of_nodes(),
        node_type='vnf',
        index=graph.number_of_nodes(),
        name='divider',
        position=0
    )
    return graph


def add_ports(graph: nx.Graph) -> nx.Graph:
    graph.add_node(
        graph.number_of_nodes(),
        node_type='port',
        index=graph.number_of_nodes(),
        name='out_port_0'
    )
    graph.add_node(
        graph.number_of_nodes(),
        node_type='port',
        index=graph.number_of_nodes(),
        name='in_port_0'
    )
    return graph


def add_sfc_nodes(graph: nx.Graph, num_sfcs: int, idx_offset: int=0) -> nx.Graph:
    return _add_object(
        graph=graph,
        num_elements=num_sfcs,
        idx_offset=idx_offset,
        node_type='sfc'
    )


def add_vnf_of_sfc_nodes(graph: nx.Graph, num_vnfs: int, sfc_num: int, idx_offset: int) -> nx.Graph:
    for i in range(num_vnfs):
        graph.add_node(
            i + idx_offset,
            node_type='vnf',
            index=i + idx_offset,
            name=f'vnf_{sfc_num:d}_{i:d}',
            position=i + 1
        )
    return graph


def add_cpu_to_socket_edges(graph: nx.Graph, name2idx: Dict[str, int],
                            num_sockets: int, num_cpus: int) -> nx.Graph:
    for core_num in range(num_cpus):
        socket_num, _ = divmod(core_num + 1, int(24 / num_sockets))
        graph.add_edge(name2idx[f'socket_{socket_num}'], name2idx[f'cpu_{core_num}'])
    return graph


def add_tx_to_cpu_edges(graph: nx.Graph, name2idx: Dict[str, int],
                        tx_threads: List[dutils.TxThread], cpu_offset: int) -> nx.Graph:
    for tx_thread_num in range(len(tx_threads)):
        tx_thread = f'tx_thread_{tx_thread_num}'
        cpu = f'cpu_{tx_thread_num + cpu_offset}'
        graph.add_edge(name2idx[tx_thread], name2idx[cpu])
    return graph


def add_vnf_to_cpu_edges(graph: nx.Graph, name2idx: Dict[str, int],
                         cpus: List[dutils.Machine], sfcs: List[dutils.Sfc],
                         vnfs2cpus: Dict[dutils.Job, dutils.Machine],
                         num_tx: int) -> nx.Graph:
    # machine2num = {m: i for i, m in enumerate(cpus)}
    vnf2name = _make_vnf2name(sfcs)
    for job, cpu in vnfs2cpus.items():
        vnf_name = vnf2name[job]
        cpu_name = f'cpu_{cpu.physical_id - 1}'
        graph.add_edge(name2idx[cpu_name], name2idx[vnf_name])
    graph.add_edge(name2idx[f'cpu_{num_tx + 1}'], name2idx['divider'])
    return graph


def add_vnf_to_tx_edges(graph: nx.Graph, name2idx: Dict[str, int],
                        tx_threads: List[dutils.TxThread], sfcs: List[dutils.Sfc],
                        job2tx: Dict[dutils.Job, dutils.TxThread]) -> nx.Graph:
    tx2num = {tx: i for i, tx in enumerate(tx_threads)}
    vnf2name = _make_vnf2name(sfcs)
    saw_divider = False
    for job, tx_thread in job2tx.items():
        if job in vnf2name:
            vnf_name = vnf2name[job]
        elif not saw_divider:
            # Divider VNF is added to the job2tx dict but is not part of the
            # vnf2name dict. Thus, assume the missing job is the divider. Allow
            # only one missing NF, though.
            saw_divider = True
            vnf_name = 'divider'
        else:
            raise KeyError(f"Unknown job with {json.dumps(job.to_dict())}")
        tx_name = f'tx_thread_{tx2num[tx_thread]}'
        graph.add_edge(name2idx[tx_name], name2idx[vnf_name])
    return graph


def add_rx_edges(graph: nx.Graph, name2idx: Dict[str, int]) -> nx.Graph:
    graph.add_edge(name2idx['divider'], name2idx['rx_thread_0'])
    graph.add_edge(name2idx['cpu_0'], name2idx['rx_thread_0'])
    graph.add_edge(name2idx['divider'], name2idx['in_port_0'])
    return graph


def add_vnf_to_sfc_edges(graph: nx.Graph, name2idx: Dict[str, int],
                         sfcs: List[dutils.Sfc]) -> nx.Graph:
    for sfc_num, sfc in enumerate(sfcs):
        sfc_name = f'sfc_{sfc_num}'
        for vnf_num, job in enumerate(sfc.jobs):
            vnf_name = f'vnf_{sfc_num}_{vnf_num}'
            graph.add_edge(name2idx[sfc_name], name2idx[vnf_name])
    return graph


def add_vnf_edges(graph: nx.Graph, name2idx: Dict[str, int],
                  sfcs: List[dutils.Sfc]) -> nx.Graph:
    for sfc_num, sfc in enumerate(sfcs):
        graph.add_edge(name2idx['divider'], name2idx[f'vnf_{sfc_num}_0'])
        for vnf_num, vnfs in enumerate(sfc.jobs):
            suc_name = 'out_port_0' if vnf_num == len(sfc.jobs) - 1 \
                else f'vnf_{sfc_num}_{vnf_num + 1}'
            vnf_name = f'vnf_{sfc_num}_{vnf_num}'
            graph.add_edge(name2idx[vnf_name], name2idx[suc_name])
    return graph


def _throughput(slice: pd.DataFrame) -> float:
    td = (slice.iloc[-1].TS - slice.iloc[0].TS)
    return slice.RX.max() / (td.seconds + td.microseconds / 1e6)


def _hard_overload(slice: pd.DataFrame) -> bool:
    return slice.RX_DROP.max() > 0


def _soft_overload(slice: pd.DataFrame, pps: int) -> bool:
    return slice.RX_Q.median() > pps * 1e-3


def add_throughput(graph: nx.Graph, stats: pd.DataFrame, sfcs: List[dutils.Sfc],
                   gi: dutils.GeneralInfo, name2idx: Dict[str, int]) -> nx.Graph:
    jobs_iids = [(job, iid) for job, iid in dutils.map_vnfs_to_iids(gi, sfcs).items()]
    jobs_iids.append(('divider', 1))
    job2name = _make_vnf2name(sfcs)
    job2name['divider'] = 'divider'
    stats = stats.set_index("INSTANCEID")
    for job, iid in jobs_iids:
        slice = stats.loc[iid]
        idx = name2idx[job2name[job]]
        graph.nodes[idx]['throughput'] = float(_throughput(slice))
    return graph


def add_soft_hard_overload(graph: nx.Graph, stats: pd.DataFrame, gi: dutils.GeneralInfo,
                           job2machine: Dict[dutils.Job, dutils.Machine],
                           sfcs: List[dutils.Sfc], machines: List[dutils.Machine],
                           name2idx: Dict[str, int]) -> nx.Graph:
    machine2name = {m: f'cpu_{m.physical_id - 1}' for i, m in enumerate(machines)}
    job2iid = dutils.map_vnfs_to_iids(gi, sfcs)
    stats = stats.set_index("INSTANCEID")
    for job, machine in job2machine.items():
        cpu_idx = name2idx[machine2name[machine]]
        if 'hard_overload' not in graph.nodes[cpu_idx]:
            graph.nodes[cpu_idx]['hard_overload'] = False
        if 'soft_overload' not in graph.nodes[cpu_idx]:
            graph.nodes[cpu_idx]['soft_overload'] = False
        graph.nodes[cpu_idx]['hard_overload'] |= bool(_hard_overload(stats.loc[job2iid[job]]))
        graph.nodes[cpu_idx]['soft_overload'] |= bool(_soft_overload(stats.loc[job2iid[job]], job.rate()))
    return graph


def add_metadata(graph: nx.Graph, sfcs: List[dutils.Sfc], name2idx: Dict[str, int]) -> nx.Graph:
    divider_throughput = 0
    for i, sfc in enumerate(sfcs):
        for j, job in enumerate(sfc.jobs):
            if j == 0:
                divider_throughput += job.rate()
            vnf_name = f'vnf_{i}_{j}'
            vnf_idx = name2idx[vnf_name]
            graph.nodes[vnf_idx]['arrival_rate'] = float(job.rate())
            graph.nodes[vnf_idx]['cost'] = float(job.vnf.compute_per_packet)
            graph.nodes[vnf_idx]['demand'] = float(job.demand)
    div_idx = name2idx['divider']
    graph.nodes[div_idx]['arrival_rate'] = float(divider_throughput)
    graph.nodes[div_idx]['cost'] = 110.
    graph.nodes[div_idx]['demand'] = 110. * float(divider_throughput)
    return graph


def _add_ratios(graph: nx.Graph, cpu_idx: int) -> None:
    load = 0.
    for v in graph.neighbors(cpu_idx):
        if graph.nodes[v]['node_type'] != 'vnf': continue
        load += graph.nodes[v]['demand']
    for v in graph.neighbors(cpu_idx):
        if graph.nodes[v]['node_type'] != 'vnf': continue
        graph.nodes[v]['ratio'] = graph.nodes[v]['demand'] / load


def add_ratios(graph: nx.Graph) -> nx.Graph:
    for u, attr in graph.nodes(data=True):
        if attr['node_type'] != 'cpu': continue
        _add_ratios(graph, u)
    return graph


def _add_cpu_metadata(graph: nx.Graph, node_idx: int):
    def acu(key, agg=None) -> float:
        if agg is None:
            agg = lambda x: 0 if len(x) == 0 else np.sum(x)
        return agg([graph.nodes[v][key] for v in nx.neighbors(graph, node_idx)
                    if graph.nodes[v]['node_type'] == 'vnf'])
    graph.nodes[node_idx]['cost'] = acu('cost')
    graph.nodes[node_idx]['arrival_rate'] = acu('arrival_rate')
    graph.nodes[node_idx]['demand'] = acu('demand')
    graph.nodes[node_idx]['min_ratio'] = acu(key='ratio', agg=lambda x: -1 if len(x) == 0 else np.min(x))
    graph.nodes[node_idx]['max_ratio'] = acu(key='ratio', agg=lambda x: -1 if len(x) == 0 else np.max(x))
    graph.nodes[node_idx]['ratio_ratio'] = acu(
        key='ratio',
        agg=lambda x: -1 if len(x) == 0 else np.abs(1. - np.min(x) / np.max(x))
    )


def add_cpu_metadata(graph: nx.Graph) -> nx.Graph:
    for n, d in graph.nodes(data=True):
        if d['node_type'] == 'cpu':
            _add_cpu_metadata(graph, n)
    return graph


def add_tx_metadata(graph: nx.Graph) -> nx.Graph:
    for n, d in graph.nodes(data=True):
        if d['node_type'] == 'tx_thread':
            l = [0]
            l.extend([graph.nodes[v]['arrival_rate'] for v in nx.neighbors(graph, n)
                      if graph.nodes[v]['node_type'] == 'vnf'])
            d['arrival_rate'] = float(np.sum(l))
    return graph


def add_sfc_latency(graph: nx.Graph, sid2name: Dict[int, int], mg_latency: None | pd.DataFrame) -> nx.Graph:
    if mg_latency is None: return graph
    for sid, vals in mg_latency.groupby('fid'):
        avg_latency = np.sum(vals.loc[:, 'latency'].values * vals.loc[:, 'count']) / vals.loc[:, 'count'].sum()
        name = sid2name[int(sid)]
        graph.nodes[name]['latency'] = avg_latency * 1e-6
    return graph


def add_sfc_throughput(graph: nx.Graph, sid2name: Dict[int, int], mg_throughput: None | pd.DataFrame) -> nx.Graph:
    if mg_throughput is None: return graph
    for sid, vals in mg_throughput.groupby('iid'):
        if sid == 'all': continue
        avg_packet_rate = vals.PacketRate.mean()
        name = sid2name[int(sid)]
        graph.nodes[name]['throughput'] = avg_packet_rate * 1e6
    return graph


def add_sfc_metadata(graph: nx.Graph, gi: dutils.GeneralInfo,
                     mg_thoughput: None | pd.DataFrame, mg_latency: None | pd.DataFrame) -> nx.Graph:
    sfc_nodes = [(n, d['name']) for n, d in graph.nodes(data=True) if d['node_type'] == 'sfc']
    sfc_nodes.sort(key=lambda x: int(x[1].split('_')[-1]))
    sid2name = {gi.first_ids[i]: idx for i, (idx, name) in enumerate(sfc_nodes)}
    return add_sfc_latency(add_sfc_throughput(graph, sid2name, mg_thoughput), sid2name, mg_latency)


def _make_graph(gi: dutils.GeneralInfo, machines: List[dutils.Machine],
                sfcs: List[dutils.Sfc], job2machine: Dict[dutils.Job, dutils.Machine],
                job2tx: Dict[dutils.Job, dutils.TxThread], tx_threads: List[dutils.TxThread],
                stats: None | pd.DataFrame, mg_latency: None | pd.DataFrame,
                mg_throughput: None | pd.DataFrame) -> nx.Graph:
    graph = nx.DiGraph()

    # Add nodes.
    add_socket_nodes(graph, 2, 0)
    add_tx_nodes(graph, len(tx_threads), graph.number_of_nodes())
    _add_object(graph, 1, graph.number_of_nodes(), 'rx_thread')
    add_cpu_nodes(graph, len(machines) + 2 + len(tx_threads), graph.number_of_nodes())
    add_sfc_nodes(graph, len(sfcs), graph.number_of_nodes())
    add_divider(graph)
    for i, sfc in enumerate(sfcs):
        add_vnf_of_sfc_nodes(graph, len(sfc.jobs), i, graph.number_of_nodes())
    add_ports(graph)
    name2idx = {d['name']: n for n, d in graph.nodes(data=True)}

    # Add edges.
    add_cpu_to_socket_edges(graph, name2idx, 2, len(machines) + len(tx_threads) + 2)
    add_tx_to_cpu_edges(graph, name2idx, tx_threads, 1)
    add_rx_edges(graph, name2idx)
    add_vnf_to_cpu_edges(graph, name2idx, machines, sfcs, job2machine, len(tx_threads))
    add_vnf_to_tx_edges(graph, name2idx, tx_threads, sfcs, job2tx)
    add_vnf_to_sfc_edges(graph, name2idx, sfcs)
    add_vnf_edges(graph, name2idx, sfcs)

    # Add metadata
    if stats is not None:
        add_throughput(graph, stats, sfcs, gi, name2idx)
        add_soft_hard_overload(graph, stats, gi, job2machine, sfcs, machines, name2idx)
    add_metadata(graph, sfcs, name2idx)
    add_ratios(graph)  # Has to come after add_metadata
    add_cpu_metadata(graph)
    add_tx_metadata(graph)
    add_sfc_metadata(graph, gi, mg_throughput, mg_latency)

    # gg = _replace_indices_with_names(graph)
    return graph


def make_graph(exp_dir: str) -> nx.Graph:
    stats = dutils.slice_experiment_period(dutils.load_vnf_stats(exp_dir))
    mg_latency = dutils.load_moongen_latency(exp_dir)
    mg_thoughput = dutils.load_moongen_thoughput(exp_dir)
    gi, machines, sfcs, job2machine, job2tx, tx_threads = dutils.problem_from_experiment(exp_dir)
    return _make_graph(gi, machines, sfcs, job2machine, job2tx, tx_threads, stats, mg_latency, mg_thoughput)


def _replace_indices_with_names(g: nx.Graph) -> nx.Graph:
    gg = nx.Graph()
    idx2name = {i: d['name'] for i, d in g.nodes(data=True)}
    for n, d in g.nodes(data=True):
        gg.add_node(idx2name[n], **d)
    for n1, n2 in g.edges():
        gg.add_edge(idx2name[n1], idx2name[n2])
    return gg


def extract_graph(exp_dir: str) -> Dict[str, int] | None:
    ret = None
    try:
        r, n = os.path.split(exp_dir)
        r2, n2 = os.path.split(r)
        path = f'/opt/project/data/nas/graphs-ma-diederich/{n2}-{n}.json'
        graph = make_graph(exp_dir)
        dutils.save_graph(graph, path, exp_dir)
        ret = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'max_degree': np.max(list(dict(nx.degree(graph)).values()))
        }
    except Exception as e:
        logger.error(f"Error while extracting graph for trial {exp_dir}")
        logger.exception(e)
    finally:
        return ret


def extract_graphs():
    # folders = [
    #     "/opt/project/data/nas/Random_1",
    #     "/opt/project/data/nas/8SFCs_40NFs_1",
    #     '/opt/project/data/nas/8SFCs_40NFs_Random_1',
    #     "/opt/project/data/nas/BLTest_2"
    # ]
    folders = []
    d = '/opt/project/data/nas/dqn_assignments_result/'
    for f in os.listdir(d):
        folders.append(os.path.join(d, f))
    num_processes = 16
    files = []
    for d in folders:
        for f in os.listdir(d):
            files.append(os.path.join(d, f))
    pool = multiprocessing.Pool(processes=num_processes)
    res = pool.map(extract_graph, files)
    pool.close()
    max_nodes = np.max([r['num_nodes'] for r in res if r is not None])
    max_edges = np.max([r['num_edges'] for r in res if r is not None])
    max_degree = np.max([r['max_degree'] for r in res if r is not None])
    print(f"max_nodes: {max_nodes}, max_edges: {max_edges}, max_degree: {max_degree}")


def convert_graph(g: nx.Graph, normalize: bool = True) -> None | Dict[str, np.array]:
    def std(v: float, k: str) -> float:
        min_v = stats[f'min_{k}']
        max_v = stats[f'max_{k}']
        if normalize:
            return (v - min_v) / (max_v - min_v)
        else:
            return v

    stats = {
        'num_nodes': 68,
        'num_edges': 160,
        'max_degree': 15 + 2,
        'max_rate': 2500000.0,
        'min_rate': 35548.0,
        'max_cost': 7671.0,
        'min_cost': 110.0,
        'max_demand': 1539957120.0,
        'min_demand': 8814925.0,
        'max_rate_cpu': 7500000.0,
        'min_rate_cpu': 35548.0,
        'max_cost_cpu': 26305.0,
        'min_cost_cpu': 110.0,
        'max_demand_cpu': 3078128826.0,
        'min_demand_cpu': 17778972.0,
        'min_rate_tx': 454380.0,
        'max_rate_tx': 7500000.0,
        'soft_failures': 4538,
        'num_soft_failures': 20074,
        'hard_failures': 4214,
        'num_hard_failures': 20074,
        'min_throughput': 27792.95901067119,
        'max_throughput': 2349759.0518103624,
        'min_num_vnfs_on_cpu_core': 0,
        'max_num_vnfs_on_cpu_core': 14
    }

    w_soft = lambda x: 1. / (stats['soft_failures'] / stats['num_soft_failures']) if x else \
            1. / ((stats['num_soft_failures'] - stats['soft_failures']) / stats['num_soft_failures'])
    w_hard = lambda x: 1. / (stats['hard_failures'] / stats['num_hard_failures']) if x else \
        1. / ((stats['num_hard_failures'] - stats['hard_failures']) / stats['num_hard_failures'])
    dim_pos_encoding = 10
    node_types_l = ['cpu', 'port', 'rx_thread', 'sfc', 'socket', 'tx_thread', 'vnf', 'numa']
    node_types = {t: np.eye(len(node_types_l), dtype=np.float32)[i, :] for i, t in enumerate(node_types_l)}
    len_node_type = len(node_types_l)
    node_ft_names = ['cpu_cost', 'cpu_rate', 'cpu_demand', 'cpu_ratio',
                     'vnf_cost', 'vnf_rate', 'vnf_demand', 'vnf_ratio',
                     'tx_rate']
    nft2idx = {name: len_node_type + dim_pos_encoding + i for i, name in enumerate(node_ft_names)}
    num_node_features = len_node_type + dim_pos_encoding + len(nft2idx)

    adj = np.zeros([1, stats['num_nodes'], stats['max_degree'], 1], dtype=np.int)
    msk = np.zeros([1, stats['num_nodes'], stats['max_degree'], 1], dtype=np.float32)
    # Count number of neighbors already added for this node.
    indices = {n: 0 for n in g.nodes()}
    for u in g.nodes():
        idx = indices[u]
        adj[0, u, idx, 0] = u
        msk[0, u, idx, 0] = 1
        indices[u] += 1

    for u, v in g.edges():
        idx = indices[u]
        adj[0, u, idx, 0] = v
        msk[0, u, idx, 0] = 1
        indices[u] += 1
        idx = indices[v]
        adj[0, v, idx, 0] = u
        msk[0, v, idx, 0] = 1
        indices[v] += 1

    node_features = np.zeros([1, stats['num_nodes'], num_node_features], dtype=np.float32)
    vnf_nodes = []
    cpu_nodes = []
    tx_nodes = []
    sfc_nodes = []
    targets_throughput = []
    targets_soft_overload = []
    targets_hard_overload = []
    targets_sfc_latency = []
    targets_sfc_throughput = []
    weights_soft_overload = []
    weights_hard_overload = []
    for i, (u, attrs) in enumerate(g.nodes(data=True)):
        node_features[0, u, :len_node_type] = node_types[attrs['node_type']]
        if 'position' in attrs:
            node_features[0, u, len_node_type:len_node_type + dim_pos_encoding] = dutils.positional_encoding(
                pos=attrs['position'],
                dim=dim_pos_encoding
            )
        if attrs['node_type'] == 'vnf':
            node_features[0, u, nft2idx['vnf_ratio']] = attrs['ratio']
            node_features[0, u, nft2idx['vnf_rate']] = std(attrs['arrival_rate'], 'rate')
            node_features[0, u, nft2idx['vnf_cost']] = std(attrs['cost'], 'cost')
            node_features[0, u, nft2idx['vnf_demand']] = std(attrs['demand'], 'demand')
            vnf_nodes.append(u)
            targets_throughput.append(np.array([[attrs['throughput']]]))
        elif attrs['node_type'] == 'tx_thread' and attrs['arrival_rate'] > 0:
            node_features[0, u, nft2idx['tx_rate']] = std(attrs['arrival_rate'], 'rate_tx')
            tx_nodes.append(u)
        elif attrs['node_type'] == 'cpu' and 'soft_overload' in attrs:
            # node_features[0, u, -4] = standardize(
            #     float(np.sum([1 if g.nodes[m]['node_type'] == 'vnf' else 0 for m in g.neighbors(u)])),
            #     'num_vnfs_on_cpu_core'
            # )
            node_features[0, u, nft2idx['cpu_ratio']] = attrs['ratio_ratio']
            node_features[0, u, nft2idx['cpu_rate']] = std(attrs['arrival_rate'], 'rate_cpu')
            node_features[0, u, nft2idx['cpu_cost']] = std(attrs['cost'], 'cost_cpu')
            node_features[0, u, nft2idx['cpu_demand']] = std(attrs['demand'], 'demand_cpu')
            cpu_nodes.append(u)
            targets_soft_overload.append(np.array([[0, 1]], dtype=np.float32) if attrs['soft_overload']
                                         else np.array([[1, 0]], dtype=np.float32))
            weights_soft_overload.append(w_soft(attrs['soft_overload']))
            targets_hard_overload.append(np.array([[0, 1]], dtype=np.float32) if attrs['hard_overload']
                                         else np.array([[1, 0]], dtype=np.float32))
            weights_hard_overload.append(w_hard(attrs['hard_overload']))
        elif attrs['node_type'] == 'sfc' and 'latency' in attrs:
            sfc_nodes.append(u)
            targets_sfc_latency.append(attrs['latency'])
            targets_sfc_throughput.append(attrs['throughput'])
    vnf_nodes = np.array(vnf_nodes, dtype=np.int64)
    cpu_nodes = np.array(cpu_nodes, dtype=np.int64)
    tx_nodes = np.array(tx_nodes, dtype=np.int64)
    sfc_nodes = np.array(sfc_nodes, dtype=np.int64)
    weights_soft_overload = np.array(weights_soft_overload, dtype=np.float32)
    weights_hard_overload = np.array(weights_hard_overload, dtype=np.float32)
    targets_throughput = np.concatenate(targets_throughput, axis=0)
    targets_soft_overload = np.concatenate(targets_soft_overload, axis=0)
    targets_hard_overload = np.concatenate(targets_hard_overload, axis=0)
    return {
        'nodes_vnf': vnf_nodes,
        'nodes_cpu': cpu_nodes,
        'nodes_tx': tx_nodes,
        'nodes_sfc': sfc_nodes,
        'adj': adj,
        'mask': msk,
        'node_features': node_features,
        'weights_hard_overload': weights_hard_overload,
        'weights_soft_overload': weights_soft_overload,
        'targets_throughput': targets_throughput,
        'targets_soft_overload': targets_soft_overload,
        'targets_hard_overload': targets_hard_overload,
        'targets_sfc_latency': targets_sfc_latency,
        'targets_sfc_throughput': targets_sfc_throughput
    }


def extract_features(path: str) -> None | Dict[str, np.array]:
    features = None
    try:
        graph = dutils.load_graph(path)
        features = convert_graph(graph)
    except Exception as e:
        logger.error(f"Error during converting graph in {path}")
        logger.exception(e)
    return features


def accumulate_samples(results: List[Dict[str, np.array]]) -> Dict[str, np.array]:
    count = 0
    data_set = {
        "indices_vnf_throughput": np.array([], dtype=np.int64),
        "indices_tx_throughput": np.array([], dtype=np.int64),
        "indices_any_cpu_overload": np.array([], dtype=np.int64),
        "indices_sfc_throughput": np.array([], dtype=np.int64),
        "indices_sfc_latency": np.array([], dtype=np.int64)
    }
    for res in results:
        if res is None: continue
        for k, v in res.items():
            data_set[k] = v if count == 0 else np.concatenate([data_set[k], v], axis=0)
        data_set['indices_vnf_throughput'] = np.concatenate([
            data_set['indices_vnf_throughput'],
            np.repeat(count, repeats=res['nodes_vnf'].size).astype(np.int64)
        ])
        data_set['indices_tx_throughput'] = np.concatenate([
            data_set['indices_tx_throughput'],
            np.repeat(count, repeats=res['nodes_tx'].size).astype(np.int64)
        ])
        data_set['indices_any_cpu_overload'] = np.concatenate([
            data_set['indices_any_cpu_overload'],
            np.repeat(count, repeats=res['nodes_cpu'].size).astype(np.int64)
        ])
        data_set['indices_sfc_throughput'] = np.concatenate([
            data_set['indices_sfc_throughput'],
            np.repeat(count, repeats=res['nodes_sfc'].size).astype(np.int64)
        ])
        data_set['indices_sfc_latency'] = np.concatenate([
            data_set['indices_sfc_latency'],
            np.repeat(count, repeats=res['nodes_sfc'].size).astype(np.int64)
        ])
        count += 1
    return data_set


def accumulate_files(files: List[str], path: str):
    for f in os.listdir(path):
        p = os.path.join(path, f)
        if f.find('-DT') >= 0:
            continue
        if os.path.isdir(p):
            accumulate_files(files, p)
        if p.endswith('.json'):
            files.append(p)


def dset_graph_files():
    graph_dir = '/opt/project/data/nas/graphs-ma-diederich'
    graph_dir = '/opt/project/data/nas/graphs-golden-samples'
    # graph_files = [os.path.join(graph_dir, f) for f in os.listdir(graph_dir) if f.endswith('.json')]
    graph_files = []
    accumulate_files(graph_files, graph_dir)
    random = np.random.RandomState(seed=1)
    indices = np.arange(len(graph_files))
    random.shuffle(indices)
    split = int(indices.size * 0.8)
    train_indices = indices[:split]
    val_indices = indices[split:]
    train_files = [graph_files[i] for i in train_indices]
    val_files = [graph_files[i] for i in val_indices]
    return train_files, val_files


def make_dsets():
    # pool = multiprocessing.Pool(16)
    # results_train = pool.map(extract_features, [graph_files[i] for i in train_indices])
    # results_val = pool.map(extract_features, [graph_files[j] for j in val_indices])
    # pool.close()
    train_files, val_files = dset_graph_files()
    results_train = [extract_features(f) for f in train_files]
    results_val = [extract_features(f) for f in val_files]

    results_train = accumulate_samples(results_train)
    results_val = accumulate_samples(results_val)

    # dutils.save_dset(results_train, '/opt/project/data/nas/graphs-ma-diederich/train-set.h5')
    # dutils.save_dset(results_val,   '/opt/project/data/nas/graphs-ma-diederich/val-set.h5')
    dutils.save_dset(results_train, '/opt/project/data/nas/graphs-golden-samples/train-set.h5')
    dutils.save_dset(results_val,   '/opt/project/data/nas/graphs-golden-samples/val-set.h5')


if __name__ == '__main__':
    # g = make_graph('/opt/project/data/nas/8SFCs_40NFs_Random_1/8SFCs_mpps_random-p0002-i00')
    # gg = _replace_indices_with_names(g)
    # print(gg.number_of_nodes())
    # extract_graphs()
    make_dsets()

