from __future__ import annotations

import h5py
import pandas as pd
import os
import numpy as np
import networkx as nx
import json
from typing import List, Tuple, Dict, Any


def exists_vnf_stats(exp_dir: str) -> bool:
    return os.path.exists(os.path.join(exp_dir, '!VNF_stats.txt'))


def load_vnf_stats(exp_dir: str) -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(exp_dir, '!VNF_stats.txt'),
        sep=';'
    )
    df.loc[:, 'TS'] = pd.to_datetime(df.TS.values, unit='ms')
    return df


def get_exp_start(results: pd.DataFrame) -> int:
    df_tmp = results.set_index('INSTANCEID').loc[1, :]
    df_tmp = df_tmp.loc[df_tmp.RX.values == 0, :]
    return df_tmp.iloc[-1]['TS']


def get_exp_end(results: pd.DataFrame) -> int:
    df_tmp = results.set_index('INSTANCEID').loc[1, :]
    df_tmp = df_tmp.loc[df_tmp.RX.values == df_tmp.iloc[-1, :]['RX'], :]
    return df_tmp.iloc[0]['TS']


def slice_experiment_period(results) -> pd.DataFrame:
    start_ts = get_exp_start(results)
    end_ts = get_exp_end(results)
    tmp = results.loc[results.TS.values > start_ts, :]
    tmp = tmp.loc[tmp.TS.values < end_ts, :]
    return tmp


def compute_detailed_cost(results: pd.DataFrame) -> Dict[int, pd.Series]:
    costs = {}
    txs = {}
    rxs = {}
    for iid, group in results.groupby("INSTANCEID"):
        rx = np.clip(group.RX.values[1:] - group.RX.values[:-1], 1, 1e12).astype(np.float32)
        tx = np.clip(group.TX.values[1:] - group.TX.values[:-1], 1, 1e12).astype(np.float32)
        cost = group.COST.values[1:] - group.COST.values[:-1]
        rel_cost = cost.astype(np.float32) / tx
        costs[iid] = rel_cost
        txs[iid] = tx
        rxs[iid] = rx
    return costs, txs, rxs


class Vnf(object):
    """
    Represents an abstract base class for concrete VNF implementations.
    """
    @classmethod
    def from_dict(cls, dict):
        return Vnf(**dict)

    def __init__(self, compute_per_packet: int, instance_id: int,
                 service_id: int):
        """
        Initializes object.

        Args:
            compute_per_packet (int): Amount of cycles per packet.
        """
        self.compute_per_packet = compute_per_packet
        self.instance_id = instance_id
        self.service_id = service_id

    def to_string(self):
        return "VNF with compute {:d}".format(int(self.compute_per_packet))

    def to_dict(self):
        return {
            "compute_per_packet": int(self.compute_per_packet),
            "instance_id": self.instance_id,
            "service_id": self.service_id
        }


class Machine(object):
    """
    A machine offering computational ressources to jobs.
    """
    @classmethod
    def from_dict(cls, dict: Dict):
        return Machine(**dict)

    def __init__(self, physical_id: int, capacity: int):
        """
        Args:
            physical_id: Identifier for the physical machine.
            capacity: Cycles per second.
        """
        self.capacity = int(capacity)
        self.physical_id = physical_id

    def to_string(self):
        return "Machine on core {:d} with capacity {:d}".format(
            int(self.physical_id),
            int(self.capacity)
        )

    def to_dict(self):
        return {
            "physical_id": int(self.physical_id),
            "capacity": int(self.capacity)
        }


class Job(object):
    """
    Represents a schedulable job, i.e., a VNF in this setting here.
    """

    @classmethod
    def from_dict(cls, d) -> 'Job':
        return cls(
            rate=d['rate'],
            vnf=Vnf.from_dict(d['vnf'])
        )

    def __init__(self, rate: float, vnf: Vnf):
        """
        Initializes object.

        Args:
            rate (callable): Function that returns the rate in seconds.
            vnf (vnf_types.Vnf): Concrete VNF that has to process the arriving
                packets.

        Note:
            Rate is a callable to implement function chaining. The input rate
            to one VNF is the output rate of another VNF earlier in the chain.
            If no chains exist or a VNF is the first chain simply use a lambda
            function that returns a static value.
        """
        self._rate = rate
        self.vnf = vnf

    def rate(self):
        return self._rate

    @property
    def demand(self) -> float:
        """
        Gives the compute demand in CPU units per second.

        Returns:
            float
        """
        return self.rate() * self.vnf.compute_per_packet

    @property
    def instance_id(self) -> int:
        return self.vnf.instance_id

    def to_string(self):
        return "Job with rate {} demand {} and {}".format(
            int(self.rate()),
            int(self.demand),
            self.vnf.to_string()
        )

    def to_dict(self):
        return {
            "rate": int(self.rate()),
            "vnf": self.vnf.to_dict()
        }


class Sfc(object):
    """
    Represents a service function chain.
    """

    @classmethod
    def from_dict(cls, d) -> 'Sfc':
        return cls(
            jobs=[Job.from_dict(j) for j in d['jobs']],
            rate=d['rate'],
            weight=d['weight']
        )

    def __init__(self, jobs: List[Job], rate: int, weight: float):
        """
        Initializes object.
        Args:
            jobs: List of jobs that are chained together.
            rate: Input rate in packets per second to the SFC.
            weight: Fraction rate corresponds to oeverall rate.
        """
        self.demand = int(rate)
        self.jobs = jobs
        self.weight = weight
        self.rate = rate

    def to_string(self):
        s = "SFC with demand {:d}, weight {:.2f} and Jobs:\n".format(
            int(self.demand),
            self.weight
        )
        for job in self.jobs:
            s += job.to_string() + "\n"
        return s

    def to_dict(self):
        return {
            "jobs": [j.to_dict() for j in self.jobs],
            "rate": int(self.rate),
            "weight": float(self.weight)
        }


class GeneralInfo(object):

    def __init__(self, delay_in_ms: float, duration: float, logging_dir: str,
                 result_dir: str, weights: List[float], first_ids: List[int],
                 rate: float, packet_size: int, num_tx: int):
        """
        General Infor object.

        Args:
            delay_in_ms: ???
            duration: Duration of the run in seconds.
            logging_dir: Direcotry where files are wirtten to.
            result_dir: DIrectory in which result files are written to.
            weights: The weights of the individual SFCs. How much traffic each
                SFC receives from the overall system input.
            first_ids: The instance IDs of the first VNFs in each chain.
            rate: The system rate in Pps.
            packet_size: THe size of the packets.
            num_tx: The number of TX threads.

        """
        self.delay_in_ms = delay_in_ms
        self.duration = duration
        self.logging_dir = logging_dir
        self.result_dir = result_dir
        self.weights = weights
        self.first_ids = first_ids
        self.rate = rate
        self.packet_size = packet_size
        self.num_tx = num_tx


class TxThread(object):

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def __init__(self, index: int, pps: float=8.4e6):
        self.index = index
        self.pps = pps
        self.measured_cost = None

    def to_dict(self):
        return {
            'index': self.index,
            'pps': self.pps,
            'measured_cost': self.measured_cost
        }


def map_vnfs_to_iids(gi: GeneralInfo, sfcs: List[Sfc]) -> Dict[Job, int]:
    iid_to_jobs = {}
    for i, sfc in enumerate(sfcs):
        first_id = gi.first_ids[i]
        for j, vnf in enumerate(sfc.jobs):
            iid_to_jobs[vnf] = first_id + j
    return iid_to_jobs


def map_iid_to_sfc(sfcs: List[Sfc], gi: GeneralInfo) -> Dict[int, int]:
    iid_to_sfc = {}
    for i, sfc in enumerate(sfcs):
        for j, job in enumerate(sfc.jobs):
            iid_to_sfc[gi.first_ids[i] + j] = i
    return iid_to_sfc


def map_iid_to_vnf_chain_index(sfcs: List[Sfc], gi: GeneralInfo) -> Dict[int, int]:
    iid_to_vnf_idx = {}
    for i, sfc in enumerate(sfcs):
        for j, job in enumerate(sfc.jobs):
            iid_to_vnf_idx[gi.first_ids[i] + j] = j
    return iid_to_vnf_idx


def sfcs_from_run_config_and_problem(general_info: GeneralInfo, problem: Dict) -> List[Sfc]:
    """
    Crate the SFCs used in a particular experiment. The run_config provides rate
    and weights, the problem the cycle counts of VNFs
    """
    pps = general_info.rate

    sfcs = []
    count = 0
    for i, vnf_list in enumerate(problem['problem']):
        sfc_rate = int(general_info.weights[i] * pps)
        # assert sfc_rate == vnf_list[0]['rate'], 'Calculated and stored rate do not match.' + \
        #     f' calculated is {sfc_rate}, stored is {vnf_list[0]["rate"]}.' + \
        #     f' total rate is {general_info.rate}'
        vnfs = [Job(
            rate=sfc_rate,
            vnf=Vnf(
                compute_per_packet=d['vnf']['compute_per_packet'],
                service_id=general_info.first_ids[i] + j,
                instance_id=2 + j + count
            )
        ) for j, d in enumerate(vnf_list)]
        count += len(vnf_list)
        sfcs.append(Sfc(vnfs, sfc_rate, general_info.weights[i]))
    return sfcs


def machines_from_problem(problem: Dict) -> List[Machine]:
    """
    Create machines from the saved problem instance.
    """
    if 'machines' in problem:
        return [Machine(**d) for d in problem['machines']]
    else:
        return [Machine(i, int(2.2e9)) for i in range(8, 12)]


def machines_from_general_info(general_info: GeneralInfo) -> List[Machine]:
    return [Machine(i, int(2.2e9)) for i in range(general_info.num_tx + 3, 24)]


def machines_from_problem_dict(machines: List[Dict[str, Any]]) -> List[Machine]:
    return [Machine(d['physical_id'], d['capacity']) for d in machines]


def placement_from_problem(machines: List[Machine], sfcs: List[Sfc],
                           problem: dict, offset: int) -> Dict[Job, Machine]:
    """
    Reconstruct the placement that was used in the given experimentm
    """
    placement = {}
    for pl in problem['placement']:
        placement[sfcs[pl['sfc']].jobs[pl['job']]] = machines[pl['machine'] - offset]
    return placement


def make_tx_assignment(general_info: GeneralInfo, sfcs: List[Sfc], threads: List[TxThread],
                       tx_assignment: Dict[str, List[int]]) -> Dict[Job, TxThread]:
    # Initialize with the divider NF which is needed in this case here.
    jobs = [Job(general_info.rate, Vnf(40, 1, 1))]
    for sfc in sfcs:
        jobs.extend([j for j in sfc.jobs])
    jobs = {j.instance_id: j for j in jobs}
    assignment = {}
    for index, instance_ids in tx_assignment.items():
        thread = threads[int(index)]
        for instance_id in instance_ids:
            job = jobs[instance_id]
            assignment[job] = thread
    return assignment


def problem_from_experiment(exp_folder: str) -> Tuple[GeneralInfo, List[Machine],
                                                      List[Sfc], Dict[Job, Machine],
                                                      Dict[Job, TxThread], List[TxThread]]:
    """
    Create the complete problem instance from an experiment folder. That is,
    the machines, SFCs and the placement as well as run_config.
    """
    with open(os.path.join(exp_folder, 'problem.json'), 'r') as fh:
        problem = json.load(fh)
    with open(os.path.join(exp_folder, 'config', 'automator.json'), 'r') as fh:
        data = json.load(fh)
        general_info = GeneralInfo(**data['General'])
        tx_assignment = data['Manager']['tx_assignment']
    general_info.num_tx = len(tx_assignment)
    machines = machines_from_general_info(general_info)
    # Use len(tx_assignment) instead of general_info.num_tx. That attribute
    # is not set for some runs.
    threads = [TxThread(i) for i in range(len(tx_assignment))]
    sfcs = sfcs_from_run_config_and_problem(general_info, problem)
    placement = placement_from_problem(machines, sfcs, problem, 24 - len(machines))
    tx_assignment_ = make_tx_assignment(general_info, sfcs, threads, tx_assignment)
    return general_info, machines, sfcs, placement, tx_assignment_, threads


def load_intel_processor_counter_monitor(exp_folder: str) -> Dict[str, pd.DataFrame]:
    """
    Load the intel processor counter monitor statistics files. Split the huge
    file into smaller chunks.

    :param exp_folder:
    :return:
    """
    with open(os.path.join(exp_folder, 'pcm.csv'), 'r') as fh:
        components = fh.readline().split(',')
    sections = {}
    # Exclude the new line character.
    for i, component in enumerate(components[:-1]):
        if component not in sections:
            sections[component] = []
        sections[component].append(i)
    all_data = pd.read_csv(os.path.join(exp_folder, 'pcm.csv'), skiprows=1)
    ret = {k: all_data.iloc[:, v] for k, v in sections.items()}
    return ret


def save_graph(graph: nx.Graph, path: str, exp_dir: str) -> None:
    with open(path, 'w') as fh:
        json.dump(
            obj={
                'graph': nx.jit_data(graph),
                'path': exp_dir
            },
            fp=fh
        )


def load_graph(path: str) -> nx.Graph:
    with open(path, 'r') as fh:
        data = json.load(fh)
    g = nx.jit_graph(data['graph'])
    g.graph['path'] = data['path'] if 'path' in data else None
    return g


def positional_encoding(pos: int, dim: int) -> float:
    k = np.repeat(np.arange(dim / 2), repeats=2)
    mask = np.tile(np.array([0., 1.]), reps=int(dim/2))
    w = 1. / 10000 ** (2 * k / dim)
    return (1 - mask) * np.cos((pos + 1) * w) + mask * np.sin((pos + 1) * w)


def save_dset(dset: Dict[str, np.array], path: str) -> None:
    f = h5py.File(path, 'w')
    for k, v in dset.items():
        f.create_dataset(name=k, data=v)
    f.close()


def load_dset(path: str) -> Dict[str, np.array]:
    f = h5py.File(path, 'r')
    dset = {}
    for k in f.keys():
        dset[k] = f[k][()]
    f.close()
    return dset


def load_moongen_thoughput(exp_dir: str) -> pd.DataFrame:
    rx_files = [f for f in os.listdir(exp_dir) if f.startswith('RX-')]
    dfs = []
    for rx_file in rx_files:
        parts = rx_file.split('.')[0].split('-')
        iid = parts[1] if parts[1] == 'all' else parts[2]
        df = pd.read_csv(os.path.join(exp_dir, rx_file), sep=',')
        df['iid'] = iid
        dfs.append(df)
    return pd.concat(dfs, axis=0)


def load_moongen_latency(exp_dir: str) -> pd.DataFrame:
    files = [f for f in os.listdir(exp_dir) if f.startswith('Latency-SFC')]
    dfs = []
    for f in files:
        fid = int(f.rstrip('.csv').split('-')[-1])
        dfs.append(pd.read_csv(os.path.join(exp_dir, f), sep=',', names=['latency', 'count']))
        dfs[-1]['fid'] = fid
    return pd.concat(dfs, axis=0)


def _analyze_graphs(graphs: List[nx.Graph]) -> Dict[str, float]:
    stats = []
    for g in graphs:
        costs = [d['cost'] for _, d in g.nodes(data=True) if 'cost' in d and d['node_type'] == 'vnf']
        ars = [d['arrival_rate'] for _, d in g.nodes(data=True) if 'arrival_rate' in d and d['node_type'] == 'vnf']
        demand = [d['demand'] for _, d in g.nodes(data=True) if 'demand' in d and d['node_type'] == 'vnf']
        ars_tx = np.array(
            [d['arrival_rate'] for _, d in g.nodes(data=True) if 'arrival_rate' in d and d['node_type'] == 'tx_thread'])
        costs_cpu = np.array([d['cost'] for _, d in g.nodes(data=True) if 'cost' in d and d['node_type'] == 'cpu'])
        ars_cpu = np.array(
            [d['arrival_rate'] for _, d in g.nodes(data=True) if 'arrival_rate' in d and d['node_type'] == 'cpu'])
        demand_cpu = np.array([d['demand'] for _, d in g.nodes(data=True) if 'demand' in d and d['node_type'] == 'cpu'])
        thp = [d['throughput'] for _, d in g.nodes(data=True) if 'throughput' in d and d['node_type'] == 'vnf']
        sos = [d['soft_overload'] for _, d in g.nodes(data=True) if 'soft_overload' in d]
        hos = [d['hard_overload'] for _, d in g.nodes(data=True) if 'hard_overload' in d]
        n_vnfs_on_cpu = []
        for n in g.nodes():
            if g.nodes[n]['node_type'] != 'cpu': continue
            count = 0
            for m in nx.neighbors(g, n):
                if g.nodes[m]['node_type'] == 'vnf':
                    count += 1
            n_vnfs_on_cpu.append(count)
        stats.append({
            'num_nodes': g.number_of_nodes(),
            'num_edges': g.number_of_edges(),
            'max_degree': np.max(list(dict(nx.degree(g)).values())),
            'min_cost': np.min(costs),
            'max_cost': np.max(costs),
            'min_rate': np.min(ars),
            'max_rate': np.max(ars),
            'min_demand': np.min(demand),
            'max_demand': np.max(demand),
            'min_cost_cpu': np.min(costs_cpu[costs_cpu > 0]),
            'max_cost_cpu': np.max(costs_cpu),
            'min_rate_cpu': np.min(ars_cpu[ars_cpu > 0]),
            'max_rate_cpu': np.max(ars_cpu),
            'min_demand_cpu': np.min(demand_cpu[demand_cpu > 0]),
            'max_demand_cpu': np.max(demand_cpu),
            'min_rate_tx': np.min(ars_tx[ars_tx > 0]),
            'max_rate_tx': np.max(ars_tx),
            'soft_failures': np.sum(sos),
            'hard_failures': np.sum(hos),
            'num_soft_failures': len(sos),
            'num_hard_failures': len(hos),
            'min_throughput': np.min(thp),
            'max_throughput': np.max(thp),
            'min_num_vnfs_on_cpu_core': np.min(n_vnfs_on_cpu),
            'max_num_vnfs_on_cpu_core': np.max(n_vnfs_on_cpu)
        })


    return {
        'num_nodes': np.max([r['num_nodes'] for r in stats]),
        'num_edges': np.max([r['num_edges'] for r in stats]),
        "max_degree": np.max([r['max_degree'] for r in stats]),
        "max_rate": np.max([r['max_rate'] for r in stats]),
        "min_rate": np.min([r['min_rate'] for r in stats]),
        "max_cost": np.max([r['max_cost'] for r in stats]),
        "min_cost": np.min([r['min_cost'] for r in stats]),
        "max_demand": np.max([r['max_demand'] for r in stats]),
        "min_demand": np.min([r['min_demand'] for r in stats]),
        "max_rate_cpu": np.max([r['max_rate_cpu'] for r in stats]),
        "min_rate_cpu": np.min([r['min_rate_cpu'] for r in stats]),
        "max_cost_cpu": np.max([r['max_cost_cpu'] for r in stats]),
        "min_cost_cpu": np.min([r['min_cost_cpu'] for r in stats]),
        "max_demand_cpu": np.max([r['max_demand_cpu'] for r in stats]),
        "min_demand_cpu": np.min([r['min_demand_cpu'] for r in stats]),
        "min_rate_tx": np.min([r['min_rate_tx'] for r in stats]),
        "max_rate_tx": np.max([r['max_rate_tx'] for r in stats]),
        "soft_failures": np.sum([r['soft_failures'] for r in stats]),
        "num_soft_failures": np.sum([r['num_soft_failures'] for r in stats]),
        "hard_failures": np.sum([r['hard_failures'] for r in stats]),
        "num_hard_failures": np.sum([r['num_hard_failures'] for r in stats]),
        "min_throughput": np.min([r['min_throughput'] for r in stats]),
        "max_throughput": np.max([r['max_throughput'] for r in stats]),
        "min_num_vnfs_on_cpu_core": np.min([r['min_num_vnfs_on_cpu_core'] for r in stats]),
        "max_num_vnfs_on_cpu_core": np.max([r['max_num_vnfs_on_cpu_core'] for r in stats])
    }


def analyze_graphs():
    d = '/opt/project/data/nas/graphs-ma-diederich'
    d = '/opt/project/data/graphs-golden-samples'
    graphs = []
    for f in os.listdir(d):
        graphs.append(load_graph(os.path.join(d, f)))
    return _analyze_graphs(graphs)

