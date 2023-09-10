import os
import json
from typing import Tuple, List, Dict, Any


class Vnf(object):
    """
    Represents an abstract base class for concrete VNF implementations.
    """
    @classmethod
    def from_dict(cls, dict):
        return Vnf(**dict)

    def __init__(self, compute_per_packet: int, instance_id: int):
        """
        Initializes object.

        Args:
            compute_per_packet (int): Amount of cycles per packet.
        """
        self.compute_per_packet = compute_per_packet
        self.instance_id = instance_id

    def to_string(self):
        return "VNF with compute {:d}".format(int(self.compute_per_packet))

    def to_dict(self):
        return {
            "compute_per_packet": int(self.compute_per_packet),
            "instance_id": self.instance_id
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

    def __init__(self, index: int, pps: float=8.4e6):
        self.index = index
        self.pps = pps


def sfcs_from_run_config_and_problem(general_info: GeneralInfo, problem: Dict) -> List[Sfc]:
    """
    Crate the SFCs used in a particular experiment. The run_config provides rate
    and weights, the problem the cycle counts of VNFs
    """
    pps = general_info.rate

    sfcs = []
    for i, vnf_list in enumerate(problem['problem']):
        sfc_rate = int(general_info.weights[i] * pps)
        assert sfc_rate == vnf_list[0]['rate'], 'Calculated and stored rate do not match.' + \
            f'calculated is {sfc_rate}, stored is {vnf_list[0]["rate"]}.'
        vnfs = [Job(
            rate=sfc_rate,
            vnf=Vnf(d['vnf']['compute_per_packet'], general_info.first_ids[i] + j)
        ) for j, d in enumerate(vnf_list)]
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


def placement_from_problem(machines: List[Machine], sfcs: List[Sfc],
                           problem: dict) -> Dict[Job, Machine]:
    """
    Reconstruct the placement that was used in the given experimentm
    """
    offset = 24 - len(machines)
    placement = {}
    for pl in problem['placement']:
        placement[sfcs[pl['sfc']].jobs[pl['job']]] = machines[pl['machine'] - offset]
    return placement


def make_tx_assignment(general_info: GeneralInfo, sfcs: List[Sfc], threads: List[TxThread],
                       tx_assignment: Dict[str, List[int]]) -> Dict[Job, TxThread]:
    # Initialize with the divider NF which is needed in this case here.
    jobs = [Job(general_info.rate, Vnf(40, 1))]
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
    machines = machines_from_general_info(general_info)
    threads = [TxThread(i) for i in range(general_info.num_tx)]
    sfcs = sfcs_from_run_config_and_problem(general_info, problem)
    placement = placement_from_problem(machines, sfcs, problem)
    tx_assignment_ = make_tx_assignment(general_info, sfcs, threads, tx_assignment)
    return general_info, machines, sfcs, placement, tx_assignment_, threads


if __name__ == '__main__':
    ret = problem_from_experiment('/opt/projects/vnf-perf-prediction/data/nas/BLTest_1/8SFCs_bl_test-p3469-i00/')
    # tp = get_tp(
    #     sfc_list=ret[2],
    #     tx_cores=ret[0].num_tx,
    #     print_out=True
    # )
    # print(f"System Rate is {tp}")
    print("Done")
