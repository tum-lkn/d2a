import numpy as np
import os
import json
from typing import List, Tuple, Dict, Union
from environment.config import get_available_cycles, GetMaxT, Run
import time
from environment import config_automator as conf


class Vnf(object):
    """
    Represents an abstract base class for concrete VNF implementations.
    """
    @classmethod
    def from_dict(cls, dict):
        return Vnf(**dict)

    def __init__(self, compute_per_packet: int):
        """
        Initializes object.

        Args:
            compute_per_packet (int): Amount of cycles per packet.
        """
        self.compute_per_packet = compute_per_packet
        self.instance_id = 0

    def to_string(self):
        return "VNF with compute {:d}".format(int(self.compute_per_packet))

    def to_dict(self):
        return {"compute_per_packet": int(self.compute_per_packet),
                "instance_id": self.instance_id}


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
    def __init__(self, rate: int, vnf: Vnf):
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


class TxThread(object):

    def __init__(self, index: int, pps: float=8.4e6):
        self.index = index
        self.pps = pps


class Sfc(object):
    """
    Represents a service function chain.
    """
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
        self._rate = rate

    @property
    def rate(self) -> int:
        return self._rate

    @rate.setter
    def rate(self, r: int) -> None:
        self.demand = r
        self._rate = r
        for job in self.jobs:
            job._rate = r

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


def export_problem(sfcs: List[Sfc], placement: Dict[Job, Machine], machines: List[Machine],
                   num_tx_threads: int) -> dict:
    """
    Create dictionary of problem instance.
    """
    sfcs_export = []
    placement_export = []
    for i, sfc in enumerate(sfcs):
        sfc_exp = []
        for j, job in enumerate(sfc.jobs):
            sfc_exp.append(job.to_dict())
            placement_export.append({
                'sfc': i,
                'job': j,
                'machine': placement[job].physical_id
            })
        sfcs_export.append(sfc_exp)
    return {
        'problem': sfcs_export,
        'placement': placement_export,
        'machines': [m.to_dict() for m in machines],
        'num_tx_threads': num_tx_threads
    }


class HeavyTailedGenerator(object):
    """
        Implements a generator in which the compute demand of each VNF is distributed
        according to heavy tailed distribution. Traffic is split across SFCs based
        on weights drawn from a uniform distribution.

        The generative process is as follows:

        1) Splitting ratios for each VNF are drawn.
        2) Arrival rate to each SFC and thus each VNF is determined.
        3) Until max number of VNFs or no machines available:
            3.1) Draw CPU cycles per packet.
            3.2) Draw SFC VNF should be assigned to.
            3.3) Search for machine that can support the demand.
            3.4) Update available capacity on that machine.
            3.5) If no such machine can be found, use the one that can satisfy
                 most of the requested demand and scale cycles of VNF accordingly.
    """
    def __init__(self, machines: List[Machine], num_sfcs: Union[int, Tuple[int, int]],
                 max_num_vnfs_per_sfc: int, seed: int, system_rate: Union[float, Tuple[float, float]],
                 load_level=0.9, max_num_vnfs_in_system=16):
        """
            Initializes object.

            Args:
                machines (list): List of Machine objects that serve as template
                    for new problem instances.
                num_sfcs: The number of SFCs that are generated.
                max_num_vnfs_per_sfc: The maximum length of SFCs.
                seed:
                system_rate: Callable that returns the arrival rate to the system
                    in packets per second.
                load_level: The load level each CPU should have. The available
                    compute resources are capped by this value.
        """
        self.machines = machines
        self.num_sfcs = num_sfcs
        self.max_num_vnfs_per_sfc = max_num_vnfs_per_sfc
        self.seed = seed
        self.load_level = load_level
        self.system_rate = system_rate
        self.max_num_vnfs_in_system = max_num_vnfs_in_system
        self.random = np.random.RandomState(seed=seed)
        t_c, c_t = get_available_cycles()
        self.t_arg_to_cycles = t_c
        self.cycles_to_t_arg = c_t
        self.cycles = list(t_c.values())
        self.cycles.sort()
        self.get_max_t = GetMaxT()

    def draw_compute(self, rate: int, capacity: int, p=None) -> int:
        max_t = np.min([
            len(self.t_arg_to_cycles) - 1,
            self.get_max_t(rate, capacity)
        ])
        if p is None:
            t = self.random.choice(a=np.arange(max_t))
        else:
            t = self.random.choice(
                a=np.arange(max_t),
                p=p
            )
        # t = self.random.choice(np.arange(max_t))
        return self.t_arg_to_cycles[t]

    def init_free_demand(self, machines: List[Machine]) -> dict:
        """
        Creates a dict mapping machines to available compute.
        Args:
            machines:

        Returns:

        """
        return {m: m.capacity * self.load_level for m in machines}

    def _draw_system_rate(self) -> float:
        if type(self.system_rate) == float:
            return self.system_rate
        elif type(self.system_rate) == tuple:
            return self.random.uniform(self.system_rate[0], self.system_rate[1])

    def draw_arrival_rates(self, num_sfcs) -> np.array:
        """
        Draw splitting ratios for SFCs.

        Returns:
            ratios: Array with the amount of traffic each SFC gets in PPS.
        """
        success = False
        ratios = None
        while not success:
            success = True
            alpha = int(self.random.randint(1, 10))
            ratios = self.random.dirichlet([alpha] * num_sfcs)
            if ratios.min() < 0.01:
                success = False
        # ratios = self.random.uniform(0.35, 0.8, size=num_sfcs)
        ratios /= np.sum(ratios)
        rates = ratios * self._draw_system_rate()
        return ratios, np.floor(rates)

    def get_machine(self, rate: int, compute: int, machine_capacity: Dict[Machine, int]
                    ) -> Tuple[Machine, int]:
        """
            Select a machine for job parameters. If no such machine is available
            choses the one that satisfies the demand as much as possible and scales
            the compute demand of the VNF to a level such that the rate can be
            satisfied with the granted compute.
            Returns None if no machine could support a minimum amount of compute.

            Args:
                rate: Arrival rate to VNF in PPS.
                compute: Computational demand in cycles per packet.
                machine_capacity: A dict mapping machines to available cycles.

            Returns:
                chosen_machine (Machine): The machine a job can be placed on or None if no
                    such machine can be found
                new_compute (float): New computational demand of VNF.
        """
        chosen_machine = None
        new_compute = compute
        demand = rate * compute
        satisfied = 0.
        for machine, av_cycles in machine_capacity.items():
            if av_cycles >= demand:
                chosen_machine = machine
                satisfied = 1
                break
            else:
                tmp = av_cycles / demand
                if tmp > satisfied:
                    satisfied = tmp
                    chosen_machine = machine
        if satisfied < 1:
            new_compute = 1e9
            t_arg = self.cycles_to_t_arg[compute] - 1
            while new_compute * rate > machine_capacity[chosen_machine] and t_arg >= 0:
                new_compute = self.t_arg_to_cycles[t_arg]
                t_arg -= 1

            if t_arg < 0:
                chosen_machine = None
        return chosen_machine, new_compute

    def _get_num_sfcs(self):
        """
        Get the number of SFCs. If self.num_sfcs is a tuple, then draw the
        number at random. If it is an int then return that int.
        """
        if type(self.num_sfcs) == tuple:
            return self.random.randint(self.num_sfcs[0], self.num_sfcs[1])
        else:
            return self.num_sfcs

    def generate_jobs(self, sfc_rates: np.array, num_sfcs) -> Tuple[List[List[Job]],
                                                          Dict[Job, Machine]]:
        """
            Generate input for new SFCs.
            First, place one VNF in each chain. Then, assign VNFs to chains randomly
            until no machine can be found that support the posed requirements.

            Args:
                sfc_rates: The rates in PPS for each SFC.

            Returns:
                jobs: List with VNFs for each SFC.
                placement: Dict mapping jobs to machines.
        """
        jobs = [[] for _ in range(num_sfcs)]
        free_cap = self.init_free_demand(self.machines)
        admissible_chains = list(range(num_sfcs))
        placement = {}
        num_jobs = 0

        it = machine = 0
        while machine is not None and len(admissible_chains) > 0 and num_jobs < self.max_num_vnfs_in_system:
            if it < num_sfcs:
                idx = it
            else:
                chain_idx = self.random.randint(0, len(admissible_chains))
                idx = admissible_chains[chain_idx]
                # If number of jobs is equal to the max number of VNFs in the
                # chain remove the index of this chain from the list. THus,
                # the chain will no longer get new members.
                if len(jobs[idx]) + 1 >= self.max_num_vnfs_per_sfc:
                    admissible_chains.pop(chain_idx)
            machine, new_compute = self.get_machine(
                rate=sfc_rates[idx],
                compute=self.draw_compute(sfc_rates[idx], self.machines[0].capacity),
                machine_capacity=free_cap
            )
            if machine is None:
                break
            else:
                job = Job(sfc_rates[idx], Vnf(new_compute))
                free_cap[machine] -= job.demand
                jobs[idx].append(job)
                placement[job] = machine
                num_jobs += 1
            it += 1
        return jobs, placement

    def __next__(self) -> Tuple[List[Sfc], Dict[Job, Machine]]:
        """
        Generate a new set of SFCs.

        Returns:
            sfcs: A list of new SFCs.
            placement: Mapping of Jobs to Machines.
        """
        successful = False
        jobs = None
        sfc_rates = None
        weights = None
        placement = None

        it = 0
        while not successful and it < 10:
            num_sfcs = self._get_num_sfcs()
            weights, sfc_rates = self.draw_arrival_rates(num_sfcs)
            it += 1
            jobs, placement = self.generate_jobs(sfc_rates, num_sfcs)
            successful = True
            for chain in jobs:
                if len(chain) == 0:
                    successful = False
                    break
                else:
                    continue
        if not successful:
            raise RuntimeError("Could not generate a minimum set of chains. Check"
                               " Your settings.")
        sfcs = [Sfc(
            jobs=j,
            rate=r,
            weight=w
        ) for j, r, w in zip(jobs, sfc_rates, weights.tolist())]
        return sfcs, placement

    def __iter__(self):
        return self


class HeavyTailedGeneratorWithTx(object):
    """
        Implements a generator in which the compute demand of each VNF is distributed
        according to heavy tailed distribution. Traffic is split across SFCs based
        on weights drawn from a uniform distribution.

        The generative process is as follows:

        1) Splitting ratios for each VNF are drawn.
        2) Arrival rate to each SFC and thus each VNF is determined.
        3) Until max number of VNFs or no machines available:
            3.1) Draw CPU cycles per packet.
            3.2) Draw SFC VNF should be assigned to.
            3.3) Search for machine that can support the demand.
            3.4) Update available capacity on that machine.
            3.5) If no such machine can be found, use the one that can satisfy
                 most of the requested demand and scale cycles of VNF accordingly.
    """

    def __init__(self, machines: List[Machine], num_sfcs: Union[int, Tuple[int, int]],
                 max_num_vnfs_per_sfc: int, seed: int, system_rate: Union[float, Tuple[float, float]],
                 load_level=0.9, max_num_vnfs_in_system=16, num_tx=4, tx_cap=8.4e6):
        """
            Initializes object.

            Args:
                machines (list): List of Machine objects that serve as template
                    for new problem instances.
                num_sfcs: The number of SFCs that are generated.
                max_num_vnfs_per_sfc: The maximum length of SFCs.
                seed:
                system_rate: Callable that returns the arrival rate to the system
                    in packets per second.
                load_level: The load level each CPU should have. The available
                    compute resources are capped by this value.
                num_tx: count of TX threads
                tx_cap: maximum achivable rate per TX
        """
        # Keep the _machines! The .machines list gets reversed later on in the
        # generate jobs method. For this we need to the original list. Since
        # the list is passed by reference, the sorting will break code at other
        # places that relies on the correct orders of machines in the array.
        self._machines = machines
        self.machines = [m for m in self._machines]
        self.num_sfcs = num_sfcs
        self.max_num_vnfs_per_sfc = max_num_vnfs_per_sfc
        self.seed = seed
        self.load_level = load_level
        self.system_rate = system_rate
        self.max_num_vnfs_in_system = max_num_vnfs_in_system
        self.num_tx = num_tx
        self.tx_cap = tx_cap
        self.tx_machines = []
        self.total_system_rate_left = self.num_tx * self.tx_cap
        self.max_num_vnfs_in_system = max_num_vnfs_in_system
        self.num_tx = num_tx
        self.current_tx_count = 0
        self.tx_machines = []
        self.random = np.random.RandomState(seed=seed)
        t_c, c_t = get_available_cycles()
        self.t_arg_to_cycles = t_c
        self.cycles_to_t_arg = c_t
        self.cycles = list(t_c.values())
        self.cycles.sort()
        self.get_max_t = GetMaxT()

    def get_uniform_over_art_to_cycles(self):
        return 0.1 # np.repeat(1., repeats=len(self.t_arg_to_cycles)) / len(self.t_arg_to_cycles)

    def draw_compute(self, rate: int, capacity: int, p=None) -> int:
        max_t = np.min([
            len(self.t_arg_to_cycles) - 1,
            self.get_max_t(rate, capacity)
        ])
        if p is None:
            t = self.random.choice(a=np.arange(max_t))
        else:
            t = self.random.choice(
                a=np.arange(max_t),
                p=np.repeat(1., repeats=max_t) / max_t
            )
        # t = self.random.choice(np.arange(max_t))
        return self.t_arg_to_cycles[t]

    def init_free_demand(self, machines: List[Machine]) -> dict:
        """
        Creates a dict mapping machines to available compute.
        Args:
            machines:

        Returns:

        """
        return {m: m.capacity * self.load_level for m in machines}

    def _draw_system_rate(self) -> float:
        if type(self.system_rate) == float:
            return self.system_rate
        elif type(self.system_rate) == tuple:
            return self.random.uniform(self.system_rate[0], self.system_rate[1])

    def draw_arrival_rates(self, num_sfcs) -> np.array:
        """
        Draw splitting ratios for SFCs.

        Returns:
            ratios: Array with the amount of traffic each SFC gets in PPS.
        """
        success = False
        ratios = None
        while not success:
            success = True
            alpha = int(self.random.randint(1, 10))
            ratios = self.random.dirichlet([alpha] * num_sfcs)
            if ratios.min() < 0.01:
                success = False
        # ratios = self.random.uniform(0.35, 0.8, size=num_sfcs)
        ratios /= np.sum(ratios)
        rates = ratios * self._draw_system_rate()
        return ratios, np.floor(rates)

    def get_machine(self, rate: int, compute: int, machine_capacity: Dict[Machine, int]
                    ) -> Tuple[Machine, int]:
        """
            Select a machine for job parameters. If no such machine is available
            choses the one that satisfies the demand as much as possible and scales
            the compute demand of the VNF to a level such that the rate can be
            satisfied with the granted compute.
            Returns None if no machine could support a minimum amount of compute.

            Args:
                rate: Arrival rate to VNF in PPS.
                compute: Computational demand in cycles per packet.
                machine_capacity: A dict mapping machines to available cycles.

            Returns:
                chosen_machine (Machine): The machine a job can be placed on or None if no
                    such machine can be found
                new_compute (float): New computational demand of VNF.
        """
        chosen_machine = None
        new_compute = compute
        demand = rate * compute
        satisfied = 0.
        for machine, av_cycles in machine_capacity.items():
            if av_cycles >= demand:
                chosen_machine = machine
                satisfied = 1
                break
            else:
                tmp = av_cycles / demand
                if tmp > satisfied:
                    satisfied = tmp
                    chosen_machine = machine
        if satisfied < 1:
            new_compute = 1e9
            t_arg = self.cycles_to_t_arg[compute] - 1
            while new_compute * rate > machine_capacity[chosen_machine] and t_arg >= 0:
                new_compute = self.t_arg_to_cycles[t_arg]
                t_arg -= 1

            if t_arg < 0:
                chosen_machine = None
        return chosen_machine, new_compute

    def _get_num_sfcs(self):
        """
        Get the number of SFCs. If self.num_sfcs is a tuple, then draw the
        number at random. If it is an int then return that int.
        """
        if type(self.num_sfcs) == tuple:
            return self.random.randint(self.num_sfcs[0], self.num_sfcs[1])
        else:
            return self.num_sfcs

    def generate_jobs(self, sfc_rates: np.array, num_sfcs) -> Tuple[List[List[Job]],
                                                                    Dict[Job, Machine]]:
        """
            Generate input for new SFCs.
            First, place one VNF in each chain. Then, assign VNFs to chains randomly
            until no machine can be found that support the posed requirements.

            Args:
                sfc_rates: The rates in PPS for each SFC.

            Returns:
                jobs: List with VNFs for each SFC.
                placement: Dict mapping jobs to machines.
        """
        jobs = [[] for _ in range(num_sfcs)]
        # Reverse the order of the machines. It is unclear why, we assume its
        # done for a reason and dont want to break the logic.
        self.machines = [m for m in self._machines]
        admissible_chains = list(range(num_sfcs))
        placement = {}
        num_jobs = 0

        # We gave all machines to ProbGen. Reserve the cores for TX-Threads
        self.tx_machines.clear()
        self.tx_machines = [self.machines[:self.num_tx]]
        self.machines = self.machines[self.num_tx:]
        free_cap = self.init_free_demand(self.machines)
        self.total_system_rate_left = self.num_tx * self.tx_cap

        current_rate = 0
        used_tx = 1
        realloc_tx = 0
        it = machine = 0
        while machine is not None and len(
                admissible_chains) > 0 and num_jobs < self.max_num_vnfs_in_system and self.total_system_rate_left > 0:
            if it < num_sfcs:
                idx = it
                # This are the first NFs of every SFCs we need the rate to count twice (Divider)
                self.total_system_rate_left -= sfc_rates[idx]
                current_rate += sfc_rates[idx]
            else:
                chain_idx = self.random.randint(0, len(admissible_chains))
                idx = admissible_chains[chain_idx]
                # If number of jobs is equal to the max number of VNFs in the
                # chain remove the index of this chain from the list. THus,
                # the chain will no longer get new members.
                if len(jobs[idx]) + 1 >= self.max_num_vnfs_per_sfc:
                    admissible_chains.pop(chain_idx)

            machine, new_compute = self.get_machine(
                rate=sfc_rates[idx],
                compute=self.draw_compute(sfc_rates[idx], self.machines[0].capacity, self.get_uniform_over_art_to_cycles()),
                machine_capacity=free_cap
            )

            if machine is None or self.total_system_rate_left <= 0:
                # We currently don't have anymore empty machines
                # Check if we can reallocate some TX-Threads
                # if machine is None and (used_tx + realloc_tx) < self.num_tx:
                #     # at least one TX core can be reallocated
                #     self.total_system_rate_left -= self.tx_cap
                #     realloc_tx += 1

                #     # Return a machine to free_cap
                #     re_machine = self.tx_machines[-1]
                #     self.tx_machines.remove(self.tx_machines[-1])
                #     free_cap[re_machine[0]] = re_machine[0].capacity * self.load_level
                #     machine = 0

                # else:
                #     break
                pass
            else:
                job = Job(sfc_rates[idx], Vnf(new_compute))
                free_cap[machine] -= job.demand
                self.total_system_rate_left -= sfc_rates[idx]
                current_rate += sfc_rates[idx]

                # For internal bookkeeping calculate the current number of tx
                if current_rate - (used_tx * self.tx_cap) >= 0:
                    if (used_tx + realloc_tx) < self.num_tx:
                        used_tx += 1

                jobs[idx].append(job)
                placement[job] = machine
                num_jobs += 1

            it += 1
        return jobs, placement

    def __next__(self) -> Tuple[List[Sfc], Dict[Job, Machine], List]:
        """
        Generate a new set of SFCs.

        Returns:
            sfcs: A list of new SFCs.
            placement: Mapping of Jobs to Machines.
        """
        successful = False
        jobs = None
        sfc_rates = None
        weights = None
        placement = None

        it = 0
        while not successful and it < 10:
            num_sfcs = self._get_num_sfcs()
            weights, sfc_rates = self.draw_arrival_rates(num_sfcs)
            it += 1
            jobs, placement = self.generate_jobs(sfc_rates, num_sfcs)
            successful = True
            for chain in jobs:
                if len(chain) == 0:
                    successful = False
                    break
                else:
                    continue
        if not successful:
            raise RuntimeError("Could not generate a minimum set of chains. Check"
                               " Your settings.")
        sfcs = [Sfc(
            jobs=j,
            rate=r,
            weight=w
        ) for j, r, w in zip(jobs, sfc_rates, weights.tolist())]

        vnf_id = 2
        for sfc in sfcs:
            for job in sfc.jobs:
                job.vnf.instance_id = vnf_id
                vnf_id += 1

        return sfcs, placement, [m[0] for m in self.tx_machines]

    def __iter__(self):
        return self


class ForwarderTestGenerator(object):
    def __init__(self, machines: List[Machine]) -> None:
        super().__init__()
        self.machines = machines

    def generate_problem(self):
        first_core = self.machines[0].physical_id
        last_core = self.machines[-1].physical_id

        # Draw core uniformly
        core = np.random.randint(first_core, last_core)

        # Draw iteration mean
        mean = np.random.randint(0, 1000)

        # Draw iteration variance
        var = np.random.randint(0, 50)

        # Draw array size
        arr_size = np.random.randint(1e6, 1e9)

        forwarder = conf.ForwarderNFConfig(1, 1, 2, core=core, iteration_mean=mean, iteration_var=float(var), array_size=arr_size, use_nic=1)

        return forwarder

    def __next__(self):
        conf = self.generate_problem()
        return conf

    def __iter__(self):
        return self


class CrissCrossGenerator(object):
    def __init__(self):
        super().__init__()
        self._state = 0

    def criss_cross(self):
        worker_list = []
        worker_list.append(conf.ForwarderNFConfig(1, 1, 2, core=8, iteration_mean=0, iteration_var=float(0), array_size=int(1e6)))
        worker_list.append(conf.ForwarderNFConfig(2, 2, 3, core=12, iteration_mean=0, iteration_var=float(0), array_size=int(1e6)))
        worker_list.append(conf.ForwarderNFConfig(3, 3, 4, core=9, iteration_mean=0, iteration_var=float(0), array_size=int(1e6)))
        worker_list.append(conf.ForwarderNFConfig(4, 4, 5, core=13, iteration_mean=0, iteration_var=float(0), array_size=int(1e6), use_nic=1))

        return worker_list

    def same_socket_1(self):
        worker_list = []
        worker_list.append(conf.ForwarderNFConfig(1, 1, 2, core=8, iteration_mean=0, iteration_var=float(0), array_size=int(1e6)))
        worker_list.append(conf.ForwarderNFConfig(2, 2, 3, core=9, iteration_mean=0, iteration_var=float(0), array_size=int(1e6)))
        worker_list.append(conf.ForwarderNFConfig(3, 3, 4, core=10, iteration_mean=0, iteration_var=float(0), array_size=int(1e6)))
        worker_list.append(conf.ForwarderNFConfig(4, 4, 5, core=11, iteration_mean=0, iteration_var=float(0), array_size=int(1e6), use_nic=1))

        return worker_list

    def same_socket_2(self):
        worker_list = []
        worker_list.append(conf.ForwarderNFConfig(1, 1, 2, core=12, iteration_mean=0, iteration_var=float(0), array_size=int(1e6)))
        worker_list.append(conf.ForwarderNFConfig(2, 2, 3, core=13, iteration_mean=0, iteration_var=float(0), array_size=int(1e6)))
        worker_list.append(conf.ForwarderNFConfig(3, 3, 4, core=14, iteration_mean=0, iteration_var=float(0), array_size=int(1e6)))
        worker_list.append(conf.ForwarderNFConfig(4, 4, 5, core=15, iteration_mean=0, iteration_var=float(0), array_size=int(1e6), use_nic=1))

        return worker_list

    def half(self):
        worker_list = []
        worker_list.append(conf.ForwarderNFConfig(1, 1, 2, core=8, iteration_mean=0, iteration_var=float(0), array_size=int(1e6)))
        worker_list.append(conf.ForwarderNFConfig(2, 2, 3, core=9, iteration_mean=0, iteration_var=float(0), array_size=int(1e6)))
        worker_list.append(conf.ForwarderNFConfig(3, 3, 4, core=12, iteration_mean=0, iteration_var=float(0), array_size=int(1e6)))
        worker_list.append(conf.ForwarderNFConfig(4, 4, 5, core=13, iteration_mean=0, iteration_var=float(0), array_size=int(1e6), use_nic=1))

        return worker_list

    def next(self):
        if self._state == 0:
            self._state = 1
            return self.same_socket_1()
        elif self._state == 1:
            self._state = 2
            return self.criss_cross()
        elif self._state == 2:
            self._state = 3
            return self.same_socket_2()
        elif self._state == 3:
            self._state = 0
            return self.half()


class GoldenSampleGenerator(object):
    def __init__(self, path_to_samples: str, seed=None):
        self.random = np.random.RandomState(seed=seed)
        with open(path_to_samples, 'r') as fh:
            self.problems_dicts = json.load(fh)

    def __next__(self) -> Tuple[List[Sfc], None, List[TxThread]]:
        txs = [TxThread(i) for i in range(5)]
        problem_dict = self.problems_dicts[self.random.randint(0, len(self.problems_dicts))]
        sfcs = sfcs_from_problem(problem_dict)
        for sfc in sfcs:
            sfc.rate = int(np.round(self.random.uniform(0.25, 1.), 2) * sfc.rate)
        return sfcs, None, txs

    def __iter__(self):
        return self


def sfcs_from_run_config_and_problem(run_config: Run, problem: Dict) -> List[Sfc]:
    """
    Crate the SFCs used in a particular experiment. The run_config provides rate
    and weights, the problem the cycle counts of VNFs
    """
    bit_s = run_config.bit_s
    pps = int(run_config.bit_s / ((run_config.packet_size + 4) * 8))
    pps = 2e6

    sfcs = []
    for i, vnf_list in enumerate(problem['problem']):
        sfc_rate = int(run_config.weights[i] * pps)
        vnfs = [Job(sfc_rate, Vnf(d['vnf']['compute_per_packet'])) for d in vnf_list]
        sfcs.append(Sfc(vnfs, sfc_rate, run_config.weights[i]))
    return sfcs


def get_bits_from_problem(problem: Dict) -> int:
    bit_s = 0

    for sfc in problem['problem']:
        bit_s += sfc[0]['rate']

    return int(bit_s)


def get_weights_from_problem(problem: Dict) -> List[float]:
    weights = []
    
    total_bits = get_bits_from_problem(problem)

    for sfc in problem['problem']:
        weights.append(sfc[0]['rate'] / total_bits)

    return weights


def sfcs_from_problem(problem: Dict, packet_size: int = 64) -> List[Sfc]:
    """
    Crate the SFCs used in a particular experiment.
    """
    bit_s = get_bits_from_problem(problem)
    pps = int(bit_s / ((packet_size + 4) * 8))
    weights = get_weights_from_problem(problem)

    sfcs = []
    for i, vnf_list in enumerate(problem['problem']):
        sfc_rate = int(weights[i] * bit_s)
        vnfs = [Job(sfc_rate, Vnf(d['vnf']['compute_per_packet'])) for d in vnf_list]
        sfcs.append(Sfc(vnfs, sfc_rate, weights[i]))
    return sfcs


def machines_from_problem(problem: Dict) -> List[Machine]:
    """
    Create machines from the saved problem instance.
    """
    if 'machines' in problem:
        return [Machine(**d) for d in problem['machines']]
    else:
        return [Machine(i, int(2.2e9)) for i in range(8, 12)]


def placement_from_problem(machines: List[Machine], sfcs: List[Sfc],
                           problem: dict) -> Dict[Vnf, Machine]:
    """
    Reconstruct the placement that was used in the given experimentm
    """
    placement = {}
    for pl in problem['placement']:
        placement[sfcs[pl['sfc']].jobs[pl['job']]] = machines[pl['machine'] - 8]
    return placement


def problem_from_experiment(exp_folder: str) -> Tuple[Run, List[Machine], List[Sfc], Dict[Job, Machine]]:
    """
    Create the complete problem instance from an experiment folder. That is,
    the machines, SFCs and the placement as well as run_config.
    """
    with open(os.path.join(exp_folder, 'problem.json'), 'r') as fh:
        problem = json.load(fh)
    with open(os.path.join(exp_folder, 'run-config.json'), 'r') as fh:
        run_config = Run(**json.load(fh))
    machines = machines_from_problem(problem)
    sfcs = sfcs_from_run_config_and_problem(run_config, problem)
    placement = placement_from_problem(machines, sfcs, problem)
    return run_config, machines, sfcs, placement


def get_num_runs(exp_folder: str):
    """
    Get the number of runs that were performed for a experiment.
    """
    num_runs = 0
    while os.path.exists(os.path.join(exp_folder, "{:d}".format(num_runs))):
        num_runs += 1
    return num_runs - 1


def get_offset(exp_folder: str, part: str) -> int:
    """
    For restarts, get the offset from which running experiments should be
    continued.
    """
    offset = 0
    for f in os.listdir(exp_folder):
        if f.find('-{:s}-'.format(part)) > 0:
            tmp = int(f.split('-')[2])
            if tmp > offset:
                offset = tmp
    return offset


def existing_exp_generator(folder: str, cfs: bool, backpressure: bool):
    assert (not cfs) or (not backpressure), "Only CFS or backpressure can be set"
    assert cfs or backpressure, "CFS or backpressure must be set"
    if cfs:
        part = 'cfs'
        rc = False
    if backpressure:
        part = 'bp'
        rc = True

    max_num_exp = get_offset(folder, 'rc')
    offset = get_offset(folder, part)
    edirs = [f for f in os.listdir(folder) if f.find('-rc-') > 0]
    edirs = [f for f in edirs if int(f.split('-')[2]) > offset]
    edirs.sort(key=lambda x: int(x.split("_")[-1]))

    for f in edirs:
        parts = f.split("_")[0].split("-")
        nruns = get_num_runs(os.path.join(folder, f))
        if nruns == 0:
            nruns = 1
        run_config, machines, sfcs, placement = problem_from_experiment(
            os.path.join(folder, f)
        )

        run_config.with_backpressure = backpressure
        run_config.with_rate_cost = rc
        run_config.experiment_name = 'infinity-{:s}-{:s}-{:s}'.format(part, parts[2], parts[3])
        yield run_config, machines, sfcs, placement, nruns
        offset += 1


def pending_exp_generator(folder, recurse=True):
    def load_dict(file_path):
        with open(file_path, "r") as fh:
            d = json.load(fh)
        return d
    num_templates_found = 0
    files = os.listdir(folder)
    files.sort(key=lambda x: 0 if x.split('-')[2] == '0000' else int(x.split('-')[2].lstrip('0')))
    for exp_dir in files:
        print(exp_dir)
        full_exp_path = os.path.join(folder, exp_dir)
        status_txt = os.path.join(full_exp_path, "status.json")
        if not os.path.exists(status_txt):
            continue
        elif load_dict(status_txt)['status'] == 'PENDING':
            run_config, machines, sfcs, placement = problem_from_experiment(full_exp_path)
            with open(status_txt, "w") as fh:
                json.dump({"status": "RUNNING"}, fh)
            num_templates_found += 1
            yield run_config, machines, sfcs, placement, 1, status_txt
        else:
            continue

    files = os.listdir('/home/nfv/FAILED_RUNS')
    files.sort(key=lambda x: 0 if x.split('-')[2] == '0000' else int(x.split('-')[2].lstrip('0')))
    for exp_dir in files:
        print(exp_dir)
        full_exp_path = os.path.join(folder, exp_dir)
        status_txt = os.path.join(full_exp_path, "status.json")
        if not os.path.exists(status_txt):
            continue
        elif load_dict(status_txt)['status'] == 'PENDING':
            run_config, machines, sfcs, placement = problem_from_experiment(full_exp_path)
            with open(status_txt, "w") as fh:
                json.dump({"status": "RUNNING"}, fh)
            num_templates_found += 1
            yield run_config, machines, sfcs, placement, 1, status_txt
        else:
            continue
    while True:
        print("---------------------- Noe pending runs")
        time.sleep(600)
    # if num_templates_found == 0:
    #     try:
    #         while True:
    #             print("----------------------------No pending runs.")
    #             # time.sleep(60)
    #             files = os.listdir('/home/nfv/NAS/infinity5')
    #             files.sort(key=lambda x: 0 if x.split('-')[2] == '0000' else int(x.split('-')[2].lstrip('0')))
    #             for exp_dir in files:
    #                 tmp = exp_dir.split('_')[0]
    #                 prefix, mode, num, algo = tmp.split('-')
    #                 if int(num.lstrip('0')) > 400:
    #                     continue
    #                 elif not os.path.exists(status_txt):
    #                     continue
    #                 elif algo in ['RoundRobin', 'BestFirstFit', 'LeastLoadedFirst', 'EpsilonGreedyPerf04', 'EpsilonGreedyIdeal01']:
    #                     full_exp_path = os.path.join(folder, exp_dir)
    #                     status_txt = os.path.join(full_exp_path, "status.json")
    #                     run_config, machines, sfcs, placement = problem_from_experiment(full_exp_path)
    #                     yield run_config, machines, sfcs, placement, 1, status_txt
    #     except Exception as e:
    #         while True:
    #             print(e)
    #             time.sleep(60)


if __name__ == '__main__':
    from environment.config import to_moongen_cmd
    count = 0
    for run_config, machines, sfcs, placement, nruns in existing_exp_generator(
            folder='../../nokia-scheduling/data/nas/placement-evaluations/',
            cfs=True,
            backpressure=False):
        print(run_config.experiment_name)
        print(run_config.run_manager_cmd)
        print(to_moongen_cmd(run_config))
        print('num sfcs {}, sfc lengths {}'.format(
            len(sfcs), ";".join([str(len(sfc.jobs)) for sfc in sfcs])))
        print(json.dumps(export_problem(sfcs, placement, machines), indent=1))
        print('-----------------------------------------------------------')
        count += 1
        if count > 5:
            break
