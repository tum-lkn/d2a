import numpy as np
from typing import List, Dict, Tuple
from environment import utils as pgen


#import problem_generator as pgen


def make_machines(tx_cores: int, cycles_per_second: int) -> List[pgen.Machine]:
    # We need the number of cores that can be used for TX and NFs
    # Right now hardcoded (Reserved cores: 1xRX, 1xMan, 1x Divider)
    free_cores = 21
    nf_cores = free_cores - tx_cores

    if nf_cores == 0:
        machines = []
    else:
        machines = [pgen.Machine(i, cycles_per_second) for i in range(nf_cores)]
    return machines


def place_vnfs_on_cores(sfc_list: List[pgen.Sfc],
                        machines: List[pgen.Machine]) -> Dict[pgen.Job, pgen.Machine]:
    # Currently we still get the sfc_list and placement. Effectively, we only need VNFs and number of cores.
    # Extract a VNF list out of the sfclist
    job_list = []
    for sfc in sfc_list:
        for job in sfc.jobs:
            job_list.append(job)
    job_list.sort(key=lambda x: x.demand, reverse=True)

    # Place the VNFs on the nf_cores (We dont really care about the physical IDs)
    placement = {}
    machines_loads = np.zeros(len(machines))
    for job in job_list:
        # Place this job
        idx = np.argmin(machines_loads)
        machines_loads[idx] += job.demand
        placement[job] = machines[idx]
    return placement


def assign_vnfs_to_tx_threads(sfc_list: List[pgen.Sfc],
                              tx_threads: List[pgen.TxThread], inc_div) -> Dict[pgen.Job, pgen.TxThread]:
    load = np.zeros(len(tx_threads))
    jobs = []
    for sfc in sfc_list:
        jobs.extend(sfc.jobs)
        if(inc_div):
            jobs.append(pgen.Job(5e6, pgen.Vnf(40, 1)))
    jobs.sort(key=lambda x: -1. * x.rate())

    assignment = {}
    for job in jobs:
        idx = np.argmin(load)
        assignment[job] = tx_threads[idx]
        load[idx] += job.rate()
    return assignment


def map_machines_to_jobs(placement: Dict[pgen.Job, pgen.Machine]) -> Dict[pgen.Machine, List[pgen.Job]]:
    machine_to_jobs = {}
    for job, machine in placement.items():
        if machine not in machine_to_jobs:
            machine_to_jobs[machine] = []
        machine_to_jobs[machine].append(job)
    return machine_to_jobs


def calculate_machine_load(placement: Dict[pgen.Job, pgen.Machine],
                           machines: List[pgen.Machine]) -> np.array:
    machine_to_jobs = map_machines_to_jobs(placement)
    machines_loads = np.zeros(len(machines))
    for i, machine in enumerate(machines):
        for job in machine_to_jobs[machine]:
            machines_loads[i] += job.demand
    return machines_loads


def estimate_achieved_rate(placement: Dict[pgen.Job, pgen.Machine],
                           sched_lat: float) -> Dict[pgen.Job, int]:
    performance = {}
    machine_to_jobs = map_machines_to_jobs(placement)
    for machine, jobs in machine_to_jobs.items():
        time_slice = sched_lat / len(jobs)
        for job in jobs:
            arriving_packets = np.ceil(job.rate() * sched_lat)
            time_per_packet = job.vnf.compute_per_packet / machine.capacity
            time_for_packets = arriving_packets * time_per_packet
            if time_for_packets < time_slice:
                performance[job] = job.rate()
            else:
                performance[job] = int(np.min([job.rate(), time_slice / time_per_packet / sched_lat]))
    return performance


def simulate_achieved_rate(placement: Dict[pgen.Job, pgen.Machine],
                           sched_lat: float) -> Dict[pgen.Job, int]:
    """
        Estimate the performance of each VNF by simulating the behavior of the
        CFS scheduler on each core.

        Args:
            placement: Assignment of jobs to machines.
            sched_lat: Scheduling latency in seconds, duration during which
                each runnable job on the CPU core is guaranteed to get its
                turn on the CPU.

        The CFS scheduler assigns each runnable job on the core the same mount
        of time. That is, the scheduling period is divided by the number of
        runnable jobs, resulting in a specific time slice for each job.
        In our case, the VNFs, i.e., jobs, can voluntarily yield the CPU, which
        reduces the time slice. The VNFs are sorted on the CPU based on the CPU
        time that they consumed.

        The system is simulated as follows. The jobs are sorted based on their
        demand. The VNFs are sorted in increasing order based on the time they
        will take of the CPU core. The higher the index of a VNF, the higher the
        CPU time. THe first VNF needs the fewest CPU time.
        Then, the simulation checks for each VNF in turn if the VNF can manage
        its packet rate in the alotted time slice. If the VNF manages or is
        faster, the performance of the VNF is set to its rate and the time slice
        is reduced accordingly.
        If the time slice is reduced, the packets that later VNFs have to process
        is also reduced, since the period of time that they cannot process packets
        and packets stay in the buffer is reduced.
        A VNF is restricted, once it cannot handle the packets that arrive during
        the actual scheduling latency in its time slice from the original scheduling
        latency. In that case, all subsequent VNFs will also not achieve their
        rate.
        The trivial cases of the simulation are if all VNFs meet their rate, or
        no VNF meets its rate. The simulation finishes after one iteration. If
        the VNFs are mixed, i.e., some VNFs manage their rate and others do not,
        multiple iterations are needed until an equilibrium is found.
    """
    performance = {}
    machine_to_jobs = map_machines_to_jobs(placement)
    for jobs in machine_to_jobs.values():
        jobs.sort(key=lambda x: x.demand)
    for machine, jobs in machine_to_jobs.items():
        actual_lat = sched_lat
        prev_lat = sched_lat + 1
        all_failed = False
        all_succeeded = False
        iterations = 0
        while actual_lat < prev_lat and not (all_failed or all_succeeded) and iterations < 1000:
            iterations += 1
            all_failed = True
            all_succeeded = True
            prev_lat = actual_lat
            actual_lat = sched_lat
            time_slice = sched_lat / len(jobs)
            for job in jobs:
                arriving_packets = np.ceil(job.rate() * prev_lat)
                time_per_packet = job.vnf.compute_per_packet / machine.capacity
                time_for_packets = arriving_packets * time_per_packet
                if time_for_packets < time_slice:
                    all_failed = False
                    actual_lat -= time_slice - time_for_packets
                    performance[job] = job.rate()
                else:
                    all_succeeded = False
                    performance[job] = int(np.min([job.rate(), time_slice / time_per_packet / prev_lat]))
    return performance


def update_rates(performance: Dict[pgen.Job, int], sfcs: List[pgen.Sfc]):
    """
        Replace the VNFs rates with the minimum of the previous rate and its predicted
        current rate.
        Replace the rate of the SFC with the minimum rate of all VNFs.
    """
    for sfc in sfcs:
        prev_job = None
        min_rate = 1e12
        for i, job in enumerate(sfc.jobs):
            if i == 0:
                job._rate = performance[job]
            else:
                job._rate = np.min([performance[job], prev_job.rate()])
            if job._rate < min_rate:
                min_rate = job._rate
            prev_job = job
        sfc.rate = min_rate


def estimate_achieved_rate_tx_thread(assignment: Dict[pgen.Job, pgen.TxThread],
                                     threads: List[pgen.TxThread]) -> Dict[pgen.Job, int]:
    load = {thread: 0 for thread in threads}
    performance = {}
    for job, thread in assignment.items():
        # if thread > len(load) - 1:
        #     for job, thread in assignment.items():
        #         performance[job] = 0
        #     return performance
        # else:
        load[thread] += job.rate()
    
    for job, thread in assignment.items():
        new_rate = np.floor(job.rate() / load[thread] * thread.pps)
        performance[job] = np.min([new_rate, job.rate()])
    return performance


def heuristic_solution(sfc_list: List[pgen.Sfc], tx_cores: int,
                       cycles_per_second=2.2e9, divider=8.4e6,
                       sched_latency=1e-3, inc_div=False) -> Tuple[Dict[pgen.Job, pgen.Machine],
                                                    Dict[pgen.Job, pgen.TxThread],
                                                    List[pgen.Sfc]]:
    sfc_list = [pgen.Sfc.from_dict(sfc.to_dict()) for sfc in sfc_list]
    machines = make_machines(tx_cores, int(cycles_per_second))
    tx_threads = [pgen.TxThread(i, divider) for i in range(tx_cores)]
    cpu_placement = place_vnfs_on_cores(sfc_list, machines)

    cpu_performance = simulate_achieved_rate(cpu_placement, sched_latency)
    update_rates(cpu_performance, sfc_list)

    tx_assignment = assign_vnfs_to_tx_threads(sfc_list, tx_threads, inc_div)
    tx_performance = estimate_achieved_rate_tx_thread(tx_assignment, tx_threads)
    update_rates(tx_performance, sfc_list)
    return cpu_placement, tx_assignment, sfc_list


def get_tp(sfc_list: List[pgen.Sfc], tx_cores: int, print_out=False,
           cycles_per_second=2.2e9, divider=8.4e6):
    """

    Args:
         sfc_list
         tx_cores
         print_out
         cycles_per_second: Number of cycles of a CPU core.
         divider: Maximum throughput of a TX thread in pps.
    """
    if tx_cores == 0:
        return 0.0

    machines = make_machines(tx_cores, int(cycles_per_second))
    placement = place_vnfs_on_cores(sfc_list, machines)

    divider_rate = sum([chain.rate for chain in sfc_list])
    divider_nf = pgen.Vnf(40, 1)
    divider_nf.instance_id = 1
    divider_job = pgen.Job(divider_rate, divider_nf)

    tx_nf_list = []
    for i in range(tx_cores):
        tx_nf_list.append(list())

    vnfs_per_core = np.zeros(24)
    for sfc in sfc_list:
        for job in sfc.jobs:
            core = placement[job].physical_id
            vnfs_per_core[core] += 1

    sfc_tp = [[] for _ in range(len(sfc_list))]
    idx = 0
    for sfc in sfc_list:
        sfc_tp[idx].append(sfc.rate)
        for job in sfc.jobs:
            core = placement[job].physical_id
            theo_tp = (cycles_per_second / vnfs_per_core[core]) / job.vnf.compute_per_packet
            real_tp = np.minimum(theo_tp, job._rate)
            sfc_tp[idx].append(real_tp)
        idx += 1

    # Correct individual VNF throughput. Bubble the minimum to the later VNFs.
    for sfc_idx, rates in enumerate(sfc_tp):
        min_rate = 1e15
        for vnf_idx, rate in enumerate(rates):
            if rate < min_rate:
                min_rate = rate
            rates[vnf_idx] = min_rate

    # Update the rate of each job with the bottlenecked performance of each VNF.
    new_sfc_list = [pgen.Sfc.from_dict(sfc.to_dict()) for sfc in sfc_list]
    for sfc_idx, rates in enumerate(sfc_tp):
        for vnf_idx, rate in enumerate(rates):
            sfc = new_sfc_list[sfc_idx]
            job = sfc.jobs[vnf_idx]
            job._rate = rate

    vnf_list = [divider_job]
    for sfc in new_sfc_list:
        for job in sfc.jobs:
            vnf_list.append(job)
    vnf_list.sort(key=lambda x: x.rate(), reverse=True)

    loads = np.zeros(tx_cores, dtype=np.float32)
    for vnf in vnf_list:
        idx = np.argmin(loads)
        tx_nf_list[idx].append(vnf.vnf.instance_id)
        loads[idx] += vnf.rate()

    tx_tp = [[] for _ in range(len(sfc_list))]
    idx = 0
    for sfc_idx, sfc in enumerate(new_sfc_list):
        for job_idx, job in enumerate(sfc.jobs):
            current_rate = job.rate()
            total_tx_load = None
            for i, tx in enumerate(tx_nf_list):
                if job.vnf.instance_id in tx:
                    total_tx_load = loads[i]
            assert total_tx_load is not None, f"No TX Thread found for VNF with IID {job.instance_id}"

            new_rate = (current_rate / total_tx_load) * divider
            rate = np.minimum(current_rate, new_rate)
            sfc_tp[sfc_idx][job_idx] = new_rate
            tx_tp[idx].append(rate)
        idx += 1

    # Update the rate of each job with the bottlenecked performance of each VNF.
    new_sfc_list = [pgen.Sfc.from_dict(sfc.to_dict()) for sfc in sfc_list]
    for sfc_idx, rates in enumerate(sfc_tp):
        for vnf_idx, rate in enumerate(rates):
            sfc = new_sfc_list[sfc_idx]
            job = sfc.jobs[vnf_idx]
            job._rate = rate

    new_tp = [[] for _ in range(len(new_sfc_list))]
    idx = 0
    for i, sfc in enumerate(tx_tp):
        for j, vnf in enumerate(sfc):
            new_tp[idx].append(np.minimum(vnf, sfc_tp[i][j + 1]))
        idx += 1

    if print_out:
        print('VNF Table')
        for sfc in sfc_tp:
            str = ''
            for vnf in sfc:
                str += '{:6.2f} '.format(vnf)
            print(str)

        print('\nTX-Thread')
        for sfc in tx_tp:
            str = ''
            for vnf in sfc:
                str += '{:6.2f} '.format(vnf)
            print(str)

        print('\nComplete')
        for sfc in new_tp:
            str = ''
            for vnf in sfc:
                str += '{:6.2f} '.format(vnf)
            print(str)

    nf_pps = sum([min(tps) for tps in new_tp])

    sys_rate = nf_pps
    return sys_rate
