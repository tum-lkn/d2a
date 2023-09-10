#from gym.spaces import spaces
from __future__ import annotations

import time

import ray
import gym
import ray.rllib.agents.sac as sac
import ray.rllib.agents.dqn.dqn as dqn
import sklearn.base
from ray.rllib.models.catalog import ModelCatalog
import numpy as np
import uuid
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.schedulers.pbt import PopulationBasedTraining
from ray.tune.schedulers.async_hyperband import ASHAScheduler
import os
from ray.tune import grid_search
from joblib import load
from cpu_assignment.cpu_model import CPUPlayerModel, CPUPlayerSACModel,\
    CPUPlayerPolicyModel, CPUSensor, CpuAndVnfSensor, CPUPlayerWithVnfObjectsModel,\
    CPUPlayerDQNModel

from environment import problem_generator_2 as pgen
from ray.tune.registry import register_env
from typing import Any, Dict, Iterable, List, Tuple
from scripts.eval_perfmodel import load_model as load_perf_model
import environment.tp_sim as tp
import torch
from ray.rllib.utils.typing import ModelConfigDict, ModelInputDict, \
    TensorStructType


from evaluation.validation_data import ProblemData


NAMESPACE = 'rl-bin-pack'


class SklearnTemplate(object):
    def predict_proba(self, x) -> np.array:
        raise NotImplemented

    def score(self, x) -> np.array:
        raise NotImplemented

    def predict_log_proba(self, **kwargs) -> np.array:
        raise NotImplemented


@ray.remote
class RandomForestActor(object):
    def __init__(self, model_path: str):
        self.model = load(model_path)

    def predict_proba(self, total_rate: int, total_compute: int):
        """
        Predict probability that the cores is overloaded.
        """
        p = self.model.predict_proba(
            np.array([[total_rate, total_compute]], dtype=np.float32)
        )[0, 1]
        return float(p)


@ray.remote
class PerfModelActor(object):
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model = load_perf_model(checkpoint_path)

    def predict_proba(self, node_features: np.array, adj: np.array, mask: np.array,
                      query_idx: int):
        with torch.no_grad():
            p = self.model(
                node_features=torch.tensor(node_features, dtype=torch.float32, requires_grad=False),
                query_idxs=torch.tensor([query_idx], dtype=torch.int64, requires_grad=False),
                masks=torch.tensor(mask, dtype=torch.float32, requires_grad=False),
                adj=torch.tensor(adj, dtype=torch.float32, requires_grad=False)
            ).detach().numpy()[0, 1]
        return float(p)


class LogRegActor(object):
    def __init__(self, **kwargs):
        """
                       num_vnfs          rate          cost     demand  min_ratio  max_ratio
                coef  11.391285  7.528015e+00      2.835782  33.426307  -6.468456   8.326788
                min    1.000000  3.554800e+04    175.000000   0.005495   0.004017   0.110489
                max   15.000000  7.830358e+06  26305.000000   2.497223   1.000000   1.000000
                Intercept [-14.15275639]
        """
        self.intercept = -14.15275639
        self.coef = np.array([[11.39128455,  7.5280154 ,  2.83578214, 33.42630657,
                              -6.46845557, 8.32678817]])
        self.min_vals = np.array([1.00000000e+00, 3.55480000e+04, 1.75000000e+02,
                                  5.49501818e-03, 4.01684110e-03, 1.10489350e-01])
        self.max_vals = np.array([1.50000000e+01, 7.83035800e+06, 2.63050000e+04,
                                  2.49722312e+00, 1.00000000e+00, 1.00000000e+00])

    def predict_proba(self, num_vnfs: float, min_ratio: float, max_ratio: float,
                      ratio_ratio: float, rate: float, cost: float, demand: float) -> float:
        fts = np.array([num_vnfs, rate, cost, demand, min_ratio, max_ratio])
        fts = (fts - self.min_vals) / (self.max_vals - self.min_vals)
        p = 1. / (1 + np.exp(-1. * (np.sum(np.multiply(fts, self.coef)) + self.intercept)))
        return p


class CPUHelper(object):
    def __init__(self, id: int, machine: pgen.Machine, load_level, overload_estimation,
                 model: SklearnTemplate = None, graph_helper=None) -> None:
        super().__init__()
        assert 0 < load_level <= 1.001, f"Load level out of bounds: 0 < {load_level} <= 1."
        # Identification
        self.id = id
        self.machine = machine

        # Usage stats
        self.job_assingment: List[pgen.Job] = []
        self.nf_jobs: List[pgen.Job] = []
        self.went_over_limit = False
        self.total_demand = 0.
        self.total_rate = 0.
        self.total_compute = 0.
        self.log_reg_coef = np.array([5.74945834, 11.93203677,  3.82811917])
        self.log_reg_bias = -8.45516459
        self.threshold = 0.5
        self.model: [RandomForestActor | PerfModelActor | LogRegActor] = model
        self.load_level = load_level
        self.overload_estimation = overload_estimation
        self.graph_helper = graph_helper

    @property
    def num_jobs(self) -> int:
        return len(self.nf_jobs)

    def get_job_assignment(self):
        self.job_assingment.clear()
        for nf_job in self.nf_jobs:
            self.job_assingment.append(nf_job)
    
    def export_assignment(self, a_dict):
        self.get_job_assignment()
        for job in self.job_assingment:
            a_dict[job] = self.machine
        return a_dict

    def get_total_demand(self) -> float:
        # demand = 0.0
        # for nf_job in self.nf_jobs:
        #     demand += nf_job.demand
        # return demand
        return self.total_demand

    def query_model(self, job: pgen.Job) -> bool:
        if type(self.model) == RandomForestActor:
            p = ray.get(self.model.predict_proba.remote(
                self.total_rate + job.rate(),
                self.total_compute + job.vnf.compute_per_packet
            ))
        elif type(self.model) == LogRegActor:
            demands = [j.demand for j in self.nf_jobs]
            demands.append(job.demand)
            demands = np.array(demands) / (self.total_demand + job.demand)
            min_ratio = np.min(demands)
            max_ratio = np.max(demands)
            p = self.model.predict_proba(
                num_vnfs=float(self.num_jobs) + 1.,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
                ratio_ratio=np.abs(1. - min_ratio / max_ratio),
                rate=self.total_rate + job.rate(),
                cost=self.total_compute + job.vnf.compute_per_packet,
                demand=(self.total_demand + job.demand) / 2.2e9
            )
        else:
            data = self.graph_helper.get_features(self.machine.physical_id)
            p = ray.get(self.model.predict_proba.remote(
                node_features=data['node_features'],
                adj=data['adj'],
                mask=data['mask'],
                query_idx=data['cpu_idx']
            ))
        model_overload = p >= 0.5
        del p
        # If either of the two criteria is true return true.
        return model_overload

    def _would_go_over_limit_rc(self, job: pgen.Job) -> bool:
        is_overload = self.total_demand + job.demand > (self.machine.capacity * self.load_level)
        return is_overload

    def _would_go_over_limit_cfs(self, job: pgen.Job) -> bool:
        # Calculate the number of available cycles as a fraction of the VNFs on the core.
        available = self.machine.capacity * self.load_level / (len(self.nf_jobs) + 1.)
        is_overload = job.demand >= available
        if not is_overload:
            # In case the VNF itself is not overloaded check if the assignment
            # overloads aony of the other VNFs. If so set to true.
            for nf in self.nf_jobs:
                if nf.demand > available:
                    is_overload = True
                    break
                else:
                    pass
        return is_overload

    def would_go_over_limit(self, job: pgen.Job) -> bool:
        # x = np.array([
        #     (self.total_compute + job.vnf.compute_per_packet) / (),
        #     (self.total_rate + job.rate()) / (),
        #     (self.total_demand + job.demand) / ()
        # ])
        # p_over_limit = 1. / (1. + np.exp(-1. * np.sum(x * self.log_reg_coef) + self.log_reg_bias))
        # return p_over_limit > self.threshold
        is_overload = {
            'rc': self._would_go_over_limit_rc,
            'cfs': self._would_go_over_limit_cfs
        }[self.overload_estimation](job)
        if self.model is None:
            pass
        elif not is_overload:
            # predict the probability that the job will overload the CPU core.
            # If the probability is larger than a threshold, then the helper
            # assumes that the job will overload the core
            # If either of the two criteria is true return true.
            is_overload = self.query_model(job)
        return is_overload and self.num_jobs > 0

    def add_job(self, job: pgen.Job):
        self.nf_jobs.append(job)
        nf_rate = job.rate()
        nf_compute = job.vnf.compute_per_packet
        self.total_rate += nf_rate
        self.total_compute += nf_compute
        self.total_demand += job.demand

        if self.total_demand > self.machine.capacity:
            self.went_over_limit = True


class CurrentVnf(object):

    def __init__(self, sfcs: List[pgen.Sfc], sort: bool) -> 'CurrentVnf':
        super().__init__()

        self.sfcs: List[pgen.Sfc] = sfcs
        self.nf_list: List[pgen.Job] = []
        self.first_ids: List[int] = []

        idx = 0
        for sfc in self.sfcs:
            self.first_ids.append(idx)            
            for job in sfc.jobs:
                self.nf_list.append(job)
                idx += 1

        if sort:
            self.nf_list.sort(key=lambda x: x.demand, reverse=True)

        self.id = 0
        self.job = self.nf_list[0]
        self.sfc_id = self.get_sfc_id(self.id)

    def next(self) -> bool:
        """
        This function steps to the next VNF in the list
        
        return: True if we reached the end
        """
        if self.id < len(self.nf_list) - 1:
            self.id += 1
            self.job = self.nf_list[self.id]
        else:
            return True

    def get_sfc_id(self, vnf_id):
        sfc_id = 0
        for i, id in enumerate(self.first_ids):
            if vnf_id >= id:
                sfc_id = i
        
        return sfc_id

    def get_num_nfs(self) -> int:
        return len(self.nf_list)


class ONVMMultiAgentCPU(MultiAgentEnv):
    def __init__(self, env_config) -> None:
        super().__init__()

        # ONVM specific config items
        self.machine_count = env_config["machine_count"]
        self.cycles_per_second = env_config["cycles_per_second"]
        self.num_sfcs = env_config["num_of_sfcs"]
        self.max_vnfs_per_sfc = env_config["max_vnfs_per_sfc"]
        self.max_vnfs = env_config["max_vnfs"]
        self.min_vnfs = env_config["min_vnfs"]
        self.load_level = env_config["load_level"]
        self.rate = env_config["rate"]
        self.reward_factor = env_config["reward_factor"]
        self.use_overload_prediction = env_config.get('overload_prediction', False)
        num_overload_actors = env_config.get('num_overload_actors', 48)
        self.overload_model = None
        if self.use_overload_prediction:
            # ctx = ray.init(address='auto', namespace=NAMESPACE)
            # print(ray.get_runtime_context().namespace)
            if env_config['overload_model_class'] in ['RandomForestActor', 'PerfModelActor']:
                self.overload_model = ray.get_actor(
                    env_config["overload_model_path"] + f'-{int(np.random.randint(0, num_overload_actors))}'
                )
            elif env_config['overload_model_class'] == 'LogRegActor':
                self.overload_model = LogRegActor()
            else:
                raise KeyError(f"Unknown overload model class {env_config['overload_model_class']}")
        self.my_env_step = 0
        self.num_tx = 5
        if 'evaluation_seed' in env_config:
            self.random = np.random.RandomState(seed=env_config['evaluation_seed'])
        else:
            self.random = np.random.RandomState()  # seed=env_config['seed'])

        self.all_machines = [pgen.Machine(i, self.cycles_per_second) for i in range(3, self.machine_count + 3)]
        num_sfcs = self.random.randint(1, self.num_sfcs + 1)
        if env_config.get("problem_generator", "HeavyTailedWithTx") == 'HeavyTailedWithTx':
            self.problem_generator = pgen.HeavyTailedGeneratorWithTx(
                machines=self.all_machines,
                num_sfcs=num_sfcs,
                max_num_vnfs_per_sfc=self.max_vnfs_per_sfc,
                load_level=self.load_level,
                max_num_vnfs_in_system=self.max_vnfs + 1,
                system_rate=self.rate,
                seed=self.random.randint(1, 1000000),
                num_tx=self.num_tx
            )
        elif env_config.get("problem_generator", "HeavyTailedWithTx") == 'GoldenSampleGenerator':
            self.problem_generator = pgen.GoldenSampleGenerator(
                '/opt/project/cpu_assignment/golden-samples-2-dot-5-mpps.json',
                seed=env_config['evaluation_seed'] if 'evaluation_seed' in env_config else None
            )
        else:
            raise KeyError(f'Unknown generator {env_config.get("problem_generator", "HeavyTailedWithTx")}')

        # Problem variables
        self.sfcs: List[pgen.Sfc] = []
        self.vnf2sfc_num: Dict[pgen.Job, int] = {}
        self.pl = None

        # Helper for TP Sim functions
        self.current_problem = None
        self.vnf_machines: List[pgen.Machine] = []
        self.tx_machines: List[pgen.Machine] = []
        self.tx_count = 0
        self.cpu_count = 0
        self.sort_vnfs = True

        # VNF Assignment stats
        self.all_vnf_done = False
        self.current_vnf: CurrentVnf = None
        self.num_vnfs = 0
        self.cpu_helper: List[CPUHelper] = []
        self.vnf_to_player_id: Dict[pgen.Job, int] = {}
        self.vnf_that_went_over_limit = {}
        self.opened_new_cpu = False
        self.over_utilized_core = False
        self.evaluation = env_config.get("evaluation", False)

        self.sensor: CpuAndVnfSensor = CpuAndVnfSensor()
        # self.reward_fct = BinPackingReward()
        # self.reward_fct = SequenceBasedReward()
        self.reward_fct = VnfPlayerBinPackingReward()
        # List that stores problems in which an overload occurred. Environment
        # samples from this pool to train on hard ones.
        self.hard_problems: List[ProblemData] = []

        # Gym Action & Observation space
        self.action_space = gym.spaces.Discrete(16)
        self.observation_space = self.sensor.observation_space
        self.reset()

    def get_divider_sfc(self):
        div_vnf = pgen.Vnf(40)
        div_vnf.instance_id = 1
        div_job = pgen.Job(self.rate, div_vnf)
        sfc = pgen.Sfc([div_job], self.rate, 1.0)
        return sfc

    def get_new_problem(self, problem: ProblemData=None):
        # Generate a new problem
        if problem is None:
            self.problem_generator._machines = self.all_machines[:self.num_tx + self.random.randint(4, 17)]
            self.sfcs, pl_gen, tx = self.problem_generator.__next__()
            self.tx_count = len(tx)
            self.current_problem = ProblemData(self.sfcs, self.tx_count)
        else:
            self.current_problem = problem
            self.sfcs = problem.sfcs
            self.tx_count = problem.tx_count

        self.vnf2sfc_num = {}
        self.vnf_idx2sfc_num = {}
        for i, sfc in enumerate(self.sfcs):
            for job in sfc.jobs:
                self.vnf2sfc_num[job] = i
        self.vnf_machines = self.all_machines[self.tx_count:]
        self.tx_machines = self.all_machines[:self.tx_count]
        self.cpu_count = len(self.vnf_machines)

        self.current_vnf = CurrentVnf(self.sfcs, self.sort_vnfs)
        self.num_vnfs = self.current_vnf.get_num_nfs()
        self.vnf_to_player_id = {vnf: i for i, vnf in enumerate(self.current_vnf.nf_list)}
        self.cpu_helper = [CPUHelper(self.vnf_machines[i].physical_id,
                                     self.vnf_machines[i],
                                     self.load_level,
                                     {
                                         VnfPlayerBinPackingReward: 'rc',
                                         LoadBalancingReward: 'rc',
                                         VnfPlayerBinPackingCfsReward: 'cfs',
                                         LoadBalancingCfsReward: 'cfs'
                                     }[type(self.reward_fct)],
                                     self.overload_model)
                           for i in range(self.cpu_count)]

    def _reset(self, problem: ProblemData = None) -> None:
        self.all_vnf_done = False
        self.num_vnfs = 0
        self.my_env_step = 0
        self.over_utilized_core = False

        # Get new problem, also generates new observation
        self.get_new_problem(problem)
        self.vnf_that_went_over_limit = {}
        self.sensor.clear_state_components()

    def reset(self, problem: ProblemData = None):
        if len(self.hard_problems) > 0 and self.random.uniform() > 0.25 \
                and not self.evaluation and problem is None:
            problem = self.hard_problems.pop()
        self._reset(problem)
        # Get new ONVM Env
        obs = {}
        for i in range(self.cpu_count):
            obs[i] = self.sensor(self, i, self.my_env_step)
        return obs

    def get_system_rate(self):
        rate = sum([sfc.rate * len(sfc.jobs) for sfc in self.sfcs])
        return rate

    def assign_vnf_to_cpu(self, cpu_ids: List[int]):
        """
        assignes the current vnf to a vnf.
        If multiple TXs want the NF, the tx with the least loads gets the NF
        If no TX wants the NF, all previous and future NFs from the according sfc are valued with 0
        
        inputs:
            tx_ids: List with the TX IDs which want the NF
        """
        if len(cpu_ids) == 1:
            cpu_id = cpu_ids[0]
            all_reject = False
        elif len(cpu_ids) == 0:
            cpu_id = np.random.randint(0, self.cpu_count)
            all_reject = True
        else:
            all_reject = False
            cpu_id = np.random.choice(cpu_ids)      
        self.opened_new_cpu = self.cpu_helper[cpu_id].num_jobs == 0
        self.cpu_helper[cpu_id].add_job(self.current_vnf.job)
        self.over_utilized_core = self.cpu_helper[cpu_id].went_over_limit
        # self.sensor.update_kv(self, self.current_vnf.id, cpu_id)
        return cpu_id

    def step(self, action_dict: Dict[int, int]):
        obs, rew, done, info = {}, {}, {}, {}

        # Check which tx threads wants the nf
        # will_cpu = []
        # for cpu, choose in action_dict.items():
        #     if choose:
        #         will_cpu.append(cpu)
        cpu = None
        for cpu, choose in action_dict.items():
            if choose:
                break
        will_cpu = [cpu]

        # Assign the VNF to a TX-Thread
        cpu_id = self.assign_vnf_to_cpu(will_cpu)
        self.sensor.update_state(self, cpu_id, self.my_env_step)

        rewards = self.reward_fct(self)
        self.all_vnf_done = self.current_vnf.next()

        for i in range(self.cpu_count):
            obs[i] = self.sensor(self, i, self.my_env_step)
            rew[i] = rewards[i]
            done[i] = False
            info[i] = {}

        if self.all_vnf_done or self.cpu_helper[cpu_id].went_over_limit:
            done["__all__"] = True
            if self.over_utilized_core and len(self.hard_problems) < 100:
                self.hard_problems.append(self.current_problem)
        else:
            done["__all__"] = False
        self.my_env_step += 1
        return obs, rew, done, info


class ONVMVnfPlayerEnv(ONVMMultiAgentCPU):
    def __init__(self, env_config: Dict[str, Any]) -> 'ONVMVnfPlayerEnv':
        super(ONVMVnfPlayerEnv, self).__init__(env_config)
        self.action_space = gym.spaces.Discrete(16)
        self.reward_fct = {
            'load_balancing': LoadBalancingReward,
            'load_balancing_cfs': LoadBalancingCfsReward,
            "bin_packing": VnfPlayerBinPackingReward,
            'bin_packing_cfs': VnfPlayerBinPackingCfsReward
        }[env_config.get("reward_fct", 'bin_packing')]()
        self.vnf_to_core = {}
        self.reset()

    def reset(self, problem: ProblemData = None):
        self._reset(problem)
        self.vnf_to_core = {}
        # obs = {i: self.sensor(self, 0, 0) for i in range(self.current_vnf.get_num_nfs())}
        obs = {0: self.sensor(self, 0, 0)}
        self.my_env_step = 0
        return obs

    def assign_vnf_to_cpu(self, cpu_ids: List[int]):
        """
        assignes the current vnf to a vnf.
        If multiple TXs want the NF, the tx with the least loads gets the NF
        If no TX wants the NF, all previous and future NFs from the according sfc are valued with 0

        inputs:
            tx_ids: List with the TX IDs which want the NF
        """
        if len(cpu_ids) == 1:
            cpu_id = cpu_ids[0]
        else:
            raise ValueError(f"Expected cpu_ids to have only one element, got {len(cpu_ids)}")
        overutilizes = self.cpu_helper[cpu_id].would_go_over_limit(self.current_vnf.job)
        self.vnf_that_went_over_limit[self.current_vnf.job] = overutilizes
        self.opened_new_cpu = self.cpu_helper[cpu_id].num_jobs == 0
        self.cpu_helper[cpu_id].add_job(self.current_vnf.job)
        self.cpu_helper[cpu_id].went_over_limit = self.cpu_helper[cpu_id].went_over_limit or overutilizes
        self.over_utilized_core = self.cpu_helper[cpu_id].went_over_limit
        return cpu_id

    def step(self, action_dict: Dict[int, int]) -> Tuple[Dict[Any, Any], Dict[Any, Any],
                                                         Dict[Any, Any], Dict[Any, Any]]:
        obs, rew, done, info = {}, {}, {}, {}
        if self.my_env_step < 0:
            rew = {k: 0. for k in action_dict.keys()}
            done['__all__'] = False
            obs[self.my_env_step + 1] = self.sensor(self, 0, self.my_env_step)
        else:
            self.vnf_to_core[self.current_vnf.job] = self.all_machines[action_dict[self.my_env_step] + self.tx_count]
            cpu_id = self.assign_vnf_to_cpu([action_dict[self.my_env_step]])
            self.sensor.update_state(self, cpu_id, self.my_env_step)

            self.all_vnf_done = self.current_vnf.next()
            if self.all_vnf_done:
                rew = self.reward_fct(self)
                done = {i: True for i in range(self.current_vnf.get_num_nfs())}
                obs = {i: self.sensor(self, 0, i) for i in range(self.current_vnf.get_num_nfs())}
                if self.over_utilized_core and len(self.hard_problems) < 100:
                    self.hard_problems.append(self.current_problem)
                done["__all__"] = True
            else:
                done["__all__"] = False
                rew[self.my_env_step + 1] = 0
                obs[self.my_env_step + 1] = self.sensor(self, cpu_id, self.my_env_step + 1)
        self.my_env_step += 1
        return obs, rew, done, info


class NegativeWhenOverLimit(object):
    def __call__(self, env: ONVMMultiAgentCPU) -> Dict[int, float]:
        rewards = {}
        for i in range(env.cpu_count):
            rew_i = (env.cpu_helper[i].machine.capacity - env.cpu_helper[i].get_total_demand()) / env.cpu_helper[i].machine.capacity
            if rew_i >= 0.0:
                rew_i = 0.0
            rewards[i] = rew_i
        return rewards


class BinPackingReward(object):
    def __call__(self, env: ONVMMultiAgentCPU) -> Dict[int, float]:
        reward = env.reward_factor * (-1 * float(env.over_utilized_core) - 0.5 * float(env.opened_new_cpu))
        return {i: reward for i in range(env.cpu_count)}


class AdvancedBinPackingReward(object):
    def __init__(self, model_path: str):
        self.model: SklearnTemplate = load(model_path)

    def __call__(self, env: ONVMMultiAgentCPU, actions: Dict[int, int]) -> Dict[int, float]:
        job = env.current_vnf.job
        x = np.array([env.cpu_helper])
        utilization_inc = job.demand / 2.2e9
        rewards = {}
        num_used_cpus = 0.
        for cpu_helper in env.cpu_helper:
            num_used_cpus += float(cpu_helper.num_jobs > 0)
        num_used_cpus += float(num_used_cpus == 0)

        for cpu_idx, action in actions.items():
            utilization_after = env.cpu_helper[cpu_idx].total_demand + action * utilization_inc
            if utilization_after < 2.2e9 and action == 1:
                rewards[cpu_idx] = 0.
            elif utilization_after < 2.2e9 and action == 0:
                rewards[cpu_idx] = 1. / num_used_cpus
            elif utilization_after >= 2.2e9 and action == 1:
                rewards[cpu_idx] = 0.
            else:
                rewards[cpu_idx] = 1. / num_used_cpus


class VnfPlayerBinPackingReward(object):
    def __call__(self, env: ONVMVnfPlayerEnv) -> Dict[int, float]:
        machine_loads = {}
        for i, cpu_helper in enumerate(env.cpu_helper):
            machine_loads[i] = 0. if cpu_helper.num_jobs == 0 \
                else np.min([2.2e9, np.sum([j.demand for j in cpu_helper.nf_jobs if not env.vnf_that_went_over_limit[j]])])
        # rewards = {i: -2. * env.reward_factor for i in range(env.current_vnf.get_num_nfs())}
        rewards = {}
        for i, cpu_helper in enumerate(env.cpu_helper):
            if cpu_helper.num_jobs == 0: continue
            for nf in cpu_helper.nf_jobs:
                if env.vnf_that_went_over_limit[nf]:
                    rewards[env.vnf_to_player_id[nf]] = -10. * env.reward_factor
                else:
                    is_on_sock2 = bool(int((i + env.tx_count + 3) / 12))
                    factor = env.reward_factor * 2. if is_on_sock2 else 1.
                    rewards[env.vnf_to_player_id[nf]] = -1. * nf.demand / machine_loads[i] * factor
        return rewards


class VnfPlayerBinPackingCfsReward(object):
    """
    Uses the 1/N overload prediction estimation.
    Same code as in the VnfPlayeBinPackingReward, just the type is needed
    for differentiation in the environment.
    """
    def __call__(self, env: ONVMVnfPlayerEnv) -> Dict[int, float]:
        machine_loads = {}
        for i, cpu_helper in enumerate(env.cpu_helper):
            machine_loads[i] = 0. if cpu_helper.num_jobs == 0 \
                else np.min([2.2e9, np.sum([j.demand for j in cpu_helper.nf_jobs if not env.vnf_that_went_over_limit[j]])])
        # rewards = {i: -2. * env.reward_factor for i in range(env.current_vnf.get_num_nfs())}
        rewards = {}
        for i, cpu_helper in enumerate(env.cpu_helper):
            if cpu_helper.num_jobs == 0: continue
            for nf in cpu_helper.nf_jobs:
                if env.vnf_that_went_over_limit[nf]:
                    rewards[env.vnf_to_player_id[nf]] = -10. * env.reward_factor
                else:
                    is_on_sock2 = bool(int((i + env.tx_count + 3) / 12))
                    factor = env.reward_factor * 2. if is_on_sock2 else 1.
                    rewards[env.vnf_to_player_id[nf]] = -1. * nf.demand / machine_loads[i] * factor
        return rewards


class SequenceBasedReward(object):
    def __call__(self, env: ONVMMultiAgentCPU) -> Dict[int, float]:
        num_used = 0
        multiplicator = 1.
        for cpu_helper in env.cpu_helper:
            num_used += float(cpu_helper.num_jobs > 0)
            multiplicator *= float(not cpu_helper.went_over_limit)
        reward = (1. - num_used / len(env.cpu_helper) + 0.1) * multiplicator
        return {i: env.reward_factor * reward for i in range(env.cpu_count)}


class LoadBalancingReward(object):
    def __call__(self, env: ONVMVnfPlayerEnv) -> Dict[int, float]:
        machine_loads = {}
        for i, cpu_helper in enumerate(env.cpu_helper):
            machine_loads[i] = 0 if cpu_helper.num_jobs == 0 \
                else np.sum([j.demand for j in cpu_helper.nf_jobs])
        rewards = {}
        for i, cpu_helper in enumerate(env.cpu_helper):
            for nf in cpu_helper.nf_jobs:
                rewards[env.vnf_to_player_id[nf]] = -1. * env.reward_factor * machine_loads[i] / 2.2e9
        return rewards


class LoadBalancingCfsRewardShared(object):
    def __call__(self, env: ONVMVnfPlayerEnv) -> Dict[int, float]:
        machine_loads = {}
        for i, cpu_helper in enumerate(env.cpu_helper):
            machine_loads[i] = 0 if cpu_helper.num_jobs == 0 \
                else np.sum([j.demand for j in cpu_helper.nf_jobs])
        max_load = np.max(list(machine_loads.values()))
        rewards = {}
        for i, cpu_helper in enumerate(env.cpu_helper):
            for nf in cpu_helper.nf_jobs:
                if env.vnf_that_went_over_limit[nf]:
                    rewards[env.vnf_to_player_id[nf]] = -10. * env.reward_factor
                else:
                    rewards[env.vnf_to_player_id[nf]] = -1. * env.reward_factor * max_load / 2.2e9
        return rewards


class LoadBalancingCfsReward(object):
    def __call__(self, env: ONVMVnfPlayerEnv) -> Dict[int, float]:
        machine_loads = {}
        for i, cpu_helper in enumerate(env.cpu_helper):
            machine_loads[i] = 0 if cpu_helper.num_jobs == 0 \
                else np.sum([j.demand for j in cpu_helper.nf_jobs])
        rewards = {}
        for i, cpu_helper in enumerate(env.cpu_helper):
            for nf in cpu_helper.nf_jobs:
                if env.vnf_that_went_over_limit[nf]:
                    rewards[env.vnf_to_player_id[nf]] = -10. * env.reward_factor
                else:
                    rewards[env.vnf_to_player_id[nf]] = -1. * env.reward_factor * machine_loads[i] / 2.2e9
        return rewards


def setup_train(num_episodes: int = 100):
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='episode_reward_mean',
        mode='max',
        max_t=2000,
        grace_period=500,
        reduction_factor=2
    )

    scheduler_pbt = PopulationBasedTraining(
        time_attr='training_iteration',
        perturbation_interval=50,
        hyperparam_mutations={
            'lr': lambda: np.random.uniform(0.0001, 0.1)
        },
        metric='episode_reward_mean',
        mode='max'
    )

    # Create Config
    config = sac.DEFAULT_CONFIG.copy()
    config["env"] = env_name
    config["env_config"] = {
        "machine_count": 21,
        "cycles_per_second": 2.2e9,
        "num_of_sfcs": 8,
        "max_vnfs_per_sfc": 8,
        "max_vnfs": 21,
        "min_vnfs": 8,
        "load_level": 0.9,
        "rate": 5e6,
        "reward_factor": 1.0,
        "seed": 1
        }
    config["optimization"] = {
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "entropy_learning_rate": 3e-4,
    }
    config["framework"] = "torch"
    config["num_workers"] = 45 #45
    config["num_envs_per_worker"] = 100
    config["explore"] = True
    config["lr"] = tune.uniform(0.00001, 0.001) #tune.sample_from(lambda x: np.random.choice([0.1, 0.01, 0.001, 0.0001]))
    config["normalize_actions"] = False

    stop = {
        "episode_reward_mean": 1e12
    }

    single_env = env_creator(config["env_config"])
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    print("ACTIVATION SPACE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", act_space)
    num_agents = single_env.max_vnfs

    # Multiagent config
    config["multiagent"] = {
        "policies": {
            'all-tx': (None, obs_space, act_space, {})
        },
        "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: 'all-tx'
    }
    config['model'] = {
        'custom_model': 'CPUPlayerSACModel',
        'custom_model_config': {
            'mha_cpu': {
                'num_attn_heads': 3,
                'dim_attn_hidden': 32,
                'dim_attn_out': 8,
                'dim_attn_q': 3,
                'dim_attn_kv': 2 + 16,
                'dim_fcn': 32
            },
            'mha_vnf': {
                'num_attn_heads': 3,
                'dim_attn_hidden': 32,
                'dim_attn_out': 8,
                'dim_attn_q': 3,  #'fill-in-setup',
                'dim_attn_kv': 3,
                'dim_fcn': 32
            },
            "output_model": {
                'dim_input': 1 * 32 + 3,
                'dim_hidden': [100, 100]
                # 'dim_hidden': tune.sample_from(
                #     lambda x: [int(x) for x in np.random.randint(16, 64, size=np.random.randint(1, 4))])

            }
        }
    }
    config['buffer_size'] = tune.sample_from(lambda x: np.random.randint(1e4, 1e6))
    # Evaluation Config
    config['evaluation_interval'] = 10
    config['evaluation_num_workers'] = 0
    config['evaluation_num_episodes'] = 10
    config['evaluation_config'] = {
        'explore': False,
        'env_config': {
            'evaluation_seed': 1
        }
    }
    ray.init()
    results = tune.run(
        sac.SACTrainer,
        scheduler=scheduler,
        # resources_per_trial={'gpu': 0, 'cpu': 40},
        num_samples=100,
        config=config,
        stop=stop,
        checkpoint_at_end=True,
        checkpoint_freq=25,
        # queue_trials=True,
        # metric='episode_reward_mean',
        checkpoint_score_attr='episode_reward_mean',
        # mode='max',
        #restore=RESTORE_PATH,
        local_dir='/opt/project/Net/',
        name='VnfPlayer2',
        trial_name_creator=lambda trial: 'StateTrainable_{:s}'.format(trial.trial_id[:11])
        )

    print(f'Found CP: {results.best_checkpoint}')


def setup_train_twinq(num_episodes: int = 100):
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='episode_reward_min',
        mode='max',
        max_t=2000,
        grace_period=500,
        reduction_factor=2
    )

    scheduler_pbt = PopulationBasedTraining(
        time_attr='training_iteration',
        perturbation_interval=50,
        hyperparam_mutations={
            'lr': lambda: np.random.uniform(0.00001, 0.001)
        },
        metric='episode_reward_min',
        mode='max'
    )

    # Create Config
    config = dqn.DEFAULT_CONFIG.copy()
    config["env"] = env_name
    config['learning_starts'] = 100
    config['train_batch_size'] = tune.randint(256, 1025)
    config['exploration_config'] = {
        'type': 'EpsilonGreedy',
        'initial_epsilon': 1.,
        'final_epsilon': tune.uniform(0.01, 0.1),
        'epsilon_timesteps': tune.randint(500000, 1000000)
    }
    config["env_config"] = {
        "machine_count": 21,
        "cycles_per_second": 2.2e9,
        "num_of_sfcs": 8,
        "max_vnfs_per_sfc": 8,
        "max_vnfs": 21,
        "min_vnfs": 8,
        "load_level": 0.9,
        "rate": 5e6,
        "reward_factor": 1.0,
        "seed": 1,
        "overload_prediction": False,
        "overload_model_path": '/opt/project/rf_soft.model',
        "evaluation": False
    }
    config["framework"] = "torch"
    config["num_workers"] = 5  #45
    config["num_envs_per_worker"] = 900
    config["explore"] = True
    config["lr"] = tune.uniform(0.00001, 0.001) #tune.sample_from(lambda x: np.random.choice([0.1, 0.01, 0.001, 0.0001]))
    config["normalize_actions"] = False

    stop = {
        "training_iteration": 2000,
        "episode_reward_mean": 1e12
    }

    single_env = env_creator(config["env_config"])
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    print("ACTIVATION SPACE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", act_space)
    num_agents = single_env.max_vnfs

    # Multiagent config
    config["multiagent"] = {
        "policies": {
            'all-tx': (None, obs_space, act_space, {})
        },
        "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: 'all-tx'
    }
    # config['fcnet_hiddens'] = [100, 100]
    # config['fcnet_activation'] = 'relu'
    config['hiddens'] = tune.sample_from(lambda x: [int(x) for x in np.random.randint(16, 125, size=np.random.randint(1, 4))])
    config['model'] = {
        'custom_model': 'CPUPlayerDQNModel',
        # 'fcnet_hiddens': tune.sample_from(lambda x: [int(x) for x in np.random.randint(16, 125, size=np.random.randint(1, 4))]), # [50, 50],
        # 'fcnet_activation': 'relu',
        'custom_model_config': {
            # "output_model": {
            #     'dim_input': 'fill-in-constructor',  #1 * 32 + 3,
            #     'dim_hidden': tune.sample_from(lambda x: [int(x) for x in np.random.randint(16, 125, size=np.random.randint(1, 4))]), #[100, 100]
            #     # 'dim_hidden': tune.sample_from(
            #     #     lambda x: [int(x) for x in np.random.randint(16, 64, size=np.random.randint(1, 4))])

            # },
            # 'mha_cpu': {
            #     'num_attn_heads': tune.randint(1, 6),
            #     'dim_attn_hidden': tune.randint(8, 32),
            #     'dim_attn_out': tune.randint(8, 32),
            #     'dim_attn_q': 3,
            #     'dim_attn_kv': 4 + 7,
            #     'dim_fcn': tune.randint(8, 128)
            # },
            'q_hiddens': [tune.randint(16, 65)],
            'cpu_embd': {
                'dim_out': tune.randint(1, 5),
                'hiddens': tune.sample_from(lambda x: [int(x) for x in np.random.randint(16, 125, size=np.random.randint(1, 4))])
            },
            'mha_vnf': {
                'num_attn_heads': tune.randint(1, 6),
                'dim_attn_hidden': tune.randint(8, 32),
                'dim_attn_out': tune.randint(8, 32),
                'dim_attn_q': 3,  #'fill-in-setup',
                'dim_attn_kv': 3,  #  + 7,
                'dim_fcn': tune.randint(8, 65)
            }
        }
    }
    config['buffer_size'] = tune.sample_from(lambda x: np.random.randint(1e4, 1e6))

    # Evaluation Config
    config['evaluation_interval'] = 10
    config['evaluation_num_workers'] = 0
    config['evaluation_num_episodes'] = 100
    config['evaluation_config'] = {
        'explore': False,
        'env_config': {
            'evaluation_seed': 1,
            "evaluation": True
        }
    }
    ray.init()
    results = tune.run(
        dqn.DQNTrainer,
        scheduler=scheduler,
        # resources_per_trial={'gpu': 0, 'cpu': 40},
        num_samples=100,
        config=config,
        stop=stop,
        keep_checkpoints_num=10,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        # queue_trials=True,
        # metric='episode_reward_mean',
        checkpoint_score_attr='episode_reward_min',
        # mode='max',
        #restore=RESTORE_PATH,
        local_dir='/opt/project/Net/',
        name='TuneVnfPlayerSingleCpu',
        trial_dirname_creator=lambda trial: trial.trial_id,
        trial_name_creator=lambda trial: f'StateTrainable_{uuid.uuid4().hex}' # .format(trial.trial_id[:11])
    )

    print(f'Found CP: {results.best_checkpoint}')


def get_user_input(thing_to_choose: str, options: List[str]) -> str:
    user_input = 0
    parts = '\n\t'.join([f'{i + 1}) {s}' for i, s in enumerate(options)])
    while not (1 <= user_input <= len(options)):
        try:
            user_input = int(input(f"Choose {thing_to_choose}:\n\t{parts}\nEnter number:"))
        except:
            pass
    choice = options[user_input - 1]
    print(f"Yout choice: {choice}\n---")
    return choice


def setup_train_twinq2(num_episodes: int = 100):
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='episode_reward_mean',
        mode='max',
        max_t=2500,
        grace_period=500,
        reduction_factor=2
    )

    scheduler_pbt = PopulationBasedTraining(
        time_attr='training_iteration',
        perturbation_interval=10,
        hyperparam_mutations={
            'lr': lambda: np.random.uniform(0.00001, 0.001)
        },
        metric='episode_reward_min',
        mode='max'
    )
    reward_function = get_user_input(
        'reward function',
        ['bin_packing_cfs', 'bin_packing', 'load_balancing_cfs', 'load_balancing']
    )
    overload_model = get_user_input('Overload Model', ['None', 'LogRegActor', 'RandomForestActor', 'PerModelActor'])
    model_path = {
        'None': None,
        'RandomForestActor': '/opt/project/rf_soft.model',
        'PerfModelActor': '/model/does/not/exist',
        'LogRegActor': 'static'
    }
    load_level = -1.
    while not (0 < load_level <= 1.):
        try:
            load_level = float(input("Choose load level in (0, 1]: "))
        except:
            pass
    print(f"Your choice: {load_level}\n---")

    # Create Config
    config = dqn.DEFAULT_CONFIG.copy()
    config["env"] = env_name
    config['learning_starts'] = 100
    config['train_batch_size'] = 379
    config['exploration_config'] = {
        'type': 'EpsilonGreedy',
        'initial_epsilon': 1.,
        'final_epsilon': 0.05,
        'epsilon_timesteps': 531901
    }
    config["env_config"] = {
        "machine_count": 21,
        "cycles_per_second": 2.2e9,
        "num_of_sfcs": 5,
        "max_vnfs_per_sfc": 8,
        "max_vnfs": 32,
        "min_vnfs": 5,
        "load_level": load_level,
        "rate": 5e6,
        "reward_factor": 1.0,
        "overload_prediction": overload_model != 'None',
        'overload_model_class': overload_model,
        "overload_model_path": model_path,  #'/opt/project/rf_soft.model',
        "seed": 1,
        "problem_generator": "GoldenSampleGenerator",
        "reward_fct": reward_function  # "bin_packing_cfs"  #"load_balancing_cfs"  #  "bin_packing"  #
    }
    config["framework"] = "torch"
    config["num_workers"] = 8  # 2  #45
    config["num_envs_per_worker"] = 625  # 2500
    config["explore"] = True
    config["lr"] = 0.000879 #tune.sample_from(lambda x: np.random.choice([0.1, 0.01, 0.001, 0.0001]))
    config["normalize_actions"] = False

    stop = {
        "training_iteration": 5000,
        "episode_reward_mean": 1e12
    }

    ray.init(namespace=NAMESPACE, _redis_max_memory=10**9, object_store_memory=10**10)
    time.sleep(5)
    print("Check if an actor is needed.")
    if config["env_config"]["overload_prediction"] and \
            config['env_config']['overload_model_class'] == 'RandomForestActor':
        print("Create actor, sleep for 5s.")
        actor = [RandomForestActor \
            .options(name=config["env_config"]["overload_model_path"] + f'-{i}') \
            .remote(model_path=config["env_config"]["overload_model_path"]) for i in range(48)]
        time.sleep(5)
    else:
        print("Did not create an actor")
        actor = None

    single_env = env_creator(config["env_config"])
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    print("ACTIVATION SPACE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", act_space)
    num_agents = single_env.max_vnfs

    # Multiagent config
    config["multiagent"] = {
        "policies": {
            'all-tx': (None, obs_space, act_space, {})
        },
        "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: 'all-tx'
    }
    # config['fcnet_hiddens'] = [100, 100]
    # config['fcnet_activation'] = 'relu'
    config['hiddens'] = [71]
    config['model'] = {
        'custom_model': 'CPUPlayerDQNModel',
        # 'fcnet_hiddens': [110, 28, 52], # [50, 50],
        # 'fcnet_activation': 'relu',
        'custom_model_config': {
            # "output_model": {
            #     'dim_input': 'fill-in-constructor',  #1 * 32 + 3,
            #     'dim_hidden': tune.sample_from(lambda x: [int(x) for x in np.random.randint(16, 125, size=np.random.randint(1, 4))]), #[100, 100]
            #     # 'dim_hidden': tune.sample_from(
            #     #     lambda x: [int(x) for x in np.random.randint(16, 64, size=np.random.randint(1, 4))])

            # },
            # 'mha_cpu': {
            #     'num_attn_heads': tune.randint(1, 6),
            #     'dim_attn_hidden': tune.randint(8, 32),
            #     'dim_attn_out': tune.randint(8, 32),
            #     'dim_attn_q': 3,
            #     'dim_attn_kv': 4 + 7,
            #     'dim_fcn': tune.randint(8, 128)
            # },
            'q_hiddens': [52],
            'cpu_embd': {
                'dim_out': 8,
                'hiddens': [102, 75, 94]
            },
            'mha_vnf': {
                'num_attn_heads': 3,
                'dim_attn_hidden': 26,
                'dim_attn_out': 26,
                'dim_attn_q': 3,  #'fill-in-setup',
                'dim_attn_kv': 3,
                'dim_fcn': 20
            }
        }
    }
    config['buffer_size'] = 760000

    # Evaluation Config
    config['evaluation_interval'] = 1000
    config['evaluation_num_workers'] = 0
    config['evaluation_num_episodes'] = 100
    config['evaluation_config'] = {
        'explore': False,
        'env_config': {
            'evaluation_seed': 1,
            'evaluation': True
        }
    }
    load_part = f'{int(load_level * 100):d}'
    rwd_fct_part = {
        'bin_packing_cfs': 'BinPackingOneOverN',
        'bin_packing': 'BinPackingRatio',
        'load_balancing_cfs': 'LoadBalancingSingleOneOverN',
        'load_balancing': 'LoadBalancingSharedRatio'
    }[reward_function]
    model_part = {
        'None': '',
        'LogRegActor': 'LogReg',
        'RandomForestActor': 'RF',
        'PerfModelActor': 'GNN'
    }[overload_model]
    results = tune.run(
        dqn.DQNTrainer,
        scheduler=scheduler_pbt,
        # resources_per_trial={'gpu': 0, 'cpu': 40},
        num_samples=5,
        config=config,
        stop=stop,
        keep_checkpoints_num=10,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        # queue_trials=True,
        # metric='episode_reward_min',
        checkpoint_score_attr='episode_reward_min',
        # mode='max',
        #restore=RESTORE_PATH,
        local_dir='/opt/project/Net/',
        name=f'TuneVnfPlayerCpuOnlyCpuRatios{load_part}{rwd_fct_part}{model_part}GoldenSamples',
        trial_dirname_creator=lambda trial: trial.trial_id,
        trial_name_creator=lambda trial: f'StateTrainable_{uuid.uuid4().hex}'  # .format(trial.trial_id[:11])
    )

    print(f'Found CP: {results.best_checkpoint}')


def get_normalizers():
    config = {
        "machine_count": 21,
        "cycles_per_second": 2.2e9,
        "num_of_sfcs": 8,
        "max_vnfs_per_sfc": 8,
        "max_vnfs": 21,
        "min_vnfs": 8,
        "load_level": 0.9,
        "rate": 5e6,
        "reward_factor": 1.0,
        "seed": 1
    }
    env = ONVMMultiAgentCPU(config)
    min_compute = 1e12
    max_compute = 0
    min_rate = 1e12
    max_rate = 0
    min_demand = 1e19
    max_demand = 0
    min_sfcs = 1e9
    max_sfcs = 0
    min_vnfs = 1e9
    max_vnfs = 0
    num_nfs = []
    for i in range(10000):
        env.reset()
        min_vnfs = int(np.min([min_vnfs, len(env.current_vnf.nf_list)]))
        max_vnfs = int(np.max([max_vnfs, len(env.current_vnf.nf_list)]))
        num_nfs.append(len(env.current_vnf.nf_list))
        min_sfcs = int(np.min([min_sfcs, len(env.current_vnf.sfcs)]))
        max_sfcs = int(np.max([max_sfcs, len(env.current_vnf.sfcs)]))
        for vnf in env.current_vnf.nf_list:
            min_compute = int(np.min([min_compute, vnf.vnf.compute_per_packet]))
            max_compute = int(np.max([max_compute, vnf.vnf.compute_per_packet]))
            min_rate = int(np.min([min_rate, vnf._rate]))
            max_rate = int(np.max([max_rate, vnf._rate]))
            min_demand = int(np.min([min_demand, vnf.demand]))
            max_demand = int(np.max([max_demand, vnf.demand]))
    print(f"Compute: {{{min_compute:d}, ..., {max_compute:d}}}")
    print(f"Rate: {{{min_rate:d}, ..., {max_rate:d}}}")
    print(f"Demand: {{{min_demand:d}, ..., {max_demand:d}}}")
    print(f"Num Sfcs: {{{min_sfcs:d}, ..., {max_sfcs:d}}}")
    print(f"Num Vnfs: {{{min_vnfs:d}, ..., {max_vnfs:d}}}")


def try_env():
    config = {
        "machine_count": 21,
        "cycles_per_second": 2.2e9,
        "num_of_sfcs": 5,
        "max_vnfs_per_sfc": 8,
        "max_vnfs": 32,
        "min_vnfs": 5,
        "load_level": .8,
        "rate": 2.5e6,
        "reward_factor": 1.0,
        "overload_prediction": True,
        'overload_model_class': 'LogRegActor',
        "overload_model_path": '/opt/project/rf_soft.model',
        "seed": 1,
        "problem_generator": "GoldenSampleGenerator",
        "reward_fct": "bin_packing_cfs" #"load_balancing_cfs" # "bin_packing"  # "load_balancing"
    }
    env = ONVMVnfPlayerEnv(config)

    all_rewards = []
    obs = env.reset()
    all_done = False
    while not all_done:
        act = {k: np.random.randint(0, 1) for k in obs.keys()}
        obs, rew, done, info = env.step(act)
        sample = {
            f'vnf_player_{k}': {kk: vv.detach().numpy() for kk, vv in env.sensor.deserialize(torch.tensor(v)).items()}
            for k, v in obs.items()}
        all_done = done['__all__']
        all_rewards.append(rew)

    all_rewards = []
    obs = env.reset()
    all_done = False
    while not all_done:
        act = {k: np.random.randint(0, 2) for k in obs.keys()}
        obs, rew, done, info = env.step(act)
        sample = {
            f'vnf_player_{k}': {kk: vv.detach().numpy() for kk, vv in env.sensor.deserialize(torch.tensor(v)).items()}
            for k, v in obs.items()}
        all_done = done['__all__']
        all_rewards.append(rew)
    print('done')


def env_creator(config):
    # return ONVMMultiAgentCPU(config)
    return ONVMVnfPlayerEnv(config)


if __name__ == '__main__':
    # switch = 'test_env'
    # In new container edit vim /opt/conda/lib/python3.8/site-packages/ray/rllib/evaluation/episode.py
    # to avoid the birthday paradoxon.

    user_in = 0
    while user_in not in [1, 2]:
        try:
            user_in = int(input("Execution mode:\n\t1) twinq\n\t2) test_env\nEnter number: "))
        except:
            pass
    switch = {1: 'twinq', 2: 'test_env'}[user_in]

    # switch = 'test_env'
    if switch == 'test_env':
        try_env()
    else:
        # get_normalizers()
        # ray.init()

        # env_name = "ONVMMultiAgentCPU"
        env_name = "ONVMVnfPlayerEnv"
        register_env(env_name, env_creator)

        # ModelCatalog.register_custom_model('CPUPlayerModel', CPUPlayerModel)
        ModelCatalog.register_custom_model('CPUPlayerDQNModel', CPUPlayerDQNModel)
        ModelCatalog.register_custom_model('CPUPlayerSACModel', CPUPlayerSACModel)
        ModelCatalog.register_custom_model('CPUPlayerPolicyModel', CPUPlayerPolicyModel)
        ModelCatalog.register_custom_model('CPUPlayerWithVnfObjectsModel', CPUPlayerWithVnfObjectsModel)

        #pbt()
        #asha()
        if switch == 'sac':
            setup_train(500)
        elif switch == 'twinq':
            print("setup_train_twinq")
            setup_train_twinq2(500)
        ray.shutdown()
