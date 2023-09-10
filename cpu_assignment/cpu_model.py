from __future__ import annotations

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn
import gym
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.sac.sac_torch_model import SACTorchModel
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from typing import Dict, Any, List, Optional, Tuple
from model_components import attention_layer as al
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.typing import ModelConfigDict
import numpy as np
from gym.spaces import Box
# from rl_cpu_assignment import ONVMMultiAgentCPU


def positional_encoding(pos: int, dim: int) -> float:
    k = np.repeat(np.arange(dim / 2), repeats=2)
    mask = np.tile(np.array([0., 1.]), reps=int(dim/2))
    w = 1. / 10000 ** (2 * k / dim)
    return (1 - mask) * np.cos((pos + 1) * w) + mask * np.sin((pos + 1) * w)


class CPUSensor(object):
    def __init__(self) -> None:
        super().__init__()
        self.max_num_cpu = 21
        self.max_num_vnfs = 40

        # Dimensions
        self.dim_kv = (self.max_num_vnfs, 1)
        self.dim_mask = (self.max_num_vnfs, 1)
        self.dim_q = (1, self.max_num_cpu + 1)
        self.dim_state = (1, self.max_num_cpu)

        self.kv = np.zeros(shape=self.dim_kv, dtype=np.float32)
        self.q = np.zeros(shape=self.dim_q, dtype=np.float32)
        self.cpu_state = np.zeros(shape=self.dim_state, dtype=np.float32)
        self.mask = np.zeros(shape=self.dim_mask, dtype=np.float32)

    @property
    def observation_space(self):
        #REFACTOR
        return gym.spaces.Box(-2.0, 1e9, shape=(1, self.q.size + self.kv.size + self.cpu_state.size + 1 + self.mask.size))

    def set_kv(self, env):
        cap = env.cpu_helper[0].machine.capacity
        self.cpu_state = np.zeros(shape=self.dim_state, dtype=np.float32)
        self.kv = np.zeros(shape=self.dim_kv, dtype=np.float32)
        num = 0
        for i, job in enumerate(env.current_vnf.nf_list):
                # Resource concerning elements
                self.kv[num, 0] = job.demand / cap
                num += 1
        
        self.mask[:num, :] = 1
        self.mask[num:, :] = 0

    def update_kv(self, env, nf_id, cpu_id):
        cap = env.cpu_helper[cpu_id].machine.capacity
        self.mask[:cpu_id, :] = 0
        self.cpu_state[0, cpu_id] += self.kv[nf_id, 0] * cap

    def serialize(self, kv, mask, q, state, current_load) -> np.array:
        obs = np.concatenate([
            kv.flatten(),
            mask.flatten(),
            q.astype('float32').flatten(),
            state.astype('float32').flatten(),
            np.array([current_load], dtype=np.float32)
        ])
        return np.expand_dims(obs, axis=0)

    def deserialize(self, vec: np.array):
        kv_start, kv_end = 0, self.kv.size
        mask_start, mask_end = kv_end, kv_end + self.mask.size
        q_start, q_end = mask_end, mask_end + self.q.size
        state_start, state_end = q_end, q_end + self.cpu_state.size + 1 #+ 7

        kv = torch.reshape(vec[:, kv_start:kv_end ], [vec.shape[0], self.kv.shape[0], self.kv.shape[1]])
        mask = torch.reshape(vec[:, mask_start:mask_end ], [vec.shape[0], self.mask.shape[0], self.mask.shape[1]])
        q = torch.reshape(vec[:, q_start:q_end ], [vec.shape[0], self.q.shape[0], self.q.shape[1]])
        state = torch.reshape(vec[:, state_start:state_end ], [vec.shape[0], self.cpu_state.shape[1] + 1]) #+7
        return kv, mask, q, state

    def sense(self, env, cpu_id: int) -> np.array:
        cap = env.cpu_helper[cpu_id].machine.capacity
        cpu_load = self.cpu_state[0, cpu_id] / cap - 1
        q = self.q.copy()
        q[0, -1] = cpu_load
        sorted_load = np.sort(self.cpu_state, axis=-1)
        q[0, :self.cpu_state.shape[1]] = sorted_load[0, :]

        return self.serialize(self.kv, self.mask, q / cap - 1, sorted_load / cap - 1, cpu_load)

    def __call__(self, env, cpu_id) -> np.array:
        return self.sense(env, cpu_id)


class CpuAndVnfSensor(object):
    """
        Ranges of variables of interest:
        Compute: {175, ..., 7737}
        Rate: {50271, ..., 5000000}
        Demand: {10560550, ..., 1979999732}
        Num Sfcs: {1, ..., 8}
        Num Vnfs: {1, ..., 21}
    """
    def __init__(self) -> None:
        super().__init__()
        self.max_num_cpu = 16
        self.max_num_vnfs = 32

        # Dimensions
        self.dim_pos_encoding = 7
        self.dim_kv_vnf = (self.max_num_vnfs, 3)  # + self.dim_pos_encoding)
        # self.dim_kv_vnf = (self.max_num_vnfs, 3 + 5 + self.max_num_cpu)  # + self.dim_pos_encoding)
        self.dim_mask_vnf = (self.max_num_vnfs, 1)
        self.dim_q_vnf = (1, 3)
        # self.dim_q_vnf = (1, 3 + 5 + self.max_num_cpu)
        # Num Jobs, Demand, Compute, Rate, Min ratio, Max ratio, ratio player on VNF, Socket1, Socket2
        self.dim_kv_cpu = (self.max_num_cpu, 8)  # + self.dim_pos_encoding)
        # Num Jobs, Demand, Compute, Rate, Socket1, Socket2
        # self.dim_kv_cpu = (self.max_num_cpu, 6)  # + self.dim_pos_encoding)
        self.dim_mask_cpu = (self.max_num_cpu, 1)
        self.dim_q_cpu = (1, self.dim_kv_cpu[1])
        self.dim_state = (1, 3)

        self.kv_vnf = np.zeros(shape=self.dim_kv_vnf, dtype=np.float32)
        self.q_vnf = np.zeros(shape=self.dim_q_vnf, dtype=np.float32)
        self.mask_vnf = np.zeros(shape=self.dim_mask_vnf, dtype=np.float32)
        self.kv_cpu = np.zeros(shape=self.dim_kv_cpu, dtype=np.float32)
        self.q_cpu = np.zeros(shape=self.dim_q_cpu, dtype=np.float32)
        self.mask_cpu = np.zeros(shape=self.dim_mask_cpu, dtype=np.float32)
        self.cpu_state = np.zeros(shape=self.dim_state, dtype=np.float32)
        self.state_components = [
            self.kv_vnf,
            self.kv_cpu,
            self.mask_vnf,
            self.mask_cpu,
            self.q_vnf
            # self.q_cpu
            # self.cpu_state
        ]
        self.state_component_names = ['kv_vnf', 'kv_cpu', 'mask_vnf', 'mask_cpu',
                                      'q_vnf']  # , 'q_cpu']
        self.step_counter = 0
        self.pos_encoding = np.row_stack([positional_encoding(i, 16)[:self.dim_pos_encoding]
                                          for i in range(np.max([self.max_num_vnfs, self.max_num_cpu]))])

    def clear_state_components(self):
        for state_component in self.state_components:
            state_component[:] = 0.
        # for i in range(self.max_num_cpu):
        #     self.kv_cpu[i, self.dim_q_cpu[-1]:] = self.pos_encoding[i, :]
        # for i in range(self.max_num_vnfs):
        #     self.kv_vnf[i, self.dim_q_vnf[-1]:] = self.pos_encoding[i, :]
        #     self.kv_cpu[i, 2 + i] = 1
        self.kv_cpu[:, 1] = -1.
        self.kv_cpu[:-12, -2] = 1.
        self.kv_cpu[-12:, -1] = 1.
        self.kv_cpu[:, 4] = -1.
        self.kv_cpu[:, 5] = -1.
        self.step_counter = 0

    @property
    def observation_space(self) -> gym.spaces.Box:
        dim = int(np.sum([x.size for x in self.state_components]))
        return gym.spaces.Box(-1e9, 1e9, shape=(1, dim))

    def _calculate_ratios(self, env, cpu_idx: int, demand: float | None) -> Tuple[np.array, float | None]:
        if len(env.cpu_helper[cpu_idx].nf_jobs) == 0 and demand is None:
            return np.array([-1.], dtype=np.float32), None
        elif len(env.cpu_helper[cpu_idx].nf_jobs) == 0 and demand is not None:
            return np.array([-1.], dtype=np.float32), 1.
        elif len(env.cpu_helper[cpu_idx].nf_jobs) > 0 and demand is None:
            ratios = np.array([j.demand for j in env.cpu_helper[cpu_idx].nf_jobs]) / env.cpu_helper[cpu_idx].total_demand
            return ratios, None
        elif len(env.cpu_helper[cpu_idx].nf_jobs) > 0 and demand is not None:
            demands = [j.demand for j in env.cpu_helper[cpu_idx].nf_jobs]
            demands.append(demand)
            ratios = np.array(demands) / np.sum(demands)
            return ratios, ratios[-1]

    def update_state(self, env, cpu_idx: int, env_step: int):
        # ratios, _ = self._calculate_ratios(env, cpu_idx, None)
        # min_jobs, max_jobs = 0, 20
        min_jobs, max_jobs = 0, 15
        # min_cost, max_cost = 175., 7737.
        min_cost, max_cost = 175., 26305.
        # min_rate, max_rate = 50271, 5e6
        min_rate, max_rate = 35548., 7.830358e+06
        self.kv_cpu[cpu_idx, 0] = (env.cpu_helper[cpu_idx].num_jobs - min_jobs) / (max_jobs - min_jobs)
        self.kv_cpu[cpu_idx, 1] = env.cpu_helper[cpu_idx].total_demand / 2.2e9 - env.load_level
        self.kv_cpu[cpu_idx, 2] = (env.cpu_helper[cpu_idx].total_compute - min_cost) / (max_cost - min_cost)
        self.kv_cpu[cpu_idx, 3] = (env.cpu_helper[cpu_idx].total_rate - min_rate) / (max_rate - min_rate)
        # self.kv_cpu[cpu_idx, 4] = np.min([np.abs(self.kv_cpu[cpu_idx, 4]), env.cpu_helper[cpu_idx].nf_jobs[-1].demand / 2.2e9])
        # self.kv_cpu[cpu_idx, 5] = np.max([self.kv_cpu[cpu_idx, 5], env.cpu_helper[cpu_idx].nf_jobs[-1].demand / 2.2e9])
        # self.kv_cpu[cpu_idx, 4] = np.min(ratios)
        # self.kv_cpu[cpu_idx, 5] = np.max(ratios)
        # self.kv_vnf[env_step, 3 + 5 + cpu_idx] = 1.
        self.mask_vnf[env_step, 0] = 0.
        self.step_counter += 1
        # self.kv_cpu[cpu_idx, 1] = env.cpu_helper[cpu_idx].total_rate
        # self.kv_cpu[cpu_idx, 1] = env.cpu_helper[cpu_idx].total_compute

    def sense(self, env, cpu_idx: int, env_step: int):
        min_jobs, max_jobs = 1, 15
        min_cost, max_cost = 175., 26305.
        min_rate, max_rate = 35548., 7.830358e+06
        cjob = env.current_vnf.nf_list[env_step]
        for cpu_idx in range(len(env.cpu_helper)):
            chelper = env.cpu_helper[cpu_idx]
            ratios, job_ratio = self._calculate_ratios(env, cpu_idx, cjob.demand)

            self.kv_cpu[cpu_idx, 0] = (chelper.num_jobs + 1 - min_jobs) / (max_jobs - min_jobs)
            self.kv_cpu[cpu_idx, 1] = (chelper.total_demand + cjob.demand) / 2.2e9 - env.load_level
            self.kv_cpu[cpu_idx, 2] = (chelper.total_compute + cjob.vnf.compute_per_packet - min_cost) / (max_cost - min_cost)
            self.kv_cpu[cpu_idx, 3] = (chelper.total_rate + cjob.rate() - min_rate) / (max_rate - min_rate)
            self.kv_cpu[cpu_idx, 4] = np.min(ratios)
            self.kv_cpu[cpu_idx, 5] = np.max(ratios)
            # self.kv_cpu[cpu_idx, 6] = job_ratio
        self.q_vnf[0, :] = self.kv_vnf[env_step, :self.dim_q_vnf[-1]]
        self.q_cpu[0, :] = self.kv_cpu[cpu_idx, :self.dim_q_cpu[-1]]
        vec = self.serialize()
        return vec

    def initialize(self, env):
        # min_cost, max_cost = 175., 7737.
        min_cost, max_cost = 175., 7671.
        # min_rate, max_rate = 50271, 5e6
        min_rate, max_rate = 35548., 2.5e6
        for idx, vnf in enumerate(env.current_vnf.nf_list):
            sfc_num = env.vnf2sfc_num[vnf]
            self.kv_vnf[idx, 0] = (vnf._rate - min_rate) / (max_rate - min_rate)
            self.kv_vnf[idx, 1] = (vnf.vnf.compute_per_packet - min_cost) / (max_cost - min_cost)
            # self.kv_vnf[idx, 2] = (vnf.demand - 10560550.) / (1979999732 - 10560550)
            self.kv_vnf[idx, 2] = vnf.demand / 2.2e9
            # self.kv_vnf[idx, 3 + sfc_num] = 1.
            self.mask_vnf[idx, 0] = 1
            idx += 1
        self.mask_cpu[:env.cpu_count, 0] = 1

    def serialize(self) -> np.array:
        return np.expand_dims(np.concatenate([x.flatten() for x in self.state_components]), axis=0)

    def deserialize(self, vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        if vec.ndim == 1:
            vec = torch.unsqueeze(vec, dim=0)
        elements = {}
        start = 0
        for state_component, state_name in zip(self.state_components, self.state_component_names):
            stop = start + state_component.size
            elements[state_name] = torch.reshape(vec[:, start:stop], [vec.shape[0]] + list(state_component.shape))
            start = stop
        return elements

    def __call__(self, env, cpu_idx: int, env_step: int):
        if self.step_counter == 0:
            self.initialize(env)
        reading = self.sense(env, cpu_idx, env_step)
        return reading


sensor = CpuAndVnfSensor()


class CPUPlayerModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        config = model_config['custom_model_config']
        dim_hidden = config['dim_hidden']
        self.attn_module = al.MultiHeadAttentionLayer(
            num_heads=config['num_attn_heads'],
            attention_class=al.SelfAttentionLayer,
            dim_in=-1,
            dim_hidden=config['dim_attn_hidden'],
            dim_out=config['dim_attn_out'],
            dim_q=config['dim_attn_q'],
            dim_k=config['dim_attn_kv'],
            dim_v=config['dim_attn_kv']
        )
        self.attn_linear = nn.Linear(config['num_attn_heads'] * config['dim_attn_out'], dim_hidden)
        self.act_fct = nn.ELU()

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, Any], state: List[Any], seq_lens: Any):
        obs = input_dict['obs']
        
        obs_s = torch.squeeze(obs)
        kv, mask, q, state = sensor.deserialize(obs_s)
        
        out, weights = self.attn_module(
            keys=kv,
            values=kv,
            queries=q,
            attention_mask = mask            
        )
        
        out = self.act_fct(self.attn_linear(out))
        out = torch.squeeze(out, dim=1)
        out = torch.cat([out, state], dim=-1)

        #input_dict["obs"] = out
        return out


class CPUPlayerWithVnfObjectsModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.cpu_attn_weights = None
        self.vnf_attn_weights = None
        config = model_config['custom_model_config']
        self.attn_module_cpu = al.MultiHeadAttentionLayer(
            num_heads=config['mha_cpu']['num_attn_heads'],
            attention_class=al.SelfAttentionLayer,
            dim_in=-1,
            dim_hidden=config['mha_cpu']['dim_attn_hidden'],
            dim_out=config['mha_cpu']['dim_attn_out'],
            dim_q=config['mha_cpu']['dim_attn_q'],
            dim_k=config['mha_cpu']['dim_attn_kv'],
            dim_v=config['mha_cpu']['dim_attn_kv']
        )
        self.attn_module_vnf = al.MultiHeadAttentionLayer(
            num_heads=config['mha_vnf']['num_attn_heads'],
            attention_class=al.SelfAttentionLayer,
            dim_in=-1,
            dim_hidden=config['mha_vnf']['dim_attn_hidden'],
            dim_out=config['mha_vnf']['dim_attn_out'],
            dim_q=config['mha_vnf']['dim_attn_q'],  # config['mha_vnf']['dim_attn_kv'] + config['mha_cpu']['dim_fcn'],
            dim_k=config['mha_vnf']['dim_attn_kv'],
            dim_v=config['mha_vnf']['dim_attn_kv']
        )
        self.attn_cpu_linear = nn.Linear(
            config['mha_cpu']['num_attn_heads'] * config['mha_cpu']['dim_attn_out'],
            config['mha_cpu']['dim_fcn']
        )
        self.attn_vnf_linear = nn.Linear(
            config['mha_vnf']['num_attn_heads'] * config['mha_vnf']['dim_attn_out'],
            config['mha_vnf']['dim_fcn']
        )
        self.act_fct = nn.ELU()

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, Any], state: List[Any], seq_lens: Any):
        obs = input_dict['obs']

        obs_s = torch.squeeze(obs)
        sample = sensor.deserialize(obs_s)

        out, weights = self.attn_module_cpu(
            keys=sample['kv_cpu'],
            values=sample['kv_cpu'],
            queries=sample['q_vnf'],
            attention_mask=sample['mask_cpu']
        )
        self.cpu_attn_weights = weights
        out = torch.squeeze(out, dim=1)
        out_cpu = self.act_fct(self.attn_cpu_linear(out))
        # q_vnf = torch.cat([torch.unsqueeze(out_cpu, dim=-2), sample['q_vnf']], dim=-1)

        out, weights = self.attn_module_vnf(
            keys=sample['kv_vnf'],
            values=sample['kv_vnf'],
            queries=sample['q_vnf'],
            attention_mask=sample['mask_vnf']
        )
        self.vnf_attn_weights = weights
        out = torch.squeeze(out, dim=1)
        out = self.act_fct(self.attn_vnf_linear(out))
        out = torch.cat(
            [
                out,
                out_cpu,
                torch.squeeze(sample['q_vnf'], dim=-2)
            ], dim=-1)
        return out


class CPUPlayerPolicyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        config = model_config['custom_model_config']
        dim_input = config['dim_input']
        dim_hidden = config['dim_hidden']

        self.layers = []

        for i, dim in enumerate(dim_hidden):
            if i == 0:
                self.layers.append(nn.Linear(dim_input, dim))
            else:
                if len(dim_hidden) > 1:
                    self.layers.append(nn.Linear(dim_hidden[i-1], dim))
                else:
                    continue

        self.out = nn.Linear(dim_hidden[-1], num_outputs)
        self.act_fct = nn.ELU()
        
    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, Any], state: List[Any], seq_lens: Any):
        obs = input_dict['obs']
        obs_s = torch.squeeze(obs, dim=-2)
        # print(obs.shape, obs_s.shape)
        sample = sensor.deserialize(obs_s)
        kv_cpu = torch.reshape(sample['kv_cpu'], shape=[-1, 32])
        out = torch.cat([kv_cpu, torch.squeeze(sample['q_vnf'], dim=-2)], dim=-1)
        for i, layer in enumerate(self.layers):
            out = self.act_fct(layer(out))
        output = self.out(out)
        return output, state


class CPUPlayerSACModel(SACTorchModel):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space,
                 num_outputs: Optional[int], model_config: ModelConfigDict, name:
            str, policy_model_config: ModelConfigDict = None,
                 q_model_config: ModelConfigDict = None, twin_q: bool = False,
                 initial_alpha: float = 1, target_entropy: Optional[float] = None):
        self._model_config = model_config
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name, policy_model_config=policy_model_config,
                         q_model_config=q_model_config, twin_q=twin_q,
                         initial_alpha=initial_alpha, target_entropy=target_entropy)
        # self.base_model = CPUPlayerWithVnfObjectsModel(obs_space, action_space, num_outputs, model_config, name)

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:

        # out = self.base_model.forward(input_dict, state, seq_lens)

        # return out, state
        return input_dict["obs"], state

    @override(SACTorchModel)
    def build_policy_model(self, obs_space, num_outputs, policy_model_config,
                           name):
        """Builds the policy model used by this SAC.

        Override this method in a sub-class of SACTFModel to implement your
        own policy net. Alternatively, simply set `custom_model` within the
        top level SAC `policy_model` config key to make this default
        implementation of `build_policy_model` use your custom policy network.

        Returns:
            TorchModelV2: The TorchModelV2 policy sub-model.
        """
        
        # policy_model_config['custom_model_config']['dim_input'] = self.model_config['custom_model_config']['dim_hidden'] + 22

        model = ModelCatalog.get_model_v2(
            obs_space,
            self.action_space,
            num_outputs,
            {
                'custom_model': 'CPUPlayerPolicyModel',
                'dim_input': self._model_config['custom_model_config']['output_model']['dim_input'],
                "custom_model_config": self._model_config['custom_model_config']['output_model']
            },
            # policy_model_config,
            framework="torch",
            name=name)
        return model

    @override(SACTorchModel)
    def build_q_model(self, obs_space, action_space, num_outputs,
                      q_model_config, name):
        """Builds one of the (twin) Q-nets used by this SAC.

        Override this method in a sub-class of SACTFModel to implement your
        own Q-nets. Alternatively, simply set `custom_model` within the
        top level SAC `Q_model` config key to make this default implementation
        of `build_q_model` use your custom Q-nets.

        Returns:
            TorchModelV2: The TorchModelV2 Q-net sub-model.
        """
        self.concat_obs_and_actions = False
        if self.discrete:
            input_space = obs_space
        else:
            orig_space = getattr(obs_space, "original_space", obs_space)
            if isinstance(orig_space, Box) and len(orig_space.shape) == 1:
                input_space = Box(
                    float("-inf"),
                    float("inf"),
                    shape=(orig_space.shape[0] + action_space.shape[0], ))
                self.concat_obs_and_actions = True
            else:
                if isinstance(orig_space, gym.spaces.Tuple):
                    spaces = list(orig_space.spaces)
                elif isinstance(orig_space, gym.spaces.Dict):
                    spaces = list(orig_space.spaces.values())
                else:
                    spaces = [obs_space]
                input_space = gym.spaces.Tuple(spaces + [action_space])

        # q_model_config['custom_model_config']['dim_input'] = self.model_config['custom_model_config']['dim_hidden'] + 22

        model = ModelCatalog.get_model_v2(
            input_space,
            action_space,
            num_outputs,
            {
                'custom_model': 'CPUPlayerPolicyModel',
                'dim_input': self._model_config['custom_model_config']['output_model']['dim_input'],
                "custom_model_config": self._model_config['custom_model_config']['output_model']
            },
            framework="torch",
            name=name)
        return model


class CPUPlayerDQNModel(DQNTorchModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, mha_vnf,
                 cpu_embd, q_hiddens, **kwargs):
        super(CPUPlayerDQNModel, self).__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=16 * cpu_embd['dim_out'],  # mha_vnf['dim_fcn'] + 4 * 16 + 3,
            model_config=model_config,
            name=name,
            q_hiddens=q_hiddens,
            **kwargs
        )
        self.attn_module_vnf = al.MultiHeadAttentionLayer(
            num_heads=mha_vnf['num_attn_heads'],
            attention_class=al.SelfAttentionLayer,
            dim_in=-1,
            dim_hidden=mha_vnf['dim_attn_hidden'],
            dim_out=mha_vnf['dim_attn_out'],
            dim_q=mha_vnf['dim_attn_q'],  # config['mha_vnf']['dim_attn_kv'] + config['mha_cpu']['dim_fcn'],
            dim_k=mha_vnf['dim_attn_kv'],
            dim_v=mha_vnf['dim_attn_kv']
        )
        # self.attn_module_cpu = al.MultiHeadAttentionLayer(
        #     num_heads=mha_cpu['num_attn_heads'],
        #     attention_class=al.SelfAttentionLayer,
        #     dim_in=-1,
        #     dim_hidden=mha_cpu['dim_attn_hidden'],
        #     dim_out=mha_cpu['dim_attn_out'],
        #     dim_q=mha_cpu['dim_attn_q'],
        #     dim_k=mha_cpu['dim_attn_kv'],
        #     dim_v=mha_cpu['dim_attn_kv']
        # )
        # self.attn_cpu_linear = nn.Linear(
        #     mha_cpu['num_attn_heads'] * mha_cpu['dim_attn_out'],
        #     mha_cpu['dim_fcn']
        # )
        self.attn_vnf_linear = nn.Linear(
            mha_vnf['num_attn_heads'] * mha_vnf['dim_attn_out'],
            mha_vnf['dim_fcn']
        )
        modules = []
        d_in = mha_vnf['dim_fcn'] + sensor.dim_q_vnf[-1] + sensor.dim_q_cpu[-1]
        for d_out in cpu_embd['hiddens']:
            modules.append(torch.nn.Linear(d_in, d_out))
            modules.append(torch.nn.ReLU())
            d_in = d_out
        modules.append(torch.nn.Linear(d_in, cpu_embd['dim_out']))
        modules.append(torch.nn.ReLU())
        self.transform_cpus = torch.nn.Sequential(*modules)
        self.act_fct = nn.ReLU()
        self.cpu_attn_weights = None
        self.vnf_attn_weights = None
        self.cpu_embd_dim_out = cpu_embd['dim_out']

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        obs = input_dict['obs']
        obs_s = torch.squeeze(obs, dim=-2)
        # print(obs.shape, obs_s.shape)
        sample = sensor.deserialize(obs_s)
        # out_cpu = torch.reshape(sample['kv_cpu'], shape=[-1, 4 * 16])
        # out = torch.cat([kv_cpu, torch.squeeze(sample['q_vnf'], dim=-2)], dim=-1)
        # out, weights = self.attn_module_cpu(
        #     keys=sample['kv_cpu'],
        #     values=sample['kv_cpu'],
        #     queries=sample['q_vnf'],
        #     attention_mask=sample['mask_cpu']
        # )
        # self.cpu_attn_weights = weights
        # out = torch.squeeze(out, dim=1)
        # out_cpu = self.act_fct(self.attn_cpu_linear(out))
        # q_vnf = torch.cat([torch.unsqueeze(out_cpu, dim=-2), sample['q_vnf']], dim=-1)

        out, weights = self.attn_module_vnf(
            keys=sample['kv_vnf'],
            values=sample['kv_vnf'],
            queries=sample['q_vnf'],
            attention_mask=sample['mask_vnf']
        )
        self.vnf_attn_weights = weights
        # out = torch.squeeze(out, dim=1)
        out = self.act_fct(self.attn_vnf_linear(out))

        out = torch.cat(
            [
                sample['kv_cpu'],
                torch.repeat_interleave(out, 16, dim=1),
                torch.repeat_interleave(sample['q_vnf'], 16, dim=1)
            ],
            dim=-1
        )
        core_embedding = self.transform_cpus(out)
        out = torch.reshape(core_embedding, shape=[-1, 16 * self.cpu_embd_dim_out])
        # out = torch.cat(
        #     [
        #         out,
        #         out_cpu,
        #         torch.squeeze(sample['q_vnf'], dim=-2)
        #     ], dim=-1)
        return out, state
