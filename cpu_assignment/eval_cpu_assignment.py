import os
import sys
import pandas as pd
from ray.tune import Analysis
import matplotlib.pyplot as plt
from joblib import load

import numpy as np
import torch
sys.path.insert(0, '/opt/project')
import ray
import time
import ray.rllib.agents.sac as sac
import ray.rllib.agents.dqn.dqn as dqn
# from cpu_assignment.cpu_model import CPUPlayerWithVnfObjectsModel, CPUPlayerSACModel, CPUPlayerPolicyModel
from cpu_assignment.cpu_model import CPUPlayerWithVnfObjectsModel, CPUPlayerSACModel, CPUPlayerPolicyModel, CpuAndVnfSensor, CPUPlayerDQNModel
from ray.rllib.models.catalog import ModelCatalog

from environment.tp_sim import *

from environment import problem_generator_2 as pgen
from evaluation.validation_data import ProblemData
from typing import Dict, List, Tuple, Any
from cpu_assignment.rl_cpu_assignment import ONVMMultiAgentCPU, ONVMVnfPlayerEnv, RandomForestActor
from evaluation import validation_data as vd
import json
import evaluation.plot_reward as pltr
import logging

logger = logging.getLogger('eval-cpu-assignment')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

# PATH = '/opt/project/Net/CPU-Assignment/SAC_ONVMMultiAgent_9d727_00000_0_2021-10-19_16-21-29/checkpoint_030000/checkpoint-30000'
# TRIAL_NAME = 'StateTrainable_2a6cc_00000_0_buffer_size=276197,lr=0.0002169_2021-12-02_18-03-34'
# PATH = f'/opt/project/Net/VnfPlayer/{TRIAL_NAME}'
# CHECKPOINT = 'checkpoint_000250/checkpoint-250'

# ana = Analysis('/opt/project/Net/TuneVnfPlayer')
# trial_dir = ana.get_best_logdir(metric='episode_reward_mean', mode='max')
trial_dir = '/opt/project/Net/TuneVnfPlayerLong/7748f_00001'
trial_dir = '/opt/project/Net/TuneVnfPlayerSingleCpu/541d6_00007'
trial_dir = '/opt/project/Net/TuneVnfPlayerCpuOnlyPbt/43a78_00006'
trial_dir = '/opt/project/Net/TuneVnfPlayerCpuOnlyRfRewardPbt/5c202_00009'
trial_dir = '/opt/project/Net/TuneVnfPlayerCpuOnly80Util/9f3db_00004'
trial_dir = '/opt/project/data/nas/TuneResults/TuneVnfPlayerCpuOnly100UtilRf/ae661_00000'
exp_dir = '/opt/project/data/nas/TuneResults/TuneVnfPlayerCpuOnly100UtilLoadBalancing'
exp_dir = '/opt/project/data/nas/TuneResults/TuneVnfPlayerCpuOnly100BinPackingOneOverNGoldenSamples'
exp_dir = '/opt/project/data/nas/TuneResults/TuneVnfPlayerCpuOnly100LoadBalancingOneOverNGoldenSamples'
trial_dir = os.path.join(exp_dir, 'ae919_00009')
# checkpoint_dir = ana.get_best_checkpoint(trial_dir, 'episode_reward_mean', 'max')
# checkpoint_dir = os.path.join(trial_dir, 'checkpoint_005960/checkpoint_5960')
# checkpoint_dir = '/opt/project/Net/TuneVnfPlayerLong/7748f_00001/checkpoint_005960/checkpoint-5960'
checkpoint_dir = os.path.join(trial_dir, 'checkpoint_003190/checkpoint-3190')

# Trained out 80% util.
checkpoint_dir = '/opt/project/Net/TuneVnfPlayerCpuOnly80Util/9f3db_00004/checkpoint_005000/checkpoint-5000'
# Trained out 100% util with RF.
# checkpoint_dir = '/opt/project/Net/TuneVnfPlayerCpuOnly100UtilRf/ae661_00000/checkpoint_003190/checkpoint-3190'
checkpoint_dir = '/opt/project/Net/TuneVnfPlayerCpuOnly100UtilRf/ae661_00004/checkpoint_005000/checkpoint-5000'
checkpoint_dir = os.path.join(exp_dir, '41ccd_00009/checkpoint_002990/checkpoint-2990')
checkpoint_dir = os.path.join(trial_dir, 'checkpoint_005000/checkpoint-5000')

print(checkpoint_dir)


def get_sim_system_rate(env: ONVMMultiAgentCPU, is_rl: bool = True) -> float:
    tx_threads = [pgen.TxThread(i, 8.4e6) for i in range(len(env.tx_machines))]
    sfcs = env.sfcs.copy()

    cpu_placement = {}
    if is_rl:
        for helper in env.cpu_helper:
            helper.export_assignment(cpu_placement)
    else:
        cpu_placement = place_vnfs_on_cores(sfcs, env.vnf_machines)

    cpu_performance = simulate_achieved_rate(cpu_placement, 1e-3)
    update_rates(cpu_performance, sfcs)

    tx_assignment = assign_vnfs_to_tx_threads(sfcs, tx_threads, False)
    tx_performance = estimate_achieved_rate_tx_thread(tx_assignment, tx_threads)
    update_rates(tx_performance, sfcs)

    return sum([sfc.rate for sfc in sfcs])


def store_rl_tx(problem_id, tx_dict: Dict[pgen.Job, pgen.TxThread], tx_count):
    file = f'/opt/project/EvalData/RLData_2/RL_{problem_id}.json'
    file_p = open(file, 'w+')

    result = {}
    for i in range(tx_count):
        result[str(i)] = []

    for job, tx in tx_dict.items():
        result[str(tx.index)].append(job.vnf.instance_id)

    json.dump(result, file_p)
    file_p.close()


def least_loaded_first(env: ONVMVnfPlayerEnv, **kwargs) -> Tuple[List[float], Dict[object, object]]:
    # Add all machines that are available for scheduling jobs.
    loads = [0 for _ in range(len(env.all_machines) - env.tx_count)]
    assignment = {}
    for vnf in env.current_vnf.nf_list:
        idx = np.argmin(loads)
        loads[idx] += vnf.demand / 2.2e9
        assignment[vnf] = env.all_machines[idx + env.tx_count]
    return loads, assignment


def round_robin(env: ONVMVnfPlayerEnv, **kwargs) -> Tuple[List[float], Dict[object, object]]:
    loads = [0 for _ in range(len(env.all_machines) - env.tx_count)]
    assignment = {}
    for i, vnf in enumerate(env.current_vnf.nf_list):
        idx = i % len(loads)
        loads[idx] += vnf.demand / 2.2e9
        assignment[vnf] = env.all_machines[idx + env.tx_count]
    return loads, assignment


def first_fit(env: ONVMVnfPlayerEnv, load_level: float):
    loads = [0]
    assignment = {}
    for vnf in env.current_vnf.nf_list:
        assigned = False
        for i, load in enumerate(loads):
            if load + (vnf.demand / 2.2e9) < load_level:
                assigned = True
                loads[i] += vnf.demand / 2.2e9
                idx = i
                break
        if not assigned:
            # do first the assignment because of indexing starting at zero.
            idx = len(loads)
            if idx >= len(env.all_machines) - env.tx_count:
                # All machines are utilized, cannot add new one, take the least loaded.
                idx = np.argmin(loads)
                loads[idx] += vnf.demand / 2.2e9
            else:
                # Add a new machine.
                loads.append(vnf.demand / 2.2e9)
        assignment[vnf] = env.all_machines[idx + env.tx_count]
    return loads, assignment


def first_fit_adv2(env: ONVMVnfPlayerEnv) -> Tuple[List[float], Dict[Any, Any]]:
    assignment = {}
    for vnf in env.current_vnf.nf_list:
        loads = []
        idx = None
        for i, cpu_helper in enumerate(env.cpu_helper):
            loads.append(cpu_helper.load_level)
            if cpu_helper.would_go_over_limit(vnf):
                continue
            else:
                idx = i
                break
        if idx is None:
            idx = np.argmin(loads)
        env.current_vnf.job = vnf
        env.assign_vnf_to_cpu([idx])
        assignment[vnf] = env.all_machines[idx + env.tx_count]
    return [h.load_level for h in env.cpu_helper], assignment


def first_fit_adv(env: ONVMVnfPlayerEnv, load_level: float, model_path: str):
    model = load(model_path)
    features = np.zeros((16, 3))
    assignment = {}
    # self.vnf_to_core[self.current_vnf.job] = self.all_machines[action_dict[self.my_env_step] + self.tx_count]
    for vnf in env.current_vnf.nf_list:
        probs = []
        assigned = False
        idx = None
        for i in range(16):
            if features[i, 2] + vnf.demand >= 2.2e9 * load_level:
                probs.append(0.)
                continue
            # Get the probability that the VNF will *not* overload the core.
            probs.append(model.predict_proba(
                features[[i], :2] + np.array([[vnf._rate, vnf.vnf.compute_per_packet]], dtype=np.float32)
            )[0, 0])
            # Probability of not overloading core larger than 0.5?
            if probs[-1] > 0.5:
                print(f'assign to {i}')
                idx = i
                assigned = True
                features[i, 0] += vnf._rate
                features[i, 1] += vnf.vnf.compute_per_packet
                features[i, 2] += vnf.demand
                break
        if not assigned:
            i = np.argmin(probs)
            idx = i
            print(f'fallback: assign to {i}')
            features[i, 0] += vnf._rate
            features[i, 1] += vnf.vnf.compute_per_packet
            features[i, 2] += vnf.demand
        assignment[vnf] = env.all_machines[idx + env.tx_count]
        print(', '.join([f'{x:.2f}' for x in features[i]]))
        print(', '.join([f'{x:.2f}' for x in probs]))
        print(', '.join([f'{x/2.2e9:.2f}' for x in features[:, 2]]))
        print('---')
    return [x / 2.2e9 for x in features[:, 2]], assignment


def _restore_env(config):
    config['env_config']['evaluation_seed'] = 1
    config['env_config']['num_overload_actors'] = 1
    single_env = ONVMVnfPlayerEnv(config["env_config"])
    return single_env


def load_env_config(trial_dir) -> Dict[str, Any]:
    with open(os.path.join(trial_dir, 'params.json'), 'r') as fh:
        config = json.load(fh)
    return config


def restore_env(trial_dir):
    return _restore_env(load_env_config(trial_dir))


def make_twin_actor(trial_dir) -> bool:
    config = load_env_config(trial_dir)
    ret = config["env_config"]["overload_prediction"] and \
            config['env_config']['overload_model_class'] == 'RandomForestActor'
    return ret


def restore_agent(trial_dir: str, checkpoint_dir: str):
    with open(os.path.join(trial_dir, 'params.json'), 'r') as fh:
        config = json.load(fh)
    print(json.dumps(config, indent=1))
    default_config = dqn.DEFAULT_CONFIG.copy()
    default_config['env'] = config['env']
    default_config['learning_starts'] = config['learning_starts']
    default_config['train_batch_size'] = config['train_batch_size']
    default_config['exploration_config'] = config['exploration_config']
    default_config['env_config'] = config['env_config']
    default_config['framework'] = config["framework"]
    default_config['num_workers'] = 0
    default_config['num_envs_per_worker'] = 1
    default_config['explore'] = False
    default_config['lr'] = config["lr"]
    default_config["normalize_actions"] = config["normalize_actions"]
    # default_config['env_config']['overload_prediction'] = True
    # default_config['env_config']['overload_model_path'] = '/opt/project/rf.model'

    single_env = _restore_env(default_config)
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    # Multiagent config
    default_config["multiagent"] = {
        "policies": {
            'all-tx': (None, obs_space, act_space, {})
        },
        "policy_mapping_fn": lambda x: 'all-tx'
    }
    default_config['hiddens'] = config['hiddens']
    default_config['model'] = config['model']
    default_config['buffer_size'] = config['buffer_size']
    default_config['evaluation_interval'] = config['evaluation_interval']
    default_config['evaluation_num_workers'] = config['evaluation_num_workers']
    default_config['evaluation_num_episodes'] = config['evaluation_num_episodes']
    default_config['evaluation_config'] = config['evaluation_config']

    # agent = sac.SACTrainer(config=default_config, env=ONVMVnfPlayerEnv)
    agent = dqn.DQNTrainer(config=default_config, env=ONVMVnfPlayerEnv)
    logger.debug(f"Try to restore agent from {checkpoint_dir}")
    agent.restore(checkpoint_dir)
    return agent, single_env


def make_eval_problems():
    all_machines = [pgen.Machine(i, 2.2e9) for i in range(3, 24)]
    problems = []
    problems_small = []
    random = np.random.RandomState(seed=1)
    count = 1
    while len(problems) < 100:
        # Keep the first five machines for TX for sure, then add between
        # 4 and 16 machines to the generator.
        problem_generator = pgen.HeavyTailedGeneratorWithTx(
            machines=all_machines[:5 + random.randint(4, 17):],  # Use 17 --> max val 16 --> return all machines
            num_sfcs=(1, 6),
            max_num_vnfs_per_sfc=8,
            load_level=0.7,
            max_num_vnfs_in_system=32,
            system_rate=2.5e6,
            seed=count,
            num_tx=5
        )
        count += 1
        sfcs, assignment, tx_machines = problem_generator.__next__()
        problems.append(pgen.export_problem(sfcs, assignment, all_machines, 5))
        for sfc in sfcs:
            sfc.rate = int(sfc.weight * 1e6)
        problems_small.append(pgen.export_problem(sfcs, assignment, all_machines, 5))
    with open("golden-samples-2-dot-5-mpps.json", 'w') as fh:
        json.dump(problems, fh, indent=1)
    with open("golden-samples-1-mpps.json", 'w') as fh:
        json.dump(problems_small, fh, indent=1)


def load_eval_problems(path: str) -> List[ProblemData]:
    path = {
        '2-dot-5-mpps': 'golden-samples-2-dot-5-mpps.json',
        '1-mpps': "golden-samples-1-mpps.json"
    }[path]
    with open(path, 'r') as fh:
        problems = json.load(fh)
    sfcs = [ProblemData(pgen.sfcs_from_problem(p), p['num_tx_threads']) for p in problems]
    return sfcs


def make_bl_assignments(description: str, trial_id: str, path: str, sort_vnfs: bool, baseline: callable):
    config = load_env_config(trial_dir)
    config["env_config"]["overload_prediction"] = False
    config['env_config']['max_vnfs'] = 32
    config['env_config']['min_vnfs'] = 5
    config['env_config']['num_of_sfcs'] = 5
    single_env = _restore_env(config)
    single_env.sort_vnfs = sort_vnfs
    problems = load_eval_problems(path)
    assigned_problems = []
    for problem in problems:
        single_env.reset(problem)
        _, assignment = baseline(single_env)
        export = pgen.export_problem(single_env.sfcs, assignment,
                                     single_env.all_machines, single_env.num_tx)
        assigned_problems.append(export)
    with open(f'export-problems-{trial_id}-{path}.json', 'w') as fh:
        json.dump(
            {
                'executed': False,
                'checkpoint': checkpoint_dir,
                'result_dir': f'{trial_id}-{path}',
                'assignments': assigned_problems,
                'description': description,
                'machines_reversed': False
            },
            fh,
            indent=1
        )


def make_assignments(description: str, name: str, trial_dir: str, checkpoint_dir: str,
                     result_dir: str, rate_part: str):
    if make_twin_actor(trial_dir):
        print("Create actor, sleep for 1s.")
        config = load_env_config(trial_dir)
        actor = RandomForestActor \
            .options(name=config["env_config"]["overload_model_path"] + f'-0') \
            .remote(model_path=config["env_config"]["overload_model_path"])
        time.sleep(1)

    agent, single_env = restore_agent(checkpoint_dir=checkpoint_dir, trial_dir=trial_dir)
    problems = load_eval_problems(rate_part)
    assigned_problems = []
    for problem in problems:
        obs = single_env.reset(problem)
        done = {'__all__': False}
        while not done['__all__']:
            res = agent.compute_actions(obs, policy_id='all-tx', full_fetch=True)
            obs, rew, done, info = single_env.step(res[0])
        export = pgen.export_problem(single_env.sfcs, single_env.vnf_to_core,
                                     single_env.all_machines, single_env.num_tx)
        assigned_problems.append(export)
    with open(os.path.join(result_dir, f'export-problems-{name}-{rate_part}.json'), 'w') as fh:
        json.dump(
            {
                'executed': False,
                'checkpoint': checkpoint_dir,
                'trial_dir': trial_dir,
                'result_dir': f'{name}-{rate_part}',
                'assignments': assigned_problems,
                'description': description,
                'machines_reversed': False
            },
            fh,
            indent=1
        )


def main(trial_dir: str, checkpoint_dir: str, num_problems: int, plot_result: bool,
         fig_dir: str) -> Tuple[int, float]:
    actor = None
    if make_twin_actor(trial_dir):
        print("Create actor, sleep for 1s.")
        config = load_env_config(trial_dir)
        actor = RandomForestActor \
                .options(name=config["env_config"]["overload_model_path"] + f'-0') \
                .remote(model_path=config["env_config"]["overload_model_path"])
        time.sleep(1)
    agent, single_env = restore_agent(trial_dir, checkpoint_dir)
    single_env.evaluation = True
    # problems = vd.load_problems(vd.PATH_TO_DATAFILE)
    sensor = CpuAndVnfSensor()
    n_bins_rl = []
    n_overloaded = []
    total_reward = 0
    total_players = 0
    for i in range(num_problems):
        obs = single_env.reset()
        done = {'__all__': False}
        loads = {}
        q_vals = []
        all_probs = None
        n_vnfs_on_machines = []
        machine_loads = []
        rews = {}
        while not done['__all__']:
            sample = {
                f'cpu_player_{k}': {kk: vv.detach().numpy() for kk, vv in sensor.deserialize(torch.tensor(v)).items()}
                for k, v in obs.items()}
            res = agent.compute_actions(obs, policy_id='all-tx', full_fetch=True)
            cpu = list(res[0].values())[0]

            if cpu not in loads: loads[cpu] = 0
            loads[cpu] += single_env.current_vnf.nf_list[single_env.my_env_step].demand / 2.2e9
            q_vals.append(res[2]['q_values'].flatten())
            #  machine_loads.append(list(sample.values())[0]['kv_cpu'][0, :, 1] + 1)
            machine_loads.append(np.zeros(len(single_env.cpu_helper)))
            machine_loads[-1][cpu] += single_env.current_vnf.nf_list[single_env.my_env_step].demand / 2.2e9
            logits = res[2]['action_dist_inputs']
            probs = np.exp(logits) / np.expand_dims(np.sum(np.exp(logits), axis=1), axis=1)
            all_probs = probs if all_probs is None else np.concatenate((all_probs, probs), axis=0)
            n_vnfs_on_machines.append(np.zeros(len(single_env.cpu_helper)))
            n_vnfs_on_machines[-1][cpu] = 1

            obs, rew, done, info = single_env.step(res[0])
            rews.update(rew)
        n_overloaded.append(np.sum([1 if r <= -10. else 0 for r in rews.values()]))
        total_reward += np.sum(list(rews.values()))
        total_players += len(rews)

        if plot_result:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            # cax = ax1.imshow(np.row_stack(q_vals), cmap='Greens', vmin=-10, vmax=0)
            cax = ax1.imshow(all_probs, cmap='Greens')# , vmin=0, vmax=1)
            plt.colorbar(cax, ax=ax1, location='top', orientation='horizontal')
            cax = ax2.imshow(np.cumsum(np.row_stack(machine_loads), axis=0), cmap='Greens', vmin=0, vmax=1)
            plt.colorbar(cax, ax=ax2, location='top', orientation='horizontal')
            cax = ax3.imshow(np.row_stack(n_vnfs_on_machines), cmap='Greens', vmin=0, vmax=1)
            plt.colorbar(cax, ax=ax3, location='top', orientation='horizontal')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'job-assignment-problem-{i:d}.png'))
            plt.show()
            plt.close(fig)
        loads_l = [x for x in loads.values() if x > 0]
        n_bins_rl.append(len(loads_l))

    if actor is not None:
        ray.kill(actor)
    print(pd.Series(n_bins_rl).describe())
    print(pd.Series(n_overloaded).describe())
    print(f"rl overloaded in {np.sum(np.array(n_overloaded) > 0)} problems.")

    return np.sum(np.array(n_overloaded) > 0), total_reward / total_players


def get_best_pbt_path(exp_dir: str) -> Tuple[Tuple[str, str], Tuple[int, float]]:
    import evaluation.utils as evutils
    paths = evutils.get_pbt_max_checkpoints(exp_dir)
    perf = {}
    for trial_dir, chk_dir in paths:
        try:
            perf[(trial_dir, chk_dir)] = main(
                trial_dir,
                evutils.prepare_checkpoint_path(trial_dir, chk_dir),
                100,
                False,
                fig_dir=''
            )
        except Exception as e:
            logger.error(f"ERROR ON {os.path.join(trial_dir, chk_dir)}")
            logger.exception(e)
    infos = [(k, v) for k, v in perf.items()]
    infos.sort(key=lambda x: x[1][1])
    return infos[-1]


def baselines():
    for i in range(70, 101, 5):
        lv = i / 100.
        desc = f'Use FirstFitDecreasing with a load level of {i}%.'
        make_bl_assignments(desc, f'first-fit-decreasing-l{i}', '2-dot-5-mpps', True, lambda env: first_fit(env, lv))
        make_bl_assignments(desc, f'first-fit-decreasing-l{i}', '1-mpps', True, lambda env: first_fit(env, lv))

    desc = 'Use LeastLoadedFirst with a load level of 100%.'
    make_bl_assignments(desc, 'least-loaded-first-l100', '2-dot-5-mpps', False, least_loaded_first)
    make_bl_assignments(desc, 'least-loaded-first-l100', '1-mpps', False, least_loaded_first)

    desc = 'Use LeastLoadedFirstDecreasing with a load level of 100%.'
    make_bl_assignments(desc, 'least-loaded-first-decreasing-l100', '2-dot-5-mpps', True, least_loaded_first)
    make_bl_assignments(desc, 'least-loaded-first-decreasing-l100', '1-mpps', True, least_loaded_first)

    desc = 'Use RoundRobin with a load level of 100%.'
    make_bl_assignments(desc, 'round-robin-l100', '2-dot-5-mpps', False, round_robin)
    make_bl_assignments(desc, 'round-robin-l100', '1-mpps', False, round_robin)


def evaluate_model(exp_dir: str) -> None:
    (trial_dir, ckpt_dir), (n_overload, avg_utility) = get_best_pbt_path(exp_dir)
    with open(os.path.join(trial_dir, 'params.json'), 'r') as fh:
        config = json.load(fh)
    print(json.dumps(config['env_config'], indent=1))
    fig_base_dir = '/opt/project/Graphs'
    name = os.path.split(exp_dir)[1]
    fig_dir = os.path.join(fig_base_dir, name)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    parts = ckpt_dir.split('_')
    parts[1] = parts[1].lstrip('0')
    ckpt_dir = os.path.join(trial_dir, ckpt_dir, f'{parts[0]}-{parts[1]}')
    main(trial_dir, ckpt_dir, 10, True, fig_dir)
    with open(os.path.join(fig_dir, 'stats.txt'), 'w') as fh:
        fh.write(f'Trial dir: {trial_dir}\n')
        fh.write(f"Overloaded: {n_overload} %, average utility is: {avg_utility}\n")
    # desc = "RL policy for load balancing with shared reward, each player gets " \
    #        "the load of the most loaded machine as reward, or -10 if a machine is overloaded. " \
    #        "Training broke after 2500 iterations."
    desc = input("Enter experiment description: ")
    make_assignments(desc, name, trial_dir, ckpt_dir, fig_dir, '2-dot-5-mpps')
    make_assignments(desc, name, trial_dir, ckpt_dir, fig_dir, '1-mpps')
    plot_model_stats(trial_dir, fig_dir)


def plot_model_stats(trial_dir: str, fig_dir: str) -> None:
    step_results = pltr.get_json_result_data(trial_dir)
    fig, ax = pltr.plot_reward_min_max_mean(step_results)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'reward-stats.pdf'))
    plt.close(fig)

    fig, ax = pltr.plot_qvals_min_max_mean(step_results)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'q-stats.pdf'))
    plt.close(fig)

    fig, ax = pltr.plot_grad_norm(step_results)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'grad-norm.pdf'))
    plt.close(fig)

    fig, ax = pltr.plot_td_error(step_results)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'td-error.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    ModelCatalog.register_custom_model('CPUPlayerDQNModel', CPUPlayerDQNModel)
    ModelCatalog.register_custom_model('CPUPlayerWithVnfObjectsModel', CPUPlayerWithVnfObjectsModel)
    ModelCatalog.register_custom_model('CPUPlayerSACModel', CPUPlayerSACModel)
    ModelCatalog.register_custom_model('CPUPlayerPolicyModel', CPUPlayerPolicyModel)
    ray.init()

    desc = "Use FirstFitDecreasingAdvanced with a load level of 100% and RF for" \
           " for soft overload classification - fully trained model."
    # make_eval_problems()
    # Use RL to make assignments and evaluate them.
    # make_assignments(desc)
    # make_bl_assignments(desc, 'first-fit-decreasing-advanced-L100-soft', lambda env: first_fit_adv2(env))
    # baselines()

    # main(trial_dir, checkpoint_dir, 10, True, 1.)
    bdir = '/opt/project/data/nas/TuneResults'
    bdir = '/opt/project/Net'
    files = os.listdir(bdir)
    files.sort()
    lst = '\n\t'.join(f'{i:2d} {f}' for i, f in enumerate(files))
    e = files[int(input(f"Choose experiment to evaluate:\n\t{lst}\nEnter number: "))]
    exp_dir = os.path.join(bdir, e)
    print(f"Evaluate {exp_dir}\n---")
    evaluate_model(exp_dir)
    # get_best_pbt_path(exp_dir, load_level=1.)
