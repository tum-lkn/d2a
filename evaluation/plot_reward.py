import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import json
from typing import List, Dict, Any, Optional, Tuple
import re
from ray.tune import Analysis
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from joblib import load

import evaluation.plotutils as pltutils

plt.rcParams['image.cmap'] = 'Accent'

#PATH ='/home/developer/ray_results/SAC_2021-10-11_14-30-49/SAC_ONVMMultiAgent_d450b_00000_0_2021-10-11_14-30-49/'
#PATH2 = '/home/developer/ray_results/SAC_2021-10-11_16-35-01/SAC_ONVMMultiAgent_2e3e2_00000_0_2021-10-11_16-35-01/'
#PATH = '/opt/project/Net/TX-Assignment/SAC_ONVMMultiAgent_b5ec3_00000_0_2021-10-11_18-11-52/'
#PATH = '/opt/project/Net/TX-Assignment/SAC_ONVMMultiAgent_3d532_00000_0_2021-10-11_20-24-30/'
#PATH = '/opt/project/Net/TX-Assignment/SAC_ONVMMultiAgent_a0063_00000_0_2021-10-12_09-27-31/'
#PATH = '/opt/project/Net/TX-Assignment/SAC_ONVMMultiAgent_73aa2_00000_0_2021-10-12_11-06-30'
#PATH = '/opt/project/Net/TX-Assignment/SAC_ONVMMultiAgent_872da_00000_0_2021-10-12_15-24-44'
#PATH = '/opt/project/Net/TX-Assignment/SAC_ONVMMultiAgent_81c2d_00000_0_2021-10-12_15-53-13'
#PATH= '/opt/project/Net/TX-Assignment/SAC_ONVMMultiAgent_68446_00000_0_2021-10-12_19-41-34'
#PATH = '/opt/project/Net/TX-Assignment/SAC_ONVMMultiAgent_8f8bf_00000_0_2021-10-13_08-14-17'
#PATH = '/opt/project/Net/TX-Assignment/SAC_ONVMMultiAgent_57d09_00000_0_2021-10-13_13-27-42'

#PATH = '/opt/project/Net/TX-Assignment/SAC_ONVMMultiAgent_24a39_00000_0_2021-10-13_18-05-26'
#PATH = '/opt/project/Net/TX-Assignment/SAC_ONVMMultiAgent_677e7_00000_0_2021-10-14_07-57-40'
#PATH = '/opt/project/Net/TX-Assignment/SAC_ONVMMultiAgent_234e0_00000_0_lr=0.1_2021-10-14_17-06-57'
#PATH ='/opt/project/Net/TX-Assignment/SAC_ONVMMultiAgent_e6a39_00000_0_2021-10-14_18-38-18'

#PATH = '/opt/project/Net/TX-Assignment/SAC_ONVMMultiAgent_3024e_00000_0_2021-10-17_08-56-59'
PATH = '/opt/project/Net/CPU-Assignment/SAC_ONVMMultiAgent_9d727_00000_0_2021-10-19_16-21-29'
TRIAL_NAME = 'StateTrainable_2a6cc_00000_0_buffer_size=276197,lr=0.0002169_2021-12-02_18-03-34'
PATH = f'/opt/project/Net/VnfPlayer/{TRIAL_NAME}'


class EvalResult(object):
    def __init__(self) -> None:
        super().__init__()
        self.reward_mean = 0.0
        self.reward_max = 0.0
        self.reward_min = 0.0
        self.rewards: List[float] = []
        self.step = 0


def hexbin_plot(dataset: pd.DataFrame, factor=1., filename=None):
    min_rate, max_rate = dataset.expected_rate.min(), dataset.expected_rate.max()
    min_cost, max_cost = dataset.expected_cost.min(), dataset.expected_cost.max()
    min_rate, max_rate = dataset.measured_rate.min(), dataset.measured_rate.max()
    min_cost, max_cost = dataset.rough_measured_compute.min(), dataset.rough_measured_compute.max()
    ax = plt.subplot()
    x = np.linspace(min_rate, max_rate, 1000)
    y = np.clip(2.2e9 * factor / x, min_cost, max_cost)
    # ax.fill_between(x, y, color='orange')
    # ax.fill_between(x, y, np.max(y), color='blue')
    ax.hexbin(dataset.expected_rate.values, dataset.expected_cost.values, vmin=0, vmax=1, cmap='inferno', gridsize=1000)
    ax.plot(x, y, color='white', label='Max. capacity')
    # ax.set_xlim(min_rate, 3e6)
    # ax.set_ylim(min_cost, 1e4)
    ax.set_xlabel("Rate [pps]")
    ax.set_ylabel("Cost [Cycles/Packet]")
    ax.legend(frameon=True)
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    plt.close('all')


def _decision_boundary_analytical_model(n_particles: int, min_rate: float, max_rate: float,
                                        min_compute: float, max_compute: float):
    rate_space = np.linspace(min_rate, max_rate, n_particles)
    compute_space = np.linspace(min_compute, max_compute, n_particles)
    demand = np.repeat(rate_space, n_particles) * np.tile(compute_space, n_particles)
    demand = (demand > (2.2e9 / 1.2)).astype(np.float32)
    demand = np.reshape(demand, [n_particles, n_particles])
    return demand


def plot_decision_boundary_analytical_model(n_particles: int):
    demand = _decision_boundary_analytical_model(n_particles, 0, 5e6, 0, 1e5)
    ticks = np.linspace(0, n_particles - 1, 10).astype(np.int32)
    ax = plt.subplot()
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    space_rate = np.linspace(0, 5e6, n_particles)
    space_cost = np.linspace(0, 1e5, n_particles)
    ax.set_xticklabels(space_cost[ticks])
    ax.set_yticklabels(space_rate[ticks])
    ax.ticklabel_format(style='sci', scilimits=(0, 0))
    plt.imshow(demand, cmap='coolwarm')
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(path_to_model: str, n_particles: int, thresh=0.5):
    model = load(path_to_model)
    # min_rate, max_rate = (79415.0, 6878132.0)
    # min_cost, max_cost = (181.0, 26555.0)
    min_rate, max_rate = (79415.0, 3400000)
    min_cost, max_cost = (181.0, 8000.0)
    space_rate = np.linspace(min_rate, max_rate, n_particles)
    space_cost = np.linspace(min_cost, max_cost, n_particles)
    dat = np.column_stack((np.repeat(space_rate, n_particles), np.tile(space_cost, n_particles)))
    probs = np.reshape(model.predict_proba(dat)[:, 1], [n_particles, n_particles])
    # probs = np.reshape(model.predict_proba(np.expand_dims(dat[:, 0] * dat[:, 1], axis=1))[:, 1], [n_particles, n_particles])
    del dat
    ax = plt.subplot()
    ax.imshow(probs, cmap='coolwarm', vmin=0, vmax=1)
    ticks = np.linspace(0, n_particles - 1, 10).astype(np.int32)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'{a:.2f}' for a in space_cost[ticks] / 1e4])
    ax.set_yticklabels([f'{a:.2f}' for a in space_rate[ticks] / 1e6])
    plt.savefig('Graphs/decision-boundary-model.pdf')
    plt.show()
    plt.close('all')

    analytical = _decision_boundary_analytical_model(n_particles, min_rate, max_rate, min_cost, max_cost)
    ax = plt.subplot()
    ax.imshow(np.clip(probs + analytical, 0, 1), cmap='coolwarm', vmin=0, vmax=1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'{a:.2f}' for a in space_cost[ticks] / 1e4])
    ax.set_yticklabels([f'{a:.2f}' for a in space_rate[ticks] / 1e6])
    plt.savefig('Graphs/decision-boundary-model-and-analytical.pdf')
    plt.show()
    plt.close('all')

    ax = plt.subplot()
    ax.imshow(
        np.clip((probs > thresh).astype(np.float32) + analytical, 0, 1),
        cmap='coolwarm',
        vmin=0,
        vmax=1
    )
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'{a:.2f}' for a in space_cost[ticks] / 1e4])
    ax.set_yticklabels([f'{a:.2f}' for a in space_rate[ticks] / 1e6])
    plt.savefig('Graphs/decision-boundary-model-and-analytical-hard.pdf')
    plt.show()
    plt.close('all')


def flatten_dict(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    ret = {}
    for k, v in d.items():
        if type(v) == dict:
            ret.update(flatten_dict(v, f'{prefix}{k}/'))
        else:
            ret[f'{prefix}{k}'] = v
    return ret


def get_tf_file_name(trial_dir: str):
    print(trial_dir)
    tf_event_files = []
    for f in os.listdir(trial_dir):
        if f.startswith('events.out.tfevents'):
            tf_event_files.append(f)
    tf_event_files.sort(key=lambda x: int(x.split('.')[3]))
    return tf_event_files[-1]


def get_best_configs(exp_dir: str) -> List[Dict[str, Any]]:
    configs = []
    modes = ['min', 'mean', 'max']
    for d in os.listdir(exp_dir):
        if d is None: continue
        if not os.path.isdir(os.path.join(exp_dir, d)): continue
        trial_dir = os.path.join(exp_dir, d)
        event_acc = get_tensorboard_data(trial_dir)
        if event_acc is None:
            continue
        try:
            r_stats = {m: [s.value for s in event_acc.Scalars(f'ray/tune/evaluation/policy_reward_{m}/all-tx')] for m in modes}
            max_eval_reward = np.max(r_stats['mean'])
            if max_eval_reward > -2:
                with open(os.path.join(trial_dir, 'params.json'), 'r') as fh:
                    d = json.load(fh)
                d['trial_dir'] = trial_dir
                d.update(r_stats)
                configs.append(d)
            else:
                continue
        except:
            continue
    return configs


def get_tensorboard_data(trial_dir: str) -> EventAccumulator:
    # p = 'Net/TuneVnfPlayer/StateTrainable_08c29c6958f44aa7becbc090d67b0507_2_buffer_size=703448,epsilon_timesteps=547591,final_epsilon=0.052586,hiddens=[109,_2021-12-08_15-35-28'
    event_out_file = get_tf_file_name(trial_dir)
    if event_out_file is None:
        return None
    event_acc = EventAccumulator(os.path.join(trial_dir, event_out_file))
    event_acc.Reload()
    return event_acc
    # event_acc.Tags()  # Get dict of keys in the event file.


def plot_tensorboard_data(event_acc: EventAccumulator, trial_dir=None) -> None:
    if event_acc is None:
        return None
    if 'ray/tune/evaluation/policy_reward_min/all-tx' not in event_acc.Tags()['scalars']:
        print("evaluation scalars not in TF event file.")
        return None
    modes = ['min', 'mean', 'max']
    r_stats = {m: [s.value for s in event_acc.Scalars(f'ray/tune/evaluation/policy_reward_{m}/all-tx')] for m in modes}

    plt.plot([0, len(r_stats[modes[0]])], [-1, -1], linestyle='--', color='black')
    plt.plot([0, len(r_stats[modes[0]])], [-2, -2], linestyle='--', color='black')
    plt.plot([0, len(r_stats[modes[0]])], [-10, -10], linestyle='--', color='black')
    for m in modes:
        plt.plot(r_stats[m], label=m)
    plt.legend(frameon=False)
    plt.title(os.path.split(trial_dir)[1][:47])
    plt.show()
    plt.close('all')


def get_reward_data(step: Dict[str, Any]) -> Optional[EvalResult]:
    eval = step.get('episode_reward_mean', None)
    if eval:
        eval_res = EvalResult()
        eval_res.reward_mean = step['episode_reward_mean']
        eval_res.reward_max = step['episode_reward_max']
        eval_res.reward_min = step['episode_reward_min']
        eval_res.step = step.get('training_iteration', 0)
        return eval_res


def get_eval_data(trial_dir) -> List[EvalResult]:
    with open(os.path.join(trial_dir, 'result.json'), 'r') as fh:
        lines = fh.readlines()
    results = [get_reward_data(json.loads(l)) for l in lines]
    return results


def get_json_result_data(trial_dir) -> List[Dict[str, Any]]:
    with open(os.path.join(trial_dir, 'result.json'), 'r') as fh:
        lines = []
        for i, l in enumerate(fh.readlines()):
            try:
                line = json.loads(l)
                lines.append(line)
            except Exception as e:
                print(f"Error reading line {i}.")
                print(e)
        assert len(lines) > 0, f"No lines read from file {os.path.join(trial_dir, 'result.json')}"
    return lines


def parse_td_error_from_json(result_data: List[Dict[str, Any]]):
    td_errors = [
        json.loads(
            re.sub(
                "[\s\n]+",
                ",",
                re.sub(
                    "\[\s*",
                    "[",
                    re.sub(
                        "\s*\]",
                        "]",
                        d['info']['learner']['all-tx']['td_error']
                    )
                )
            )
        )
        for d in result_data]
    return td_errors


def get_name() -> str:
    print(PATH)
    m = re.search(r'StateTrainable_([a-zA-Z0-9]*)_', PATH)
    if m.group(1) != None:
        return f'/opt/project/Graphs/eval-reward-small_{m.group(1)}.pdf'


def plot_result(results: List[EvalResult]):
    file = get_name()
    max = [e.reward_max for e in results]
    min = [e.reward_min for e in results]
    mean = [e.reward_mean for e in results]
    steps = [e.step for e in results]
    trend = []

    # trend_p = np.polyfit(steps, mean, 1)

    # for step in steps:
    #     x = trend_p[0] * step + trend_p[1]
    #     trend.append(x)

    offset = 0

    print(np.argmax(mean))

    plt.plot(max[offset:], label='Max')
    plt.plot(min[offset:], label='Min')
    plt.plot(mean[offset:], label='Mean')
    # plt.plot(trend[offset:], 'r')
    plt.title('Evaluation Reward')
    plt.xlabel('Evaluation Iteration (10 Trainings Iterations apart)')
    plt.ylabel('Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file)
    plt.show()


def plot_losses(trial_dir: str):
    df = pd.read_csv(os.path.join(trial_dir, 'progress.csv'))
    # plt.plot(df['info/learner/all-tx/learner_stats/actor_loss'].values)
    # plt.ylabel("Actor Loss")
    # plt.show()
    # plt.close('all')

    # plt.plot(df['info/learner/all-tx/learner_stats/critic_loss'].values)
    # plt.ylabel("Critic Loss")
    # plt.show()
    # plt.close('all')
    tderror = json.loads(df.iloc[1]['info/learner/all-tx/td_error']\
                         .replace('\n', ' ').replace(' ', ',').replace(',,', ',')\
                         .replace(',,', ',').replace(',,', ',').replace('[,', '[')\
                         .replace(',]', ']'))
    tderror2 = json.loads(df.iloc[-1]['info/learner/all-tx/td_error'] \
                         .replace('\n', ' ').replace(' ', ',').replace(',,', ',') \
                         .replace(',,', ',').replace(',,', ',').replace('[,', '[') \
                         .replace(',]', ']'))

    plt.hist(tderror, label="First", alpha=0.7, bins=100)
    plt.hist(tderror2, label="Last", alpha=0.7, bins=100)
    plt.xlabel("TD Error")
    plt.ylabel("Count")
    plt.legend()
    plt.show()
    plt.close('all')

    plt.hist(json.loads(df.iloc[1, :]['hist_stats/policy_all-tx_reward']), label="First", alpha=0.7, bins=100)
    plt.hist(json.loads(df.iloc[-1, :]['hist_stats/policy_all-tx_reward']), label="Last", alpha=0.7, bins=100)
    plt.xlabel("Utility Values")
    plt.ylabel("Count")
    plt.legend()
    plt.show()
    plt.close('all')

    plt.plot([0, df['info/learner/all-tx/mean_td_error'].size], [0, 0], linestyle='--', color='black')
    plt.plot(df['info/learner/all-tx/mean_td_error'].values)
    plt.ylabel("TD Error")
    plt.show()
    plt.close('all')
    
    plt.plot(df['info/learner/all-tx/learner_stats/grad_gnorm'].values)
    plt.ylabel("Grad Norm")
    plt.show()
    plt.close('all')

    plt.plot(df['info/learner/all-tx/learner_stats/max_q'], label='max')
    plt.plot(df['info/learner/all-tx/learner_stats/min_q'], label='min')
    plt.plot(df['info/learner/all-tx/learner_stats/mean_q'], label='avg')
    plt.legend()
    plt.ylabel("Q=Stats")
    plt.show()
    plt.close('all')


def plot_reward_min_max_mean(step_results: List[Dict[str, Any]]) -> Tuple[plt.Figure, plt.Axes]:
    def smooth(data: List[float]) -> np.array:
        return pd.Series(data).rolling(10).mean().values
    fig, ax = pltutils.get_fig(1)
    mins = smooth([s['episode_reward_min'] for s in step_results])
    maxs = smooth([s['episode_reward_max'] for s in step_results])
    means = smooth([s['episode_reward_mean'] for s in step_results])
    line_props = {
        'markevery': 500,
        'markeredgecolor': 'black',
        'markerfacecolor': 'white'
    }
    ax.plot(mins, label='Min', c=pltutils.COLORS[0], marker=pltutils.MARKER[0], **line_props)
    ax.plot(maxs, label='Max', c=pltutils.COLORS[1], marker=pltutils.MARKER[1], **line_props)
    ax.plot(means, label='Avg', c=pltutils.COLORS[2], marker=pltutils.MARKER[2], **line_props)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Utility")
    ax.legend(frameon=False)
    return fig, ax


def plot_qvals_min_max_mean(step_results: List[Dict[str, Any]]) -> Tuple[plt.Figure, plt.Axes]:
    def smooth(data: List[float]) -> np.array:
        return pd.Series(data).rolling(10).mean().values
    fig, ax = pltutils.get_fig(1)
    mins = smooth([s['info']['learner']['all-tx']['learner_stats']['min_q'] for s in step_results])
    maxs = smooth([s['info']['learner']['all-tx']['learner_stats']['max_q'] for s in step_results])
    means = smooth([s['info']['learner']['all-tx']['learner_stats']['mean_q'] for s in step_results])
    line_props = {
        'markevery': 500,
        'markeredgecolor': 'black',
        'markerfacecolor': 'white'
    }
    ax.plot(mins, label='Min', c=pltutils.COLORS[0], marker=pltutils.MARKER[0], **line_props)
    ax.plot(maxs, label='Max', c=pltutils.COLORS[1], marker=pltutils.MARKER[1], **line_props)
    ax.plot(means, label='Avg', c=pltutils.COLORS[2], marker=pltutils.MARKER[2], **line_props)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Q-Value")
    ax.legend(frameon=False)
    return fig, ax


def plot_td_error(step_results: List[Dict[str, Any]]) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = pltutils.get_fig(1)
    tds = [s['info']['learner']['all-tx']['mean_td_error'] for s in step_results]
    ax.plot(tds, c=pltutils.COLORS[0], marker=pltutils.MARKER[0], markevery=500,
            markeredgecolor='black', markerfacecolor='white')
    ax.plot([0, len(tds)], [0, 0], c='black', linestyle='--')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("TD Error")
    return fig, ax


def plot_grad_norm(step_results: List[Dict[str, Any]]) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = pltutils.get_fig(1)
    tds = [s['info']['learner']['all-tx']['learner_stats']['grad_gnorm'] for s in step_results]
    ax.plot(tds, c=pltutils.COLORS[0], marker=pltutils.MARKER[0], markevery=500,
            markeredgecolor='black', markerfacecolor='white')
    ax.plot([0, len(tds)], [0, 0], c='black', linestyle='--')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Gradient Norm")
    return fig, ax


if __name__ == '__main__':
    ana = Analysis('/opt/project/Net/TuneVnfPlayer')
    trial_dir = ana.get_best_logdir(metric='episode_reward_mean', mode='max')
    # results = get_eval_data(trial_dir)
    # plot_result(results)
    get_tensorboard_data(trial_dir)
    plot_losses(trial_dir)
