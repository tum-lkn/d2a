from __future__ import annotations
import json
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import networkx as nx

import dataprep.utils as dutils
import dataprep.graph as grutils
import evaluation.plotutils as plutils
import environment.tp_sim as tp_sim


logger = logging.getLogger('plot_results')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


LABELS = {
    '43a78_00006': 'DQN-???',
    '9f3db_00004': 'DQN-80',
    'StateTrainable_bada73a7acc249788023daee71851383': 'DQN-BP-100',
    'ae661_00000': 'DQN-100-RF',
    'ae661_00000-final': 'DQN-100-RF-final',
    'first-fit-decreasing-advanced': 'FFD-100-RF2',
    'first-fit-decreasing-advanced-L100-soft': 'FFD-100-RF',
    'first-fit-decreasing-l100': 'FFD-100',
    'first-fit-decreasing-l70': 'FFD-70',
    'first-fit-decreasing-l80': 'FFD-80',
    'first-fit-decreasing-l90': 'FFD-90',
    'least-loaded-first-decreasing-l100': 'LLFD-100',
    'least-loaded-first-l100': 'LLF-100',
    'round-robin-l100': 'RR-100',
    
    'first-fit-decreasing-l100-2-dot-5-mpps': 'FFD-100-2.5',
    'first-fit-decreasing-l100-1-mpps': 'FFD-100-1',
    'first-fit-decreasing-l95-2-dot-5-mpps': 'FFD-95-2.5',
    'first-fit-decreasing-l95-1-mpps': 'FFD-95-1',
    'first-fit-decreasing-l90-2-dot-5-mpps': 'FFD-90-2.5',
    'first-fit-decreasing-l90-1-mpps': 'FFD-90-1',
    'first-fit-decreasing-l85-2-dot-5-mpps': 'FFD-85-2.5',
    'first-fit-decreasing-l85-1-mpps': 'FFD-85-1',
    'first-fit-decreasing-l80-2-dot-5-mpps': 'FFD-80-2.5',
    'first-fit-decreasing-l80-1-mpps': 'FFD-80-1',
    'first-fit-decreasing-l75-2-dot-5-mpps': 'FFD-75-2.5',
    'first-fit-decreasing-l75-1-mpps': 'FFD-75-1',
    'first-fit-decreasing-l70-2-dot-5-mpps': 'FFD-70-2.5',
    'first-fit-decreasing-l70-1-mpps': 'FFD-70-1',
    'generated-2-dot-5-mpps': 'GEN-70-2.5',
    'generated-1-mpps': 'GEN-70-1',
    'least-loaded-first-decreasing-l100-2-dot-5-mpps': 'LLFD-100-2.5',
    'least-loaded-first-decreasing-l100-1-mpps': 'LLFD-100-1',
    'least-loaded-first-l100-2-dot-5-mpps': 'LLF-100-2.5',
    'least-loaded-first-l100-1-mpps': 'LLF-100-1',
    'round-robin-l100-2-dot-5-mpps': 'RR-100-2.5',
    'round-robin-l100-1-mpps': 'RR-100-1',
    'TuneVnfPlayerCpuOnly100BinPackingOneOverNGoldenSamples-1-mpps': 'RL-1N-1',
    'TuneVnfPlayerCpuOnly100BinPackingOneOverNGoldenSamples-2-dot-5-mpps': 'RL-1N-2.5',
    'TuneVnfPlayerCpuOnly80BinPackingOneOverNLogRegGoldenSamples-1-mpps': 'RL-1N-DT-1',
    'TuneVnfPlayerCpuOnly80BinPackingOneOverNLogRegGoldenSamples-2-dot-5-mpps': 'RL-1N-DT-2.5',
    "TuneVnfPlayerCpuOnlyCpuRatios80BinPackingOneOverNLogRegGoldenSamples-1-mpps": 'RL-1N-DT2-1',
    "TuneVnfPlayerCpuOnlyCpuRatios80BinPackingOneOverNLogRegGoldenSamples-2-dot-5-mpps": 'RL-1N-DT2-2.5',
    'TuneVnfPlayerCpuOnly100LoadBalancingOneOverNGoldenSamples-1-mpps': 'RL-LB-1N-1-fail',
    'TuneVnfPlayerCpuOnly100LoadBalancingOneOverNGoldenSamples-2-dot-5-mpps': 'RL-LB-1N-2.5-fail',
    'TuneVnfPlayerCpuOnly100LoadBalancingSharedOneOverNGoldenSamples2-1-mpps': 'RL-LB-1N-1',
    'TuneVnfPlayerCpuOnly100LoadBalancingSharedOneOverNGoldenSamples2-2-dot-5-mpps': 'RL-LB-1N-2.5',
    'TuneVnfPlayerCpuOnly80LoadBalancingSharedOneOverNLogRegGoldenSamples-1-mpps': 'RL-LB-1N-DT-1',
    'TuneVnfPlayerCpuOnly80LoadBalancingSharedOneOverNLogRegGoldenSamples-2-dot-5-mpps': 'RL-LB-1N-DT-2.5'

}


def check_throughput_dqn():
    stats = dutils.slice_experiment_period(
        dutils.load_vnf_stats("data/nas/dqn_assignment_result_test/StateTrainable_bada73a7acc249788023daee71851383/dqn_eval-p0077-i00"))
    # Has a trhoughput of only 2.2 Mpps
    stats.set_index("INSTANCEID", inplace=True)
    # VNFs 20 has a rate of 0.32 Mpps, VNF7 of 0.776 Mpps, both roughtly the same amount of cost
    tmp_20 = stats.loc[20, :]
    tmp_7 = stats.loc[7, :]

    t1 = tmp_20.iloc[5000, 0]
    td = pd.Timedelta(100, unit='ms')

    ax = plt.subplot()
    tmp_20.set_index("TS").loc[t1:t1 + td, :].diff().TX.dropna().plot(label='NF20')
    tmp_7.set_index("TS").loc[t1:t1 + td, :].diff().TX.dropna().plot(label='NF7')
    ax.set_ylabel("Throughput pp/ms")
    plt.legend(frameon=False)
    plt.show()

    combined = pd.concat([
        tmp_20.loc[:, ['TX', 'COST', 'RX', 'RX_Q', 'RX_DROP', 'BURST']].diff().dropna().reset_index(drop=True),
        tmp_7.loc[:, ['TX', 'COST', 'RX', 'RX_Q', 'RX_DROP', 'BURST']].diff().dropna().reset_index(drop=True)
    ], axis=1)


def check_throughput_llfd():
    stats = dutils.slice_experiment_period(
        dutils.load_vnf_stats("data/nas/dqn_assignment_result_test/least-loaded-first-decreasing-l100/dqn_eval-p0077-i00"))
    # Has a trhoughput of only 2.2 Mpps
    stats.set_index("INSTANCEID", inplace=True)
    # VNFs 20 has a rate of 0.32 Mpps, VNF7 of 0.776 Mpps, both roughtly the same amount of cost
    tmp_20 = stats.loc[20, :]
    tmp_7 = stats.loc[7, :]

    t1 = tmp_20.iloc[5000, 0]
    td = pd.Timedelta(100, unit='ms')

    ax = plt.subplot()
    tmp_20.set_index("TS").loc[t1:t1 + td, :].diff().TX.dropna().plot(label='NF20')
    tmp_7.set_index("TS").loc[t1:t1 + td, :].diff().TX.dropna().plot(label='NF7')
    ax.set_ylabel("Throughput pp/ms")
    plt.legend(frameon=False)
    plt.show()

    combined = pd.concat([
        tmp_20.loc[:, ['TX', 'COST', 'RX', 'RX_Q', 'RX_DROP', 'BURST']].diff().dropna().reset_index(drop=True),
        tmp_7.loc[:, ['TX', 'COST', 'RX', 'RX_Q', 'RX_DROP', 'BURST']].diff().dropna().reset_index(drop=True)
    ], axis=1)


def get_overload_cpu(graphs: List[nx.Graph], attr: str) -> List[Dict[str, Any]]:
    overload = []
    for graph in graphs:
        for n, d in graph.nodes(data=True):
            if d['node_type'] == 'cpu':
                if attr not in d: continue
                if not d[attr]: continue
                overload.append(d)
    return overload


def load_graphs(exp_dir: str) -> List[nx.Graph]:
    graphs = []
    for f in os.listdir(exp_dir):
        try:
            graphs.append(grutils.make_graph(os.path.join(exp_dir, f)))
            graphs[-1].graph['exp_dir'] = os.path.join(exp_dir, f)
        except Exception as e:
            logger.error(f'Error during conversion of trial {os.path.join(exp_dir, f)}')
            logger.exception(e)
    return graphs


def load_algo_results(path: str) -> List[pd.DataFrame]:
    dfs = []
    for f in os.listdir(path):
        try:
            dfs.append(dutils.load_moongen_thoughput(os.path.join(path, f)))
        except Exception as e:
            logger.exception(e)
    return dfs


def load_algo_latency(path: str) -> List[pd.DataFrame]:
    dfs = []
    for f in os.listdir(path):
        try:
            dfs.append(dutils.load_moongen_latency(os.path.join(path, f)))
        except Exception as e:
            logger.exception(e)
    return dfs


def eval_test_runs(cfs_mode: str) -> Dict[str, List[float]]:
    """
    Get the thoughput.


    :param cfs_mode:
    :return:
    """
    # base_dir = '/opt/project/data/nas/dqn_assignment_result_test'
    # base_dir = '/opt/project/data/nas/dqn_assignment_result_test'
    if cfs_mode == 'cfs':
        base_dir = '/opt/project/data/nas/dqn_assignments_result'
    elif cfs_mode == 'rc':
        base_dir = '/opt/project/data/nas/dqn_assignments_result_rc'
    else:
        raise KeyError()
    results = {}
    for f in os.listdir(base_dir):
        if f not in LABELS:
            logger.warning(f'Label {f} not in LABELS.')
        else:
            dfs = load_algo_results(os.path.join(base_dir, f))
            # Experiment lasted 10 seconds, thus divide maximum number of send packets by 10
            results[LABELS[f]] = [float(df.set_index("iid").loc['all', 'TotalPackets'].max()) / 10. for df in dfs]
    return results


def load_problem_export(exp_dir: str) -> Dict[str, Any]:
    with open(os.path.join(exp_dir, 'problem.json'), 'r') as fh:
        problem = json.load(fh)
    return problem


def eval_aboslute_packet_cost(cfs_mode: str) -> Dict[str, List[Dict[str, int]]]:
    """
    Load vnf stats and get the total number of packets that each VNF processed.
    Sum them up and return it along with the number of CPU cores.
    """
    if cfs_mode == 'cfs':
        base_dir = '/opt/project/data/nas/dqn_assignments_result'
    elif cfs_mode == 'rc':
        base_dir = '/opt/project/data/nas/dqn_assignments_result_rc'
    else:
        raise KeyError()
    results = {}
    for f in os.listdir(base_dir):
        if f not in LABELS:
            logger.warning(f'Label {f} not in LABELS.')
        else:
            exp_dir = os.path.join(base_dir, f)
            results[LABELS[f]] = []
            for g in os.listdir(exp_dir):
                try:
                    stats = dutils.load_vnf_stats(os.path.join(exp_dir, g))
                    problem = load_problem_export(os.path.join(exp_dir, g))
                    tx = int(np.sum([grp.RX.max() for _, grp in stats.groupby("INSTANCEID")]))
                    cpus = int(np.unique([x['machine'] for x in problem['placement']]).size)
                    results[LABELS[f]].append({"packets": tx, "cpus": cpus})
                except Exception as e:
                    logger.error(f"Unexpected error processing {os.path.join(exp_dir, g)}")
                    logger.exception(e)
    return results


def latency_test_runs(cfs_mode: str):
    # Returns in miliseconds
    if cfs_mode == 'cfs':
        base_dir = '/opt/project/data/nas/dqn_assignments_result'
    elif cfs_mode == 'rc':
        base_dir = '/opt/project/data/nas/dqn_assignments_result_rc'
    else:
        raise KeyError()
    results = {}
    for f in os.listdir(base_dir):
        dfs = load_algo_latency(os.path.join(base_dir, f))
        avgs = []
        for df in dfs:
            for fid, values in df.groupby('fid'):
                # Calculate average value from histogram and convert to seconds.
                avg = np.sum(values.latency.values.astype(np.float32) * values.loc[:, 'count'].values.astype(np.float32)) / values.loc[:, 'count'].sum() * 1e-6
                avgs.append(avg)
        if f not in LABELS:
            logger.warning(f'Label {f} not in LABELS.')
        else:
            results[LABELS[f]] = np.array(avgs)
    return results


def load_graphs_for_test_runs() -> Dict[str, List[nx.Graph]]:
    base_dir = '/opt/project/data/nas/dqn_assignment_result_test'
    base_dir = '/opt/project/data/nas/dqn_assignments_result'
    results = {}
    for f in os.listdir(base_dir):
        graphs = load_graphs(os.path.join(base_dir, f))
        results[LABELS[f]] = graphs
    return results


def plot_num_used_cores_paper(all_graphs: Dict[str, List[nx.Graph]], throughput: str) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the fraction of used cores.

    Args:
         stats: Dictionary with statistics for all experiments.
         throughput: Throughput Level in {"1", "2.5"}.
    """
    max_val = 16.
    stats = count_num_cores(all_graphs)
    lbls_stats = [f'{s}-{throughput}' for s in ['RR-100', 'LLF-100', 'RL-LB-1N', 'RL-LB-1N-DT', 'FFD-100',
                                                'RL-1N', 'RL-1N-DT2']]
    lbls_fig = [n for n in plutils.NAMES]
    fig, ax = plutils.get_fig(1)
    parts = ax.violinplot(
        positions=np.arange(len(lbls_stats)),
        dataset=[stats[l] / max_val * 100 for l in lbls_stats],
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(plutils.NAME2COL[lbls_fig[i]])
        body.set_edgecolor(plutils.NAME2COL[lbls_fig[i]])
    for i, lbl in enumerate(lbls_fig):
        print(np.max(stats[lbls_stats[i]]))
        ax.scatter(np.repeat(i, len(stats[lbls_stats[i]])), stats[lbls_stats[i]] / max_val * 100,
                   alpha=0.5, c=plutils.NAME2COL[lbl], marker='_')
        ax.scatter([i], [np.mean(stats[lbls_stats[i]] / max_val * 100)], marker=plutils.NAME2MARK[lbl],
                   edgecolors='black', c='white')
        ax.scatter([i], [np.median(stats[lbls_stats[i]] / max_val * 100)], marker='_', c='black')
    ax.set_xticks(np.arange(len(lbls_stats)))
    ax.set_xticklabels(lbls_fig)
    ax.set_ylabel("Used CPU cores [%]")
    return fig, ax


def plot_throughput_paper(stats: Dict[str, np.array], throughput: str, vert: bool, ncols=1) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the throughput.

    Args:
         stats: Dictionary with statistics for all experiments.
         throughput: Throughput Level in {"1", "2.5"}.
    """
    max_val = {
        '1': 937777.2,
        '2.5': 2345644.6
    }[throughput]
    lbls_stats = [f'{s}-{throughput}' for s in ['RR-100', 'LLF-100', 'RL-LB-1N', 'RL-LB-1N-DT', 'FFD-100',
                                                'RL-1N', 'RL-1N-DT2']]
    lbls_fig = [n for n in plutils.NAMES]
    fig, ax = plutils.get_fig(ncols)
    parts = ax.violinplot(
        positions=np.arange(len(lbls_stats)),
        dataset=[stats[l] / max_val * 100 for l in lbls_stats],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        vert=vert
    )
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(plutils.NAME2COL[lbls_fig[1]])
        body.set_edgecolor(plutils.NAME2COL[lbls_fig[1]])
    for i, lbl in enumerate(lbls_fig):
        print(np.max(stats[lbls_stats[i]]))
        if vert:
            ax.scatter(np.repeat(i, len(stats[lbls_stats[i]])), stats[lbls_stats[i]] / max_val * 100,
                       alpha=0.5, c=plutils.NAME2COL[lbl], marker='_')
            ax.scatter([i], [np.mean(stats[lbls_stats[i]] / max_val * 100)], marker=plutils.NAME2MARK[lbl],
                       edgecolors='black', c='white')
            ax.scatter([i], [np.median(stats[lbls_stats[i]] / max_val * 100)], marker='_', c='black')
        else:
            ax.scatter(stats[lbls_stats[i]] / max_val * 100, np.repeat(i, len(stats[lbls_stats[i]])),
                       alpha=0.5, c=plutils.NAME2COL[lbls_fig[1]], s=1, marker='o')
            # ax.scatter(stats[lbls_stats[i]] / max_val * 100, np.repeat(i, len(stats[lbls_stats[i]])),
            #            alpha=0.5, c=plutils.NAME2COL[lbl], s=1, marker='o')
            ax.scatter([np.mean(stats[lbls_stats[i]] / max_val * 100)], [i], marker=plutils.NAME2MARK[lbl],
                       edgecolors='black', c='white')
            ax.scatter([np.median(stats[lbls_stats[i]] / max_val * 100)], [i], marker='|', c='black')
    if vert:
        ax.set_xticks(np.arange(len(lbls_stats)))
        ax.set_xticklabels(lbls_fig)
        ax.set_ylabel("Throughput [\%]")
    else:
        ax.set_yticks(np.arange(len(lbls_stats)))
        ax.set_yticklabels(lbls_fig)
        ax.set_xlabel("Throughput [\%]")
    return fig, ax


def plot_total_packet_cost_paper(stats: Dict[str, np.array], throughput: str, vert: bool, ncols=1) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the throughput.

    Args:
         stats: Dictionary with statistics for all experiments.
         throughput: Throughput Level in {"1", "2.5"}.
    """
    lbls_stats = [f'{s}-{throughput}' for s in ['RR-100', 'LLF-100', 'RL-LB-1N', 'RL-LB-1N-DT', 'FFD-100',
                                                'RL-1N', 'RL-1N-DT2']]
    lbls_fig = [n for n in plutils.NAMES]
    fig, ax = plutils.get_fig(ncols)
    parts = ax.violinplot(
        positions=np.arange(len(lbls_stats)),
        dataset=[stats[l] for l in lbls_stats],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        vert=vert
    )
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(plutils.NAME2COL[lbls_fig[1]])
        body.set_edgecolor(plutils.NAME2COL[lbls_fig[1]])
    for i, lbl in enumerate(lbls_fig):
        print(np.max(stats[lbls_stats[i]]))
        if vert:
            ax.scatter(np.repeat(i, len(stats[lbls_stats[i]])), stats[lbls_stats[i]],
                       alpha=0.5, c=plutils.NAME2COL[lbls_fig[1]], marker='_')
            ax.scatter([i], [np.mean(stats[lbls_stats[i]])], marker=plutils.NAME2MARK[lbl],
                       edgecolors='black', c='white')
            ax.scatter([i], [np.median(stats[lbls_stats[i]])], marker='_', c='black')
        else:
            ax.scatter(stats[lbls_stats[i]],np.repeat(i, len(stats[lbls_stats[i]])),
                       alpha=0.5, c=plutils.NAME2COL[lbls_fig[1]], marker='o', s=1)
            ax.scatter([np.mean(stats[lbls_stats[i]])], [i], marker=plutils.NAME2MARK[lbl],
                       edgecolors='black', c='white')
            ax.scatter([np.median(stats[lbls_stats[i]])], [i], marker='|', c='black')
    if vert:
        ax.set_xticks(np.arange(len(lbls_stats)))
        ax.set_xticklabels(lbls_fig)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.set_ylabel("Total Packet Cost [Cyles per Packet]")
    else:
        ax.set_yticks(np.arange(len(lbls_stats)))
        ax.set_yticklabels(lbls_fig)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.set_xlabel("TPC [Cpp]")
    return fig, ax


def plot_latency_paper(stats: Dict[str, np.array], throughput: str, vert: bool, ncols=1) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the throughput.

    Args:
         stats: Dictionary with statistics for all experiments.
         throughput: Throughput Level in {"1", "2.5"}.
    """
    lbls_stats = [f'{s}-{throughput}' for s in ['RR-100', 'LLF-100', 'RL-LB-1N', 'RL-LB-1N-DT', 'FFD-100',
                                                'RL-1N', 'RL-1N-DT2']]
    lbls_fig = [n for n in plutils.NAMES]
    fig, ax = plutils.get_fig(ncols)
    parts = ax.violinplot(
        positions=np.arange(len(lbls_stats)),
        dataset=[stats[l] for l in lbls_stats],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        vert=vert
    )
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(plutils.NAME2COL[lbls_fig[1]])
        body.set_edgecolor(plutils.NAME2COL[lbls_fig[1]])
    for i, lbl in enumerate(lbls_fig):
        print(np.mean(stats[lbls_stats[i]]))
        if vert:
            ax.scatter(np.repeat(i, len(stats[lbls_stats[i]])), stats[lbls_stats[i]],
                       alpha=0.5, c=plutils.NAME2COL[lbls_fig[1]], marker='_')
            # ax.scatter(np.repeat(i, len(stats[lbls_stats[i]])), stats[lbls_stats[i]],
            #            alpha=0.5, c=plutils.NAME2COL[lbl], marker='_')
            ax.scatter([i], [np.mean(stats[lbls_stats[i]])], marker=plutils.NAME2MARK[lbl],
                       edgecolors='black', c='white')
            ax.scatter([i], [np.median(stats[lbls_stats[i]])], marker='_', c='black')
        else:
            ax.scatter(stats[lbls_stats[i]],np.repeat(i, len(stats[lbls_stats[i]])),
                       alpha=0.5, c=plutils.NAME2COL[lbls_fig[1]], marker='o', s=1)
            # ax.scatter(np.repeat(i, len(stats[lbls_stats[i]])), stats[lbls_stats[i]],
            #            alpha=0.5, c=plutils.NAME2COL[lbl], marker='_')
            ax.scatter([np.mean(stats[lbls_stats[i]])], [i], marker=plutils.NAME2MARK[lbl],
                       edgecolors='black', c='white')
            ax.scatter([np.median(stats[lbls_stats[i]])], [i], marker='|', c='black')
    if vert:
        ax.set_xticks(np.arange(len(lbls_stats)))
        ax.set_xticklabels(lbls_fig)
        ax.set_ylabel("Latency [ms]")
    else:
        ax.set_yticks(np.arange(len(lbls_stats)))
        ax.set_yticklabels(lbls_fig)
        ax.set_xlabel("Latency [ms]")
    return fig, ax


def plot_throughput(stats: Dict[str, List[float]]):
    ax = plt.subplot()
    fig = plt.gcf()
    fig.set_figheight(8)
    labels, data = zip(*[(k, v) for k, v in stats.items()])
    ax.violinplot(
        positions=np.arange(len(data)),
        dataset=data,
        vert=False
    )
    ax.scatter([np.median(v) for v in data], np.arange(len(data)), marker='|')
    ax.scatter([np.mean(v) for v in data], np.arange(len(data)), marker='s', edgecolor='black', facecolor='white')
    ax.set_yticks(np.arange(len(stats)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Total Throughput [Mpps]")
    plt.tight_layout()
    plt.show()


def plot_divider_throughput(graphs: Dict[str, List[nx.Graph]]) -> None:
    vals = {}
    for k, grs in graphs.items():
        vals[k] = []
        for graph in grs:
            for n, d in graph.nodes(data=True):
                if d['name'] == 'divider':
                    vals[k].append(d['throughput'])
                else:
                    continue
    labels, values = zip(*[(k, v) for k, v in vals.items()])
    ax = plt.subplot()
    ax.violinplot(
        positions=np.arange(len(values)),
        dataset=values,
        vert=False
    )
    ax.scatter([np.median(v) for v in values], np.arange(len(values)), marker='|')
    ax.scatter([np.mean(v) for v in values], np.arange(len(values)), marker='s', edgecolor='black', facecolor='white')
    ax.set_yticks(np.arange(len(values)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(values) - 1 + 0.6)
    ax.set_xlabel("Throughput Divider [Mpps]")
    plt.tight_layout()
    plt.show()


def _count_num_cores(graph: nx.Graph) -> int:
    n_cores = 0
    for n, d in graph.nodes(data=True):
        if d['node_type'] != 'cpu': continue
        for v in nx.neighbors(graph, n):
            if graph.nodes[v]['node_type'] == 'vnf' and graph.nodes[v]['name'] != 'divider':
                n_cores += 1
                break
    return n_cores


def count_num_cores(all_graphs: Dict[str, List[nx.Graph]]) -> Dict[str, np.array]:
    used_cores = {}
    for k, graphs in all_graphs.items():
        used_cores[k] = []
        for graph in graphs:
            n_cores = _count_num_cores(graph)
            used_cores[k].append(n_cores)
    used_cores = {k: np.array(v, dtype=np.float32) for k, v in used_cores.items()}
    return used_cores


def plot_num_used_cores(all_graphs: Dict[str, List[nx.Graph]]) -> None:
    used_cores = count_num_cores(all_graphs)
    labels, values = zip(*[(k, v) for k, v in used_cores.items()])
    ax = plt.subplot()
    ax.violinplot(
        positions=np.arange(len(values)),
        dataset=values,
        vert=False
    )
    ax.scatter([np.median(v) for v in values], np.arange(len(values)), marker='|')
    ax.scatter([np.mean(v) for v in values], np.arange(len(values)), marker='s', edgecolor='black', facecolor='white')
    ax.set_yticks(np.arange(len(values)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(values) - 1 + 0.6)
    ax.set_xlabel("Num Used CPU Cores")
    plt.tight_layout()
    plt.show()


def plot_individual_core_utilization(graphs: List[nx.Graph], title: str=None) -> None:
    core_utils = {}
    for graph in graphs:
        for n, d in graph.nodes(data=True):
            if d['node_type'] == 'cpu':
                if d['name'] not in core_utils: core_utils[d['name']] = []
                core_utils[d['name']].append(d['demand'] / 2.2e9)
    ax = plt.subplot()
    fig = plt.gcf()
    fig.set_figheight(8)
    labels, values = zip(*[(k, v) for k, v in core_utils.items()])
    pos = np.arange(len(labels))
    ax.violinplot(
        positions=pos,
        dataset=values,
        vert=False
    )
    ax.plot([0.8, 0.8], [0, len(labels)], color='black', linestyle='--')
    ax.plot([0.9, 0.9], [0, len(labels)], color='black', linestyle='--')
    ax.plot([1.0, 1.0], [0, len(labels)], color='black', linestyle='--')
    ax.set_yticks(pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(-0.1, 1.1)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_demands(graphs: List[nx.Graph]) -> None:
    demands = []
    for graph in graphs:
        for n, d in graph.nodes(data=True):
            if d['node_type'] == 'cpu':
                if d['demand'] == 0: continue
                demands.append(d['demand'] / 2.2e9)
    plt.violinplot(demands)
    plt.show()
    plt.ylim(0, 1)


def get_simulated_rate_validation():
    _, val_files = grutils.dset_graph_files()
    val_graphs = [dutils.load_graph(f) for f in val_files]
    z = []
    y = []
    for graph in val_graphs:
        name2idx = {d['name']: n for n, d in graph.nodes(data=True)}
        gi, machines, sfcs, job2machine, job2tx, tx_threads = dutils.problem_from_experiment(graph.graph['path'])
        cpu_performance = tp_sim.simulate_achieved_rate(job2machine, 1e-3)
        tp_sim.update_rates(cpu_performance, sfcs)
        for i, sfc in enumerate(sfcs):
            for j, vnf in enumerate(sfc.jobs):
                z.append(graph.nodes[name2idx[f'vnf_{i}_{j}']]['throughput'])
                y.append(vnf.rate())
    return np.array(z), np.array(y)


def get_simulated_soft_overload():
    _, val_files = grutils.dset_graph_files()
    val_graphs = [dutils.load_graph(f) for f in val_files]
    z = []
    y = []
    for graph in val_graphs:
        name2idx = {d['name']: n for n, d in graph.nodes(data=True)}
        gi, machines, sfcs, job2machine, job2tx, tx_threads = dutils.problem_from_experiment(graph.graph['path'])
        machine_load = {}
        max_demand = {}
        for job, machine in job2machine.items():
            if machine not in machine_load:
                machine_load[machine] = 0
                max_demand[machine] = 0
            machine_load[machine] += 1 #job.demand
            max_demand[machine] = np.max([max_demand[machine], job.demand])
        overloaded = {}
        for m, demand in machine_load.items():
            if m not in overloaded: overloaded[m] = False
            if 2.2e9 / demand < max_demand[m]:
                overloaded[m] = True
        for m, pred in overloaded.items():
            name = f'cpu_{m.physical_id}'
            if name not in name2idx: continue
            if name2idx[name] not in graph.nodes: continue
            if 'soft_overload' not in graph.nodes[name2idx[name]]: continue
            z.append(graph.nodes[name2idx[name]]['soft_overload'])
            y.append(pred)
    return np.array(z), np.array(y)
