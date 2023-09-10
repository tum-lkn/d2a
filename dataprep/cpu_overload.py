import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm
from typing import List, Tuple, Dict, Any
import json
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import dataprep.utils as dutils


def check_for_overload(exp_folder: str) -> Dict[dutils.Machine, List[List[float]]]:
    gi, machines, sfcs, job_machine, job_threads, threads = dutils.problem_from_experiment(exp_folder)
    vnf_stats = dutils.slice_experiment_period(dutils.load_vnf_stats(exp_folder))
    iid_to_sfc = dutils.map_iid_to_sfc(sfcs, gi)
    iid_to_vnf_idx = dutils.map_iid_to_vnf_chain_index(sfcs, gi)

    rx_drops = {}
    rx_q = {}
    rx_rate = {}
    tx_rate = {}
    cost_rough = {}
    cost_detailed = {}
    cost_detailed_med = {}
    for iid, group in vnf_stats.groupby("INSTANCEID"):
        duration = (pd.to_datetime(group.TS.iloc[-1], unit='ms') -
                    pd.to_datetime(group.TS.iloc[0], unit='ms')).seconds
        rx_drops[iid] = group.RX_DROP.max()
        rx_q[iid] = group.RX_Q.mean()
        rx_rate[iid] = group.RX.max() / duration
        tx_rate[iid] = group.TX.max() / duration
        cost_rough[iid] = group.COST.max() / group.TX.max()
        tx = np.clip(group.TX.values[1:] - group.TX.values[:-1], 1, 1e12).astype(np.float32)
        cost = group.COST.values[1:] - group.COST.values[:-1]
        rel_cost = cost.astype(np.float32) / tx
        rel_cost = rel_cost[rel_cost > 0]
        cost_detailed[iid] = np.mean(rel_cost)
        cost_detailed_med[iid] = np.median(rel_cost)
    vnfs_to_iids = dutils.map_vnfs_to_iids(gi, sfcs)

    # Getting the features is difficult since the TX counter is not correct, i.e.,
    # packets are counted more than once. For non-dropping VNFs TX = 2 * RX. For
    # dropping VNFs, TX > RX. This makes the estimation of the RX rate of each
    # VNF difficult. To get accurate predictions, we need the correct input rate
    # for a VNF to get the value at which it dropped. If two VNF in a chain drop,
    # then the input rate for the subsequent VNF is smaller. Also, VNFs that can
    # cope with the reduced rate might not be able to handle the original full rate.
    # Using the rate from the input is thus not possible and can result in incorrect
    # values. The easiest approach is using the output rate of each VNF as input
    # rate the next VNF in the chain. Unfortunately the TX counter is not correct
    # in the data. TX is too large since for some reason packets are counted
    # multiple times. Thus, we use the input rate of non-dropping VNFs downstream,
    # and the TX rate of the last VNF (which is correct) to fill in the input
    # rates of dropping VNFs from the bottom. The input rate is then the lower
    # bound, i.e., the VNFs can handle this rate, for more they start dropping.
    # THe lower bound can still be incorrect for upper VNFs, however, a conservative
    # estimate is better for our purpose.
    #
    # Features are: hard_drop, soft_drop, sfc_num, vnf_idx, rough compute estimate,
    #   detailed compute estimate, detailed compute estimate median,
    #   initially estimated rate, initially estimated demand,
    #   initially estimated compute pp, rate, demand
    features = []
    overload_info = {m: [] for m in job_machine.values()}
    idx = 0
    for i, sfc in enumerate(sfcs):
        fill_later = []
        for j, job in enumerate(sfc.jobs):
            iid = vnfs_to_iids[job]
            sfc_num = iid_to_sfc[iid]
            vnf_idx = iid_to_vnf_idx[iid]
            m = job_machine[job]
            features_vnf = [rx_drops[iid] > 0, rx_q[iid] > job.rate() * 0.001,
                            sfc_num, vnf_idx, cost_rough[iid], cost_detailed[iid],
                            cost_detailed_med[iid], job._rate, job.demand,
                            job.vnf.compute_per_packet, None, None]
            overload_info[m].append(features_vnf)
            features.append(features_vnf)
            if iid == gi.first_ids[i] and rx_drops[iid] > 0:
                # First VNF in chain gets the full input rate of the SFC since
                # divider is not a bottleneck.
                features_vnf[-2] = job._rate
                features_vnf[-1] = job.demand
            elif iid == gi.first_ids[i] and rx_drops[iid] == 0:
                # First VNF in chain does not drop. Use its rx rate as input rate
                # in case of an earlier bottleneck in the TX thread.
                features_vnf[-2] = rx_rate[iid]
                features_vnf[-1] = job.demand
            elif iid == gi.first_ids[i] + len(sfc.jobs) - 1:
                # If the VNF is the last VNF in the SFC, then the TX rate is
                # correct. This rate is the lower bound on the SFC rate. Each
                # VNF in the chain has at least this rate.
                features_vnf[-2] = tx_rate[iid]
                features_vnf[-1] = features_vnf[-2] * features_vnf[-3]
            elif rx_drops[iid] > 0:
                # If the VNF dropped we do not know the input rate since the RX
                # rate is smaller than what has been sent in.
                fill_later.append((iid, idx, job, features_vnf))
            else:
                # Finally, the SFC did not drop any traffic. The RX info is
                # reliable. The RX of this VNF corresponds to the TX of the
                # predecessor and the RX of the successor.
                features_vnf[-2] = rx_rate[iid]
                features_vnf[-1] = features_vnf[-2] * features_vnf[-3]
            idx += 1
        fill_later.reverse()
        for iid, idx, job, features_vnf in fill_later:
            # For those VNFs that dropped fill in the rate of the successor VNF.
            # We always set the rate of the last VNF in the chain. If all VNFs
            # in the chain drop traffic, the output rate of the last VNF is used
            # as input rate for all other VNFs. The output rate is the lower bound
            # at which threshold dropping starts. If an VNF in between does not
            # drop, then this rate is sued for dropping VNFs earlier in the chain.
            features_vnf[-2] = features[idx + 1][-2]
            features_vnf[-1] = features_vnf[-2] * features_vnf[-3]
    return overload_info


def make_datasets(cpu_stats: List[Dict[dutils.Machine, List[float]]],
                  max_num_nfs_per_core=10) -> Tuple[np.array, np.array, np.array, np.array]:
    logger = logging.getLogger('make_datasets')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    num_cores = 0
    for exp_stats in cpu_stats:
        for vnf_stats in exp_stats.values():
            num_cores += 1
            max_num_nfs_per_core = np.max([max_num_nfs_per_core, len(vnf_stats)])
    z_hard = []
    z_soft = []
    num_nfs = []
    x = np.zeros([num_cores, max_num_nfs_per_core, 10], dtype=np.float32)
    core_idx = 0
    for i, vnf_stats in enumerate(cpu_stats):
        for core_vnfs in vnf_stats.values():
            z_hard.append(False)
            z_soft.append(False)
            num_nfs.append(len(core_vnfs))
            for j, stats in enumerate(core_vnfs):
                z_hard[-1] = z_hard[-1] or stats[0]
                z_soft[-1] = z_soft[-1] or stats[1]
                x[core_idx, j, :] = np.array(stats[2:], dtype=np.float32)
            core_idx += 1
    return z_hard, z_soft, np.array(num_nfs, dtype=np.int32), x


def make_linreg_input(x: np.array, num_nfs: List[int], normalize=True) -> np.array:
    # Features are: rough compute estimate,
    #   detailed compute estimate, detailed compute estimate median,
    #   initially estimated rate, initially estimated demand,
    #   initially estimated compute pp, rate, demand
    x_lr = np.zeros([x.shape[0], 11], dtype=np.float32)
    x_lr[:, :-3] = np.sum(x[:, :, 2:], axis=-2)
    for i in range(x_lr.shape[0]):
        x_lr[i, -3] = float(num_nfs[i])
        x_lr[i, -2] = float(np.unique(x[i, :, 0]).size)
        tmp = {}
        for j in range(int(num_nfs[i])):
            if x[i, j, 0] not in tmp:
                tmp[x[i, j, 0]] = 0
            tmp[x[i, j, 0]] += 1
        ratios = np.array(list(tmp.values())) / np.sum(list(tmp.values()))
        x_lr[i, -1] = -1. * np.sum(ratios * np.log(ratios))
    columns = ['rough_measured_compute', 'detailed_measured_compute_avg', 'detailed_measured_compute_median',
               'expected_rate', 'expected_demand', 'expected_cost', 'measured_rate',
               'measured_demand', 'num_nfs', 'num_sfcs', 'sfc_purity']
    if normalize:
        min = np.expand_dims(np.min(x_lr, axis=0), axis=0)
        max = np.expand_dims(np.max(x_lr, axis=0), axis=0)
        x_lr = (x_lr - min) / (max - min)
        x_lr = pd.DataFrame(x_lr, columns=columns)
        return x_lr, min, max
    else:
        x_lr = pd.DataFrame(x_lr, columns=columns)
        return x_lr


def get_stats(exp_root_dir: str, stats=None, stop_after=-1):
    logger = logging.getLogger('get_stats')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    if stats is None:
        stats = []
    num_files = len(os.listdir(exp_root_dir))
    last_perc = 0
    for i, f in enumerate(os.listdir(exp_root_dir)):
        perc = int(i / num_files * 100)
        if perc != last_perc:
            last_perc = perc
            print(f'Processed {last_perc:3.2f}% - {i:5d} of {num_files:5d} files')
        exp_dir = os.path.join(exp_root_dir, f)
        try:
            if os.path.isdir(exp_dir):
                stats.append(check_for_overload(exp_dir))
            else:
                pass
        except Exception as e:
            logger.error(f"Failed to process experiment in {exp_dir}.")
            logger.exception(e)
        if 0 < stop_after < len(stats):
            break
    return stats


def eval_logreg(z, x: pd.DataFrame):
    x_pd = x
    x = x.values
    train_val = int(0.8 * x.shape[0])
    idc = np.arange(x.shape[0])
    r = np.random.RandomState(seed=1)
    r.shuffle(idc)
    x = x[idc]
    z = z[idc]
    x_train, x_val = x[:train_val], x[train_val:]
    z_train, z_val = z[:train_val], z[train_val:]
    mean_acc = 0
    coef = np.zeros(x.shape[1])
    logreg = LogisticRegression(penalty='l1', solver='liblinear').fit(x_train, z_train)
    mean_acc += logreg.score(x_val, z_val)
    coef += logreg.coef_.flatten()
    print(f"Accuracy of logistic regression is {mean_acc * 100:.2f}%")
    ax = plt.subplot()
    indices = np.arange(coef.size)
    ax.bar(indices, coef)
    ax.set_ylabel("Importance")
    ax.set_xticks(indices)
    ax.set_xticklabels(x_pd.columns)
    ax.set_title(f"Accuracy: {mean_acc * 100:.2f}%")
    plt.savefig("Graphs/importance-logreg.png")
    plt.show()
    plt.close()
    y = logreg.predict(x_val)
    print(f"Accuracy: {skm.accuracy_score(z_val, y)}")
    print(f"Precision: {skm.precision_score(z_val, y)}")
    print(f"Recall: {skm.recall_score(z_val, y)}")
    return z_val, y


def eval_random_forest(z, x: pd.DataFrame, thresh=0.5):
    x_pd = x
    x = x.values
    train_val = int(0.8 * x.shape[0])
    idc = np.arange(x.shape[0])
    r = np.random.RandomState(seed=1)
    r.shuffle(idc)
    x = x[idc]
    z = z[idc]
    x_train, x_val = x[:train_val], x[train_val:]
    z_train, z_val = z[:train_val], z[train_val:]
    mean_acc = 0
    coef = np.zeros(x.shape[1])

    model = RandomForestClassifier(100).fit(x_train, z_train)
    mean_acc += model.score(x_val, z_val)
    coef += model.feature_importances_.flatten()

    coef /= 1
    mean_acc /= 1
    print(f"Accuracy of random forest is {mean_acc * 100:.2f}%")
    ax = plt.subplot()
    indices = np.arange(coef.size)
    ax.bar(indices, coef)
    ax.set_ylabel("Importance")
    ax.set_xticks(indices)
    ax.set_xticklabels(x_pd.columns)
    ax.set_title(f"Accuracy: {mean_acc * 100:.2f}%")
    plt.savefig("Graphs/importance-random-forest.png")
    plt.show()
    plt.close()
    y = model.predict(x_val)
    print(f"Accuracy: {skm.accuracy_score(z_val, y)}")
    print(f"Precision: {skm.precision_score(z_val, y)}")
    print(f"Recall: {skm.recall_score(z_val, y)}")

    y2 = (model.predict_proba(x_val)[:, 1] > thresh).astype(np.int32)
    print(f"Accuracy: {skm.accuracy_score(z_val, y2)}")
    print(f"Precision: {skm.precision_score(z_val, y2)}")
    print(f"Recall: {skm.recall_score(z_val, y2)}")
    return z_val, y
