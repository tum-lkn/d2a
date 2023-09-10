import numpy as np
import pandas as pd
import networkx as nx
import os
import sys
import matplotlib.pyplot as plt
import subprocess
from importlib import reload
from typing import List, Tuple, Dict, Any
import json

import dataprep.utils as dutils
import dataprep.graph as dgr
import scripts.plot_results as evaldqn
import evaluation.plotutils as plutils

some_exp_dir = '/opt/project/data/nas/dqn_assignments_result/generated-1-mpps/dqn_eval-p0067-i00'

#%%
# Obtain data.
results = evaldqn.eval_test_runs()
with open("/opt/project/data/moongen-throughput-results.json", 'w') as fh:
    json.dump(results, fh)
results = {k: np.array(v) for k, v in results.items()}

packet_cpu = evaldqn.eval_aboslute_packet_cost()
with open("/opt/project/data/absolute-per-packet-cost.json", "w") as fh:
    json.dump(packet_cpu, fh)
total_cost_per_packet = {k: np.array([v['cpus'] * 2.2e9 * 10 / v['packets'] for v in vals \
                         if v['packets'] > 0]) for k, vals in packet_cpu.items()}
latencies = evaldqn.latency_test_runs()
latencies = {k: [float(v) for v in vals] for k, vals in latencies.items()}
with open("/opt/project/data/moongen-latencies.json", "w") as fh:
    json.dump(latencies, fh)
latencies = {k: np.array(v) for k, v in latencies.items()}

all_graphs = evaldqn.load_graphs_for_test_runs()
for k, graphs in all_graphs.items():
    base_dir = os.path.join('/opt/project/data/graphs-golden-samples', k)
    if os.path.exists(base_dir):
        continue
    else:
        os.mkdir(base_dir)
        for i, g in enumerate(graphs):
            dutils.save_graph(g, os.path.join(base_dir, f'graph_{i}.json'), g.graph['exp_dir'])
flat_graphs = []
for gs in all_graphs.values():
    flat_graphs.extend(gs)
features = dutils._analyze_graphs(flat_graphs)

all_graphs = {}
d = '/opt/project/data/nas/graphs-golden-samples'
for x in os.listdir(d):
    d = os.path.join(d, x)
    if os.path.isdir(bd):
        all_graphs[x] = []
        for gf in os.listdir(bd):
            if not gf.endswith('json'):
                continue
            all_graphs[x].append(dutils.load_graph(os.path.join(bd, gf)))

#%%
# Check the failed CPU cores.
soft_failure_graphs = []
hard_failure_graphs = []
no_failure_graphs = []
for lbl, graphs in all_graphs.items():
    for graph in graphs:
        hard_added = False
        soft_added = False
        for n, d in graph.nodes(data=True):
            if d['node_type'] != 'cpu': continue
            if 'hard_overload' not in d: continue
            if d['hard_overload'] and not hard_added:
                hard_failure_graphs.append(graph)
                hard_added = True
            if d['soft_overload'] and not soft_added:
                soft_failure_graphs.append(graph)
                soft_added = True
        if not (hard_added or soft_added):
            no_failure_graphs.append(graph)

cpu_features = []
for graph in hard_failure_graphs:
    rets = dgr.convert_graph(graph, normalize=False)
    for cpu_idx in rets['cpu_nodes']:
        if 'hard_overload' in graph.nodes[cpu_idx]:
            if graph.nodes[cpu_idx]['hard_overload']:
                cpu_features.append(rets['node_features'][0, cpu_idx, :])
cpu_features = np.row_stack(cpu_features)
cpu_features_ = cpu_features[:, [18, 19, 20, 21]]
mins = np.array([[features['min_num_vnfs_on_cpu_core'], features['min_rate_cpu'],
        features['min_cost_cpu'], features['min_demand_cpu']]])
maxs = np.array([[features['max_num_vnfs_on_cpu_core'], features['max_rate_cpu'],
        features['max_cost_cpu'], features['max_demand_cpu']]])
tmp = pd.DataFrame(cpu_features_ * (maxs - mins) + mins, columns=['num_vnfs', 'rate', 'cost', 'demand'])
tmp = pd.DataFrame(cpu_features_, columns=['num_vnfs', 'rate', 'cost', 'demand'])


#%%
all_features = []
for g in flat_graphs: #all_graphs['RL-1N-2.5']:
    for u, attrs in g.nodes(data=True):
        if attrs['node_type'] == 'cpu' and 'soft_overload' in attrs: # and not attrs['hard_overload']:
            node_features = np.zeros(8)
            node_features[0] = float(np.sum([1 if g.nodes[m]['node_type'] == 'vnf' else 0 for m in g.neighbors(u)]))
            node_features[1] = float(np.min([g.nodes[m]['ratio'] if g.nodes[m]['node_type'] == 'vnf' else 1e9 for m in g.neighbors(u)]))
            node_features[2] = float(np.max([g.nodes[m]['ratio'] if g.nodes[m]['node_type'] == 'vnf' else 0 for m in g.neighbors(u)]))
            node_features[3] = np.abs(1 - node_features[1] / node_features[2])
            node_features[4] = attrs['arrival_rate']
            node_features[5] = attrs['cost']
            node_features[6] = attrs['demand'] / 2.2e9
            node_features[7] = attrs['hard_overload']
            all_features.append(node_features)
tmp2 = pd.DataFrame(np.row_stack(all_features), columns=['num_vnfs', 'min_ratio', 'max_ratio', 'ratioratio', 'rate', 'cost', 'demand', 'overlaod'])
# tmp2 = tmp.loc[tmp2.rate < 2.5e6]
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as skm
features = ['num_vnfs', 'rate', 'cost', 'demand', 'min_ratio', 'max_ratio']
mask = np.logical_not(np.logical_and(tmp2.demand.values > 0.8, tmp2.overlaod.values == 0))
X = tmp2.loc[mask, features]
scaler = MinMaxScaler().fit(X)
X = scaler.transform(X)
model = LogisticRegression(penalty='l1', solver='liblinear')
z = tmp2.values[mask, -1]
weights = z * 10 + 1
# weights = np.ones(X.shape[0])
model.fit(X, z, sample_weight=weights)
y = model.predict(X)
# y[tmp2.demand.values > 0.8] = 1
print(skm.accuracy_score(z, y))
print(skm.precision_score(z, y))
print(skm.recall_score(z, y))
print(pd.DataFrame(np.row_stack((model.coef_, scaler.data_min_, scaler.data_max_)), columns=features, index=['coef', 'min', 'max']))
print("Intercept", model.intercept_)

print(features)
print(model.coef_)
print(scaler.data_min_)
print(scaler.data_max_)
tmp3 = tmp2.loc[np.logical_and(z == 0, y == 1)]


#%%
# Visualize basic stats.
format = 'pdf'
evaldqn.plot_total_packet_cost_paper(total_cost_per_packet, '2.5')
plt.savefig(f'Graphs/total-cost-pp-2.5mpps.{format}')
plt.show()
plt.close('all')
evaldqn.plot_total_packet_cost_paper(total_cost_per_packet, '1')
plt.savefig(f'Graphs/total-cost-pp-1mpps.{format}')
plt.show()
plt.close('all')

evaldqn.plot_throughput(results)
plt.show()
evaldqn.plot_throughput_paper(results, '2.5')
plt.savefig(f'Graphs/throughput-2.5mpps.{format}')
plt.show()
plt.close('all')
evaldqn.plot_throughput_paper(results, '1')
plt.savefig(f'Graphs/throughput-1mpps.{format}')
plt.show()
plt.close('all')

evaldqn.plot_divider_throughput(all_graphs)

evaldqn.plot_num_used_cores(all_graphs)
evaldqn.plot_num_used_cores_paper(all_graphs, '2.5')
plt.savefig(f'Graphs/used-cores-2.5mpps.{format}')
plt.show()
plt.close('all')
evaldqn.plot_num_used_cores_paper(all_graphs, '1')
plt.savefig(f'Graphs/used-cores-1mpps.{format}')
plt.show()
plt.close('all')

evaldqn.plot_latency_paper(latencies_rc, '2.5', vert=False)
plt.savefig(f'Graphs/latencies-2.5mpps.{format}')
plt.show()
plt.close('all')
evaldqn.plot_latency_paper(latencies_rc, '1')
plt.savefig(f'Graphs/latencies-1mpps.{format}')
plt.show()
plt.close('all')

for k, graphs in all_graphs.items():
    evaldqn.plot_individual_core_utilization(graphs, k)
#%%
subprocess.call(
    "cp /opt/project/data/nas/dqn_assignment_result_test/least-loaded-first-decreasing-l100/"
    "dqn_eval-p0077-i00/Problem.gv.pdf "
    "/opt/project/Graphs/Problem-llfd-77.pdf", shell=True)

subprocess.call(
    "cp /opt/project/data/nas/dqn_assignment_result_test/first-fit-decreasing-l100/"
    "dqn_eval-p0042-i00/Problem.gv.pdf "
    "/opt/project/Graphs/Problem-ffd-42.pdf", shell=True)

#%%
# Take a look at a larger problem in the LLF setting.
stats = dutils.slice_experiment_period(dutils.load_vnf_stats(
    "data/nas/dqn_assignment_result_test/least-loaded-first-decreasing-l100/dqn_eval-p0077-i00"
))
tmp_7 = stats.set_index("INSTANCEID").loc[7, :]
tmp_20 = stats.set_index("INSTANCEID").loc[20, :]
t1 = tmp_20.iloc[5000, 0]
td = pd.Timedelta(100, unit='ms')
ax = plt.subplot()
tmp_20.set_index("TS").loc[t1:t1 + td, :].diff().TX.dropna().plot(label='NF20', ax=ax)
tmp_7.set_index("TS").loc[t1:t1 + td, :].diff().TX.dropna().plot(label='NF7', ax=ax)
ax.set_ylabel("Throughput pp/ms")
plt.legend(frameon=False)
plt.show()

combined = pd.concat([
    tmp_20.loc[:, ['TX', 'COST', 'RX', 'RX_Q', 'RX_DROP', 'BURST']].diff().dropna().reset_index(drop=True),
    tmp_7.loc[:, ['TX', 'COST', 'RX', 'RX_Q', 'RX_DROP', 'BURST']].diff().dropna().reset_index(drop=True)
], axis=1)

#%%
# Take a look at a larger problem in the LLF setting.
stats = dutils.slice_experiment_period(dutils.load_vnf_stats(
    "data/nas/dqn_assignment_result_test/first-fit-decreasing-l100/dqn_eval-p0042-i00"
))
tmp_6 = stats.set_index("INSTANCEID").loc[6, :]
tmp_7 = stats.set_index("INSTANCEID").loc[7, :]
t1 = tmp_7.iloc[5000, 0]
td = pd.Timedelta(100, unit='ms')
ax = plt.subplot()
tmp_6.set_index("TS").loc[t1:t1 + td, :].diff().TX.dropna().plot(label='NF6', ax=ax)
tmp_7.set_index("TS").loc[t1:t1 + td, :].diff().TX.dropna().plot(label='NF7', ax=ax)
ax.set_ylabel("Throughput pp/ms")
plt.legend(frameon=False)
plt.show()

combined = pd.concat([
    tmp_6.loc[:, ['TX', 'COST', 'RX', 'RX_Q', 'RX_DROP', 'BURST']].diff().dropna().reset_index(drop=True),
    tmp_7.loc[:, ['TX', 'COST', 'RX', 'RX_Q', 'RX_DROP', 'BURST']].diff().dropna().reset_index(drop=True)
], axis=1)
#%%
tmp_1 = stats.set_index("INSTANCEID").loc[1, :]
tmp_1.set_index("TS", inplace=True)
tmp_1.diff().dropna().reset_index().TX.plot()
plt.show()

tmp_1.diff().dropna().reset_index().RX.plot()
plt.show()

#%%


# bpath = '/opt/project/data/rate-cost-cfs-no-yield-schedlat-10ms'
# bpath = '/opt/project/data/rate-cost-cfs-schedlat-10ms'
# bpath = '/opt/project/data/basic-cfs-no-yield'
bpath = '/opt/project/data/basic-cfs'
df_short = pd.read_csv(os.path.join(bpath, 'stats-short-worker-3.csv'), sep=',')
df_short2 = pd.read_csv(os.path.join(bpath, 'stats-second-short-worker-3.csv'), sep=',')
df_long = pd.read_csv(os.path.join(bpath, 'stats-long-worker-3.csv'), sep=',')
df_short = df_short.loc[df_short.ticks > 0]
df_short2 = df_short2.loc[df_short2.ticks > 0]
df_long = df_long.loc[df_long.ticks > 0]
yield_short = (df_short - df_long.ticks.iloc[0]) / 2.2e6
yield_short2 = (df_short2 - df_long.ticks.iloc[0]) / 2.2e6
yield_long = (df_long - df_long.ticks.iloc[0]) / 2.2e6


diff_short = yield_short.diff().dropna()
diff_long = yield_long.diff().dropna()
diff_short2 = yield_short2.diff().dropna()

ref = np.min([df_long.ticks.values[0], df_short.ticks.values[0], df_short2.ticks.values[0]])
s = ref + 2.2e9
e = s + 0.1 * 2.2e9
# x_long = (df_long.ticks.loc[df_long.ticks.values < ref + 0.25 * 2.2e9].values - ref) / 2.2e6
tmp = df_long.loc[df_long.ticks.values > s]
x_long = (tmp.ticks.loc[tmp.ticks.values < e] - ref) / 2.2e6
y_long = np.repeat(1, x_long.size)
# x_short = (df_short.ticks.loc[df_short.ticks.values < ref + 0.25 * 2.2e9].values - ref) / 2.2e6
tmp = df_short.loc[df_short.ticks.values > s]
x_short = (tmp.ticks.loc[tmp.ticks.values < e] - ref) / 2.2e6
y_short = np.repeat(2, x_short.size)
# x_short2 = (df_short2.ticks.loc[df_short2.ticks.values < ref + 0.25 * 2.2e9].values - ref) / 2.2e6
tmp = df_short2.loc[df_short2.ticks.values > s]
x_short2 = (tmp.ticks.loc[tmp.ticks.values < e] - ref) / 2.2e6
y_short2 = np.repeat(3, x_short2.size)
fig, ax = plutils.get_fig(1, 0.1)
ax.scatter(x_long, y_long, s=5, label='long')
ax.scatter(x_short, y_short, s=5, label='short')
ax.scatter(x_short2, y_short2, s=5, label='short2')
ax.set_yticks([1 ,2 ,3])
ax.set_yticklabels(['Heavy', 'Light1', 'Light2'])
ax.set_xlabel("Time in ms")
# ax.legend(frameon=False)
plt.tight_layout()
plt.savefig('Graphs/activity-1ms-no-yield.png')
plt.show()

ax = plt.subplot()
ax.violinplot(
    dataset=[
        diff_short.ticks.loc[diff_short.ticks.values > 0.15].values,
        diff_short2.ticks.loc[diff_short2.ticks.values > 0.15].values,
        diff_long.ticks.loc[diff_long.ticks.values > 0.15].values,
    ],
    positions=[1, 2, 3],
    showextrema=False,
    vert=False
)
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(["Light", "Light", "Heavy"])
ax.set_xlabel("Inactive time [ms]")
plt.tight_layout()
plt.savefig("Graphs/inactive-times-1-ms-no-yield.png")
plt.show()

#%%
def make_fill_between_lines(df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    acti_x = []
    acti_y = []
    ticks = df.ticks
    for t1, t2 in zip(ticks.iloc[:-1].values, ticks.iloc[1:].values):
        delta = (t2 - t1) / 2.2e6
        if delta < 0.2:
            if len(acti_x) == 0:
                acti_x.append(t1)
                acti_y.append(1)
            acti_x.append(t2)
            acti_y.append(1)
        else:
            acti_x.extend([t1, t2, t2])
            acti_y.extend([0, 0, 1])
    return acti_x, acti_y

ref = np.min([df_long.ticks.values[0], df_short.ticks.values[0], df_short2.ticks.values[0]])
s = ref + 1.10 * 2.2e9
e = s + 0.075 * 2.2e9
first_ms = (int(s / 2.2e6) + 1) * 2.2e6
xticks = np.arange(first_ms, e, 25 * 2.2e6)

fig, ax = plutils.get_fig(1)
tmp = df_long.loc[np.logical_and(df_long.ticks.values > s, df_long.ticks.values < e)]
acti_x, acti_y = make_fill_between_lines(tmp)
ax.fill_between(acti_x, np.zeros(len(acti_y)), acti_y, fc=plutils.COLORS[0], hatch="+++", label='Heavy')

tmp = df_short.loc[np.logical_and(df_short.ticks.values > s, df_short.ticks.values < e)]
acti_x, acti_y = make_fill_between_lines(tmp)
ax.fill_between(acti_x, np.zeros(len(acti_y)), acti_y, fc=plutils.COLORS[1], hatch="***", label='Light1')

tmp = df_short2.loc[np.logical_and(df_short2.ticks.values > s, df_short2.ticks.values < e)]
acti_x, acti_y = make_fill_between_lines(tmp)
ax.fill_between(acti_x, np.zeros(len(acti_y)), acti_y, fc=plutils.COLORS[2], hatch="...", label='Light2')

ax.set_yticks([])
ax.set_xticks(xticks)
ax.set_xticklabels(25 * np.arange(len(xticks)))
ax.set_xlabel("Time [ms]")
ax.set_xlim(xticks[0], xticks[-1])
ax.set_ylim(0, 1)
leg = plt.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.175), ncol=3)
for patch in leg.get_patches():
    patch.set_height(patch.get_height() * 2)
plt.tight_layout()
plt.savefig("Graphs/process-activity-basic-cfs-yield.pdf")
plt.show()
plt.close('all')

#%%
latencies_rc = evaldqn.latency_test_runs('rc')
latencies_cfs = evaldqn.latency_test_runs('cfs')

throughput_rc = evaldqn.eval_test_runs('rc')
max25 = np.max(throughput_rc['LLF-100-2.5'])
max1 = np.max(throughput_rc['LLF-100-1'])
throughput_rc = {k: np.array(v) / (max25 if k.find('-2.5') >= 0 else max1) * 100 for k, v in throughput_rc.items()}

throughput_cfs = evaldqn.eval_test_runs('cfs')
max25 = np.max(throughput_cfs['LLF-100-2.5'])
max1 = np.max(throughput_cfs['LLF-100-1'])
throughput_cfs = {k: v / (max25 if k.find('-2.5') >= 0 else max1) * 100 for k, v in throughput_cfs.items()}

tcpp_rc = {k: np.array([v['cpus'] * 2.2e9 * 10 / v['packets'] for v in vals
                        if v['packets'] > 0]) for k, vals in evaldqn.eval_aboslute_packet_cost('rc').items()}
tcpp_cfs = {k: np.array([v['cpus'] * 2.2e9 * 10 / v['packets'] for v in vals
                        if v['packets'] > 0]) for k, vals in evaldqn.eval_aboslute_packet_cost('cfs').items()}

factory = {
    'rc': {
        'throughput': throughput_rc,
        'tcpp': tcpp_rc,
        'latency': latencies_rc
    },
    'cfs': {
        'throughput': throughput_cfs,
        'tcpp': tcpp_cfs,
        'latency': latencies_cfs
    },
}

paper_algos = ['RR-100', 'LLF-100', 'RL-LB-1N', 'RL-LB-1N-DT', 'FFD-100', 'RL-1N', 'RL-1N-DT2']
paper_algo_cmds = ['\\rr', '\\llf', '\\lb', '\\lbdt', '\\ffd', '\\bipa', '\\bipadt']
lines = []
for prefix, cmd in zip(paper_algos, paper_algo_cmds):
    lines.append([])
    for metric in ['throughput', 'tcpp', 'latency']:
        for rate in ['2.5', '1']:
            for mode in ['cfs', 'rc']:
                values = factory[mode][metric][f"{prefix}-{rate}"] * (100 if metric == 'throughput' else 1)
                lines[-1].append(np.mean(values))

best_indices = np.concatenate((np.argmax(np.array(lines)[:, :4], axis=0), np.argmin(np.array(lines)[:, 4:], axis=0)))
latex_lines = []
val_lengths = [4, 4, 4, 4, 9, 9, 9, 9, 4, 4, 4, 4]
for row, cmd in enumerate(paper_algo_cmds):
    latex_line = [f"{cmd:8} "]
    latex_lines.append(latex_line)
    for col, val in enumerate(lines[row]):
        begin = '\\bm{' if row == best_indices[col] else '    '
        end = '}' if row == best_indices[col] else ' '
        middle = f'{np.round(val, 2):.2f}'
        if val >= 1000:
            middle = f'{middle[0]}\\,{middle[1:]}'
        latex_line.append({
            0: f" ${begin}{middle:5s}{end}$ ",
            1: f" ${begin}{middle:9s}{end}$ ",
            2: f" ${begin}{middle:4s}{end}$ ",
        }[int(col / 4)])
for line in latex_lines:
    print('&'.join(line) + ' \\\\')


tmp = []
tmp.extend([f'{a}-1' for a in paper_algos])
tmp.extend([f'{a}-2.5' for a in paper_algos])
for k, v in latencies_rc.items():
    if k not in tmp: continue
    print(f"{k:20s}", np.mean(v))
for k, v in latencies_cfs.items():
    if k not in tmp: continue
    print(f"{k:20s}", np.mean(v))
