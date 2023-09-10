import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from typing import List, Dict, Tuple, Any


matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth
MARKER = ['s', 'o', 'v', '^', '<', '>']
# COLORS = ['#80b1d3','#fb8072','#fdb462','#bebada','#8dd3c7']
# COLORS = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
COLORS = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
# NAMES = ['RR', 'LLF', "DaDrA-LB", "DaDrA-LB-DT", 'FFD', "DaDrA-BP", "DaDrA-BP-DT"]
NAMES = ['\\texttt{{RR}}', '\\texttt{{LLF}}', "\\texttt{{LB\\textsubscript{{HS}}}}",
         "\\texttt{{LB\\textsubscript{{Col}}}}", '\\texttt{{FFD}}',
         "\\texttt{{BP\\textsubscript{{HS}}}}", "\\texttt{{BP\\textsubscript{{Col}}}}"]
HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
NAME2COL = {n: COLORS[i] for i, n in enumerate(NAMES)}
NAME2MARK = {n: MARKER[i % len(MARKER)] for i, n in enumerate(NAMES)}


def get_fig(ncols: float, aspect_ratio=0.618) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create figure and axes objects of correct size.

    Args:
        ncols (float): Percentage of one column in paper.
        aspect_ratio (float): Ratio of width to height. Default is golden ratio.

    Returns:
        fig (plt.Figure)
        ax (plt.Axes)
    """
    COLW = 3.45
    ax = plt.subplot()
    fig = plt.gcf()
    fig.set_figwidth(ncols * COLW)
    fig.set_figheight(ncols * COLW * aspect_ratio)
    return fig, ax