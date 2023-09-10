"""
Contains code that analyzes per output.
"""
import pandas as pd
import os


def load_perf(full_exp_path, run=0) -> pd.DataFrame:
    """
    Read the performance results in as a data frame and return it.

    The returned data frame has s hierarchical index on the rows. The first
    level corresponds to individual CPUs. The hierarchy corresponds to performance
    counters of perf.

    perf was started with: `perf stat (-e <counter>)+ -a -A -I 1000 -o <path/perf.csv>`

    Example:
        1.003357723	CPU30	484257		    L1-dcache-loads	        1002042185	100.00
        1.003357723	CPU31	425769		    L1-dcache-loads	        1002035171	100.00
        1.003357723	CPU0	3842670		    L1-dcache-load-misses	1002059166	100.00	0.63	of all L1-dcache hits
        1.003357723	CPU31	1908958		    cycles	                1001664766	100.00
        1.003357723	CPU0	1733771469		instructions	        1001672772	100.00	0.79	insn per cycle

        In case of the L1-dcache-load-misses the `opt_value` column contains
        the cache miss rate, i.e., its `L1-dcache-load / L1-dcache-load-misses.
        In case of of `value=instructions`, the `opt_value` columns contains
        the number of instructions per cycle.
        
        For more info see https://man7.org/linux/man-pages/man1/perf-stat.1.html
        section CSV output format.

    Args:
        full_exp_path: Path to the directory storing results of an experiment.
        run: The run number that should be checked.

    Returns:
        df: data frame with results.
    """
    columns = [
        'timestamp',
        'cpu',
        'value',
        'unit',
        'event',
        'run_time_counter',
        'percentage_counter_active',
        'opt_value',
        'opt_unit'
    ]
    df = pd.read_csv(
        os.path.join(full_exp_path, "{:d}".format(run), 'NFControlPlane', 'perf.csv'),
        sep=',',
        engine='python',  # File has ragged rows, thus python engine is needed.
        skiprows=1,
        names=columns  # Files have twelve columns
    )
    return df.set_index(["CPU", "event"]).sort_index()
