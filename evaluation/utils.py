import os


def get_pbt_max_checkpoints(exp_dir: str):
    rets = []
    for f in os.listdir(exp_dir):
        p = os.path.join(exp_dir, f)
        if os.path.isdir(p):
            rets.append((p, get_max_checkpoint(p)))
    return rets


def get_max_checkpoint(trial_dir: str) -> str:
    check_dirs = []
    for f in os.listdir(trial_dir):
        p = os.path.join(trial_dir, f)
        if os.path.isdir(p) and f.startswith('checkpoint'):
            check_dirs.append(f)
    check_dirs.sort(key=lambda x: int(x.split('_')[1].lstrip('0')))
    return check_dirs[-1]


def prepare_checkpoint_path(trial_dir: str, checkpoint_dir: str) -> str:
    name, num = checkpoint_dir.split('_')
    num = num.lstrip('0')
    return os.path.join(trial_dir, checkpoint_dir, f'{name}-{num}')
