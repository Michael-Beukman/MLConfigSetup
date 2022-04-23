import os


SAVE_PDF = False
SAVE_EPS = False

# Where stuff is stored
ARTIFACTS_DIR       = 'artifacts'
RESULTS_DIR         = 'artifacts/results'
MODELS_DIR          = 'artifacts/models'
TENSORBOARD_DIR     = 'artifacts/tensorboard'
CONFIG_DIR          = 'artifacts/config'
SLURM_DIR           = 'artifacts/slurms'
SLURM_LOG_DIR       = 'artifacts/logs/slurms'
LOG_DIR             = 'artifacts/logs'

# Create directories
CHECK_FIRST = True
if CHECK_FIRST:
    for a in [RESULTS_DIR, MODELS_DIR, CONFIG_DIR, SLURM_DIR, SLURM_LOG_DIR, TENSORBOARD_DIR]: os.makedirs(a, exist_ok=True)

def ROOT_DIR(is_local: bool):
    # Local and remote directories to go to.
    if is_local: 
        return '/home/<username>/<path_to_proj_local>'
    return '/home/<username>/<path_to_proj_remote>'

def NUM_CORES(is_local: bool):
    # How many cores does the local / remote machine have that you can realistically use.
    return 2 if is_local else 10

CONDA_ENV_NAME = '<env_name>'