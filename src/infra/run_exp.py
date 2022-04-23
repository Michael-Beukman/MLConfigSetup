import os
import subprocess

from common.utils import get_date, get_dir, path
from common.vars import CONDA_ENV_NAME, CONFIG_DIR, LOG_DIR, ROOT_DIR as ROOT_DIR_FUNC, SLURM_DIR, SLURM_LOG_DIR
import fire
from infra.config import YamlConfig, Config

def main(
    partition_name: str,
    yaml_config_file: str,
    use_slurm: bool = True, local: bool = None
):
    """This creates a slurm file and runs it

    Args:
        partition_name (str): Partition to run the code on
        yaml_config_file (str): The config file to use for everything
        use_slurm (bool): If true, uses slurm, otherwise executes the script with bash
    """
    date = get_date()
    if not os.path.exists(yaml_config_file):
        # Only partial name
        yaml_config_file = path(CONFIG_DIR, yaml_config_file.split("-")[0], yaml_config_file + ".yaml")
    
    assert os.path.exists(yaml_config_file)

    conf = Config(YamlConfig(yaml_config_file), seed='all', date=date)

    assert yaml_config_file.split("/")[-1].replace(".yaml", '') == conf.yaml_config('meta/run_name'), "The experiment name must be the same as the config file name"

    if local is None: local = not use_slurm
    hashes = conf.hash(True, True)
    ROOT_DIR = ROOT_DIR_FUNC(local)
    python_name = conf.yaml_config('meta/file')
    # Create Slurm File
    s = f'''#!/bin/bash
#SBATCH -p {partition_name}
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -J {conf.experiment_name}
#SBATCH -o {ROOT_DIR}/{get_dir(SLURM_LOG_DIR, conf.experiment_name)}/{date}-{hashes}.%N.%j.out

source ~/.bashrc
cd {ROOT_DIR}
conda activate {CONDA_ENV_NAME}
echo "{conf.experiment_name} -- with {yaml_config_file} -- {hashes}"
./run.sh {python_name} --is-local {local} --date {date} --overall-name {hashes} --yaml-config-file {yaml_config_file}
'''
    dir = get_dir(SLURM_DIR, conf.experiment_name)
    fpath = os.path.join(dir, f'{hashes}.slurm')
    with open(fpath, 'w+') as f:
        f.write(s)
    
    # Run it    
    if use_slurm:
        ans = subprocess.call(f'sbatch {fpath}'.split(" "))
    else:
        ans = subprocess.call(f'bash {fpath}'.split(" "))
    assert ans == 0
    print("Successfully Ran")
    
if __name__ == '__main__':
    fire.Fire(main)