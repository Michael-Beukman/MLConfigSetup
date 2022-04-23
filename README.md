# Config

- [Config](#config)
  - [How it works](#how-it-works)
    - [VsCode Setup](#vscode-setup)
    - [Running](#running)
  - [Setup](#setup)
  - [Next Steps](#next-steps)
  - [Acknowledgements](#acknowledgements)



This is a small repo that is one possible way to set up a maintainable infrastructure for experiments.

It does the following:
- Runs multiple seeds in parallel
- Has a config-based system
- Logs results to Weights and Biases


## How it works
Basically, we have a `YamlConfig` class, which reads the configuration from that contained in `artifacts/config/xxxx/xxxx.yaml`. We use this to run code using the number of seeds described in the configuration file. Further, this is ran in parallel too (using as many processes as specified in the configuration file).

Finally, there is a `infra/run_exp.py` file that creates a job script for a specific setting, and either runs using e.g. `slurm` or `bash`.

### VsCode Setup
If you have the `.env` file in your project, it will give intellisense for all of the classes in the code. Possibly the `/` must be changed to `\` or `\\` for Windows.

### Running
To run **any python file**, always be in the directory above `src`, and use `./run.sh`. For example, to run `infra/run_exp.py`, I'd run `./run.sh src/infra/run_exp.py <ARGS>`. On windows, I'd suggest using WSL and installing the VsCode extension for that.

## Setup

You do not have to use conda, but I recommend it. You should definitely install the packages shown below though (stable baselines and gym are just used for the demo, so you can omit them if you do not use them)
```
conda create -n env_name python=3.9
conda activate env_name
pip install ray wandb pyyaml fire 'stable-baselines3[extra]' gym pandas
```
Further, change the following in `src/common/vars.py`. If you do not use `conda`, then remove line 46 in `src/infra/run_exp.py` (which looks like `conda activate {CONDA_ENV_NAME}`)
```
CONDA_ENV_NAME = '<env_name>'
```

And the following paths, for local and remote execution respectively
```
def ROOT_DIR(is_local: bool):
    # Local and remote directories to go to.
    if is_local: 
        return '/home/<username>/<path_to_proj_local>'
    return '/home/<username>/<path_to_proj_remote>'
```


Then, set up `wandb`, see [here](https://docs.wandb.ai/quickstart). Basically just run `wandb login` after installing it.

To run the experiment:
```
./run.sh src/infra/run_exp.py --partition-name batch --use-slurm True --yaml-config-file v0000-PPO-CartPole
```

Argument explanation:
- `--partition-name` Which slurm partition must be used, does not matter for local execution
- `--use-slurm`: If true, assumes a cluster environment and runs the script with `slurm`, otherwise with `bash`.
- `--yaml-config-file`: The name of the config file. This will look for this config file in directory `artifacts/config/xxxx/xxxx-config-name.yaml`, where `xxxx` is the version, i.e. `v0000` in the example.


## Next Steps
Now, you can create a new config file (using the same format:  keep the `meta` details, but add in other parameters relevant to you). Further, create a new file in `src/runs/` to actually use that config to run stuff.

## Acknowledgements

The config idea and implementation was very much inspired by this blog post: https://jonnyjxn.medium.com/how-to-config-your-machine-learning-experiments-without-the-headaches-bb379de1b957