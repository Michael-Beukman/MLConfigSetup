import shutil

import gym
from common.utils import get_dir, set_seed, path
from common.vars import MODELS_DIR, NUM_CORES
from infra.config import Config, YamlConfig
from infra.logger import WandDBRunLogger
import ray
import fire
import pandas as pd


from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.logger import configure


def main(is_local: bool, date: str, yaml_config_file: str, overall_name: str):
    """The main file -- runs everything in parallel

    Args:
        is_local (bool): Is this local or on a remote cluster machine
        date (str): The date to use -- shared across all seeds, the job script, and results directory.
        yaml_config_file (str): The config file to use
        overall_name (str): The name to store this run at.
    """
    yaml_conf = YamlConfig(yaml_config_file)
    gpus = yaml_conf('infra/gpus', 0)
    
    # Here we limit ray to only use the number of cores we specified in the config.yaml
    CORES = yaml_conf('infra/cores', NUM_CORES(is_local))
    ray.init(num_cpus=CORES, num_gpus=gpus)
    
    # How many seeds to run.
    NUM_SEEDS = yaml_conf("infra/seeds")
    @ray.remote(num_gpus=gpus/CORES)
    def single_run(seed: int):
        # Setup
        set_seed(seed)
        print(f"Running with seed = {seed}")
        config = Config(yaml_conf, seed=seed, date=date)
        name_to_save = f"{config.unique_name()}"
        DIR = get_dir(f"{MODELS_DIR}/{config.experiment_name}/{overall_name}/{name_to_save}")
        log_dir = path(DIR, 'logs')
        
        # Copy the appropriate config for future reference -- only seed 0 does this
        if seed == 0:
            print("Running Group: ", config.hash())
            shutil.copyfile(yaml_config_file, path(f'{MODELS_DIR}/{config.experiment_name}/{overall_name}', yaml_config_file.split("/")[-1]))
        
        # Init wandb 
        run = config.init_wandb(sync_tensorboard=True) # sync_tensorboard is for SB3 too.        
        logger = WandDBRunLogger(config, run)
        # ================================================================= #
        # ================================================================= #
        
        # Now you can basically do whatever you want -- train a model, etc.
        # Save all results to f"{DIR}/abc" -- as that will keep all seeds separate.
        
        
        env = gym.make(yaml_conf("env/name"))
        if yaml_conf("method/name") == "SB3_PPO":
            model = PPO('MlpPolicy', env, verbose=1)
        else:
            raise Exception("Unsupported model", yaml_conf("method/name"))
        
        new_logger = configure(log_dir, ["stdout", "csv", "json", "tensorboard"])
        model.set_logger(new_logger)
        
        # This is specific to SB3
        callback = WandbCallback(gradient_save_freq=100, model_save_path=DIR, verbose=2)
        model.learn(total_timesteps=yaml_conf("train/steps"), callback=callback)
        episode_rewards, _ = evaluate_policy(model, env, n_eval_episodes = yaml_conf("eval/episodes"), deterministic=True, return_episode_rewards=True)
        
        df = pd.DataFrame({'Evaluation Rewards': episode_rewards})
        df.to_csv(path(DIR, 'eval_results.csv'))
        for r in episode_rewards:
            logger.log({
                "eval/rewards": r
            })
        # ================================================================= #
        # ================================================================= #
        # Finish the wandb run
        run.finish()
    
    # Run all things in parallel
    ray.get([single_run.remote(i) for i in range(NUM_SEEDS)])

if __name__ == '__main__':
    fire.Fire(main)