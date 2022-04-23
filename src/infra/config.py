import hashlib
import json
import os
from typing import Any, Dict
import wandb
import yaml
from common.utils import get_md5sum_file, path
from common.vars import CONFIG_DIR, RESULTS_DIR


class YamlConfig:
    # https://jonnyjxn.medium.com/how-to-config-your-machine-learning-experiments-without-the-headaches-bb379de1b957
    """Simple dict wrapper that adds a thin API allowing for slash-based retrieval of
    nested elements, e.g. cfg.get_config("meta/dataset_name")
    """

    def __init__(self, config_path):
        self.filename = config_path
        with open(config_path) as cf_file:
            self._data = yaml.safe_load(cf_file.read())

    def get(self, path=None, default=None):
        # we need to deep-copy self._data to avoid over-writing its data
        recursive_dict = dict(self._data)

        if path is None:
            return recursive_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                recursive_dict = recursive_dict.get(path_item)

            value = recursive_dict.get(data_item, default)

            return value
        except (TypeError, AttributeError):
            return default

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.get(*args, **kwargs)


class Config:
    """
        This is a config of a single run
    """

    def __init__(self, yaml_config: YamlConfig, seed: int, date: str) -> None:
        """A simple config class for each **run**. Effectively, this consists of a yaml_config that is shared across all runs, and a seed unique to this run.

        Args:
            yaml_config (YamlConfig): This is the config from the yaml file
            seed (int): The seed associated with this run.
            date (str): The date of this run.
        """
        self.date = date
        self.seed = seed
        self.yaml_config = yaml_config        

        self.method = yaml_config('method/name')
        self.method_parameters = yaml_config('method')
        self.environment_name = yaml_config('env/name')

        self.project_name = yaml_config('meta/project_name')
        self.experiment_name = yaml_config('meta/experiment_name')
        self.message = yaml_config('meta/message')
        self.run_name = yaml_config('meta/run_name', f'{self.experiment_name}-{self.method}-{self.environment_name}')

        self.results_directory = yaml_config('meta/results_directory', f"{RESULTS_DIR}/{self.experiment_name}/{self.run_name}/{self.date}-{self.hash(False)}/{seed}")


    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary that can be sent to wandb that contains all the important parameters.

        Returns:
            Dict[str, Any]: [description]
        """
        return {
            'run_name': self.run_name,
            'method': self.method,
            'seed': self.seed,
            'date': self.date,
            'results_directory': self.results_directory,
            'experiment_name': self.experiment_name,
            'project_name': self.project_name,
            'method_parameters': self.method_parameters,
            'environment_name': self.environment_name,
            'message': self.message,
            'yaml': self.yaml_config._data
        }
        
    def hash(self, seed=False, date=True) -> str:
        """Hashes this config parameters, returning a unique identifier.

        Args:
            seed (bool, optional): Whether or not to include the seed to the hash. If yes, different seeds of the same experiment will have different hashes. Defaults to False.
            date (bool, optional): Whether or not to include the date in the hash. Defaults to True.

        Returns:
            str: The hash, with the run-name prepended.
        """
        hash = get_md5sum_file(self.yaml_config.filename)
        if seed:
            hash += "-" + str(self.seed)
        if date:
            hash += "-" + str(self.date)
        return f"{self.run_name}-" + hash

    def pretty_name(self) -> str:
        return self.hash()

    def init_wandb(self, **kwargs) -> wandb.run:
        """
            Initialises wandb and returns the run object.
        """
        dic = dict(project=self.project_name,
                   name=self.unique_name(),
                   config=self.to_dict(),
                   job_type=self.experiment_name,
                   tags=[self.environment_name],
                   group=self.hash(), save_code=True,
                   **kwargs)
        return wandb.init(**dic)

    def unique_name(self):
        return self.hash(seed=True, date=True)
