import os
from typing import Any, Dict
import wandb
from common.utils import save_compressed_pickle
from common.types import Verbosity
from infra.config import Config


class Logger:
    """
        A basic logger that acts as the interface to log results
    """
    def __init__(self, config: Config, verbose: Verbosity = Verbosity.NONE) -> None:
        self.config = config
        self.verbose = verbose
    
    def log(self, dic: Dict[Any, Any], **kwargs):
        """Logs this dictionary

        Args:
            dic (Dict[Any, Any]): 
        """
        pass

    def end(self, name: str):
        pass


class WandBLogger(Logger):
    """Logs everything to weights and biases (https://wandb.ai/)
    """
    def __init__(self, config: Config, verbose: Verbosity = Verbosity.NONE) -> None:
        super().__init__(config, verbose)
        wandb.init(
            project=config.project_name,
            notes = config.run_name + ": " + config.messsage,
            tags = [config.environment_name, config.method, config.experiment_name],
            config=config.to_dict(),
            group=config.hash(seed=False),
            job_type="run"
        )
        
    def log(self, dic: Dict[Any, Any], **kwargs):
        wandb.log(dic, **kwargs)
    
    def end(self, name: str):
        return super().end(name)

class WandDBRunLogger(Logger):
    """Logs everything to weights and biases (https://wandb.ai/). This uses a `run` object instead of logging globally.
    """
    def __init__(self, config: Config, run: wandb.run , verbose: Verbosity = Verbosity.NONE) -> None:
        super().__init__(config, verbose)
        self.run = run
        
    def log(self, dic: Dict[Any, Any], **kwargs):
        self.run.log(dic, **kwargs)
    
    def end(self, name: str):
        return super().end(name)


class NoneLogger(Logger):
    """
        A logger that is basically a no-op. Kind of null object pattern
    """
    def __init__(self, seed: int = 0) -> None:
        super().__init__(None, verbose=Verbosity.NONE)


class PrintLogger(Logger):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(None, verbose=Verbosity.DETAILED)
    def log(self, dic: Dict[Any, Any], **kwargs):
        print(dic)
        