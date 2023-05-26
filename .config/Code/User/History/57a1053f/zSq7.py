import os
from pathlib import Path
import torch

import wandb
import hydra
import omegaconf
from omegaconf import DictConfig

from src.utils.utils import evaluate

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):

    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    os.environ["WANDB_MODE"] = cfg.wandb.wandb_mode
    run = wandb.init(
        entity=cfg.wandb.entity, 
        project=cfg.wandb.project_name,
        config = wandb.config,
        )
    
    

    # evaluate()
    
if __name__ == '__main__':
    main()