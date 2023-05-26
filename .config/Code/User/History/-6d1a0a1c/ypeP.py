from pathlib import Path
import torch

import wandb
import hydra
import omegaconf
from omegaconf import DictConfig

from src.slaps.train_slaps_frozen import Experiment as SlapsFrozenExperiment

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):

    experiment = SlapsFrozenExperiment() 
    
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    os.environ["WANDB_MODE"] = cfg.wandb.wandb_mode
    
    run = wandb.init(
        entity=config.wandb.entity, 
        project=config.wandb.project_name,
        config = wandb.config,
        )
    
    # This script is for  frozen bert
    config.bert.freeze = 'full'

    if config.slaps.model == "end2end":
        experiment.train_end_to_end(config)
    elif config.slaps.model == "end2end_mlp":
        experiment.train_end_to_end_mlp(config)

if __name__ == '__main__':
    main()