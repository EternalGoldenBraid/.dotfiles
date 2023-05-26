import os
from pathlib import Path
import torch

import wandb
import hydra
import omegaconf
from omegaconf import DictConfig

from src.slaps.train_slaps_frozen import Experiment as SlapsFrozenExperiment

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
    
    # if True:
    # if cfg.model.gnns.model_type == "slaps":
    if cfg.model == "slaps":
        experiment = SlapsFrozenExperiment() 
        # if cfg.model.slaps.model == "end2end":
        if cfg.model.model == "end2end":
            experiment.train_end_to_end(cfg)
        # elif cfg.model.slaps.model == "end2end_mlp":
        elif cfg.model.model == "end2end_mlp":
            experiment.train_end_to_end_mlp(cfg)
    else:
        raise NotImplementedError(f"Model type {cfg.model.gnns.model_type} not implemented")

if __name__ == '__main__':
    main()