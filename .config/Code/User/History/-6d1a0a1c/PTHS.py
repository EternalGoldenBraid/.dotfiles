import os
from pathlib import Path
import torch

import wandb
import hydra
import omegaconf
from omegaconf import DictConfig

from src.slaps.train_slaps_frozen import Experiment as SlapsFrozenExperiment
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
    
    # if True:
    # if cfg.model.gnns.model_type == "slaps":
    if cfg.model_name == "slaps":
        experiment = SlapsFrozenExperiment() 
        experiment.train_end_to_end(cfg)

        # if cfg.model.slaps.model == "end2end":
        # if cfg.model.model == "end2end":
            # experiment.train_end_to_end(cfg)
        # elif cfg.model.slaps.model == "end2end_mlp":
            # experiment.train_end_to_end_mlp(cfg)

    elif cfg.model_name == "mlp":
        experiment = SlapsFrozenExperiment() 
        model, dataset = experiment.train_end_to_end_mlp(cfg)
        
        model.load_state_dict(torch.load(experiment.model_save_path))
        
        evaluate(model, dataset, cfg)
        
        
    else:
        raise NotImplementedError(f"Model type {cfg.model_name} not implemented")

if __name__ == '__main__':
    main()