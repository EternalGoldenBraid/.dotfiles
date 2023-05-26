import hydra
from hydra.core.hydra_config import HydraConfig

import omegaconf
from omegaconf import DictConfig, open_dict
import torch
import wandb

def initialize(config: DictConfig):
    if config.wandb.use_wandb:
        # Set config for the training
        wandb.config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )   # track hyperparameters and run metadata

        # Set other parameters
        wandb.init(
            project=config.wandb.project_name,  # Project name
        )

    if config.finetune.use_tensor_cores:
        torch.set_float32_matmul_precision("medium")
        
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = torch.inf

    def early_stop(self, validation_loss):
        # print("counter:",self.counter, validation_loss, self.min_validation_loss, self.min_validation_loss + self.min_delta)
        if validation_loss != torch.inf:
            pass
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

def aggregate_statistics(participant_features: torch.tensor, features: torch.tensor,
                         participant_ids: torch.tensor, participant_idx_to_id: torch.tensor,
                         aggregation=['mean']) -> torch.tensor:
    
    """
        Aggregate the features of the participants in the batch.

        Args:
            participant_features: Container for the resulting features.
            features: The features of the participants in the batch.
            participant_idxs: The indices of the participants in the batch.
    """

    # Iterate over the participants in the batch
    for i in range(participant_ids.shape[0]):
        participant_id = participant_ids[i]
        participant_mask = participant_idx_to_id == participant_id

        if 'mean' in aggregation:
            participant_features[i] = features[i].clone()
            # participant_features[i] = features[participant_mask].mean(dim=0)
            
        # print("participant_features count:", participant_mask.sum())