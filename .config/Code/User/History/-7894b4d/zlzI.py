import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig

import omegaconf
from omegaconf import DictConfig, open_dict

import torch
import random
from torch.utils.data import Sampler

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
    

def aggregate_statistics(
                        participant_features: torch.tensor,
                        features: torch.tensor,
                        participant_ids: torch.tensor,
                        participant_idx_to_id: torch.tensor,
                        aggregation=['mean']) -> torch.tensor:
    
    """
        Aggregate the features of the participants in the batch.

        Args:
            participant_features: Container for the resulting features.
            features: The features of the participants in the batch.
            participant_idxs: The indices of the participants in the batch.
    """
    
    n_participants = participant_ids.shape[0]
    feature_dim = features.shape[1]
    participant_features = torch.empty((n_participants, feature_dim), device=features.device)


    # Iterate over the participants in the batch
    for i in range(participant_ids.shape[0]):
        participant_id = participant_ids[i]
        participant_mask = participant_idx_to_id == participant_id

        if 'mean' in aggregation:
            participant_features[i] = features[i].clone()
            # participant_features[i] = features[participant_mask].mean(dim=0)
            
        # print("participant_features count:", participant_mask.sum())
        
    return participant_features

# def aggregate_from_disk(path: Path, user_id: int) -> torch.tensor:
def aggregate_from_disk(args) -> torch.tensor:
    """
        Aggregate the features of the participants in the batch.

    """
    user_id, path = args
    user_dir = path/f"user_{user_id}"
    feature_files = os.listdir(user_dir)
    user_features = [torch.load(user_dir/file) for file in feature_files]

    # Aggregate features (e.g., average, max, etc.)
    user_representation = torch.stack(user_features).mean(dim=0)

    return user_representation

def save_user_feature(args):
    user_id, feature, bert_outputs_dir = args
    id_dir = bert_outputs_dir / f"user_{user_id}"
    os.makedirs(id_dir, exist_ok=True)
    file_count = len(os.listdir(id_dir))
    torch.save(feature, id_dir / f"feature_{file_count}.pt")
    

class CustomSampler(Sampler):
    def __init__(self, users_data, bert_input):
        self.users_data = users_data
        self.bert_input = bert_input

    def __iter__(self):
        # Shuffle the order of texts for each user
        for user_texts in self.users_data.values():
            random.shuffle(user_texts)

        # Concatenate, split, and flatten texts
        all_chunks = []
        for user_texts in self.users_data.values():
            long_text = torch.cat(user_texts)
            chunks = self.split_into_chunks(long_text, self.bert_input)
            all_chunks.extend(chunks)

        # Shuffle chunks
        random.shuffle(all_chunks)

        return iter(all_chunks)

    def __len__(self):
        total_length = sum(len(texts) for texts in self.users_data.values())
        return total_length

    @staticmethod
    def split_into_chunks(long_text, max_length):
        chunks = []
        for i in range(0, len(long_text), max_length):
            chunk = long_text[i:i+max_length]
            chunks.append(chunk)
        return chunks