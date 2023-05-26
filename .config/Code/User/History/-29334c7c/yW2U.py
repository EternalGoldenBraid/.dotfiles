import hydra
from hydra.core.hydra_config import HydraConfig
import pandas as pd

import omegaconf
from omegaconf import DictConfig, open_dict
import torch
import torch.nn.functional as F
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
    
def evaluate(model, dataset, cfg, k=3, features=None, adj=None):
    features = dataset.features if features is None else features
    labels = dataset.targets
    mask = dataset.train_mask
    
    df = dataset.df.copy()
    
    idx_to_name = dataset.id_to_tag

    model.eval()
    with torch.no_grad():
        # model.to(features.device)
        features = features.to('cuda')
        if adj is None:
            out = model(features[mask])
        else:
            out = model(x=features, adj_t=adj)[mask]
        
        probs = F.sigmoid(out)
        
        # print(probs)

        # Get indices of entries over the threshold
        vals, indices = torch.topk(probs, k=k, dim=1)
        # print(vals)
        # print(indices)

        # Get the names of the tags
        mapped_names = [[None] * k for i in range(indices.shape[0])]
        for i in range(indices.shape[0]):
            # mapped_names.append([idx_to_name[idx.item()] for idx in indices[i]])
            mapped_names[i] = [idx_to_name[idx.item()] for idx in indices[i]]

        # print(mapped_names)
        # df.iloc[dataset.test_mask, 'predicted_tags'] = mapped_names
        # Create a new Series for the specified subset of rows
        # predicted_tags_series = pd.Series(index=df.index[dataset.test_mask], data=mapped_names, name='predicted_tags')
        # df = df.merge(predicted_tags_series, left_index=True, right_index=True, how='left')

        df = df.merge(pd.Series(index=df.index[mask], data=mapped_names, name='predicted_tags'),
                     left_index=True, right_index=True, how='left')
        
        print(df[['tag_name', 'predicted_tags', 'story']].loc[mask.numpy()])