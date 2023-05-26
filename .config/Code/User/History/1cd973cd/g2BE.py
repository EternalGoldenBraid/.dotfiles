from pathlib import Path

import torch
import pandas as pd

import hydra
from omegaconf import DictConfig

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from src.data.daily_dataset import EventFeatureDataset, BertEncoder
from src.utils.vis_utils import visualize_dataset

# @hydra.main(config_path="conf", config_name="config")
def create_slaps_dataset(cfg: DictConfig):
    # dataset = EventFeatureDataset(root='data', filename='data_dump.h5')
    
    # data_root = Path(hydra.utils.get_original_cwd(), cfg.data.dataset.root)
    data_root = hydra.utils.to_absolute_path(cfg.data.dataset.root)
    
    transform = T.Compose([
        # BertEncoder(cfg)
        T.NormalizeFeatures('x')
        ])
    
    dataset = EventFeatureDataset(
                root=str(data_root),
                # processed_dir=cfg.data.dataset.processed_dir,
                # raw_dir=cfg.data.dataset.raw_dir,
                filename=cfg.data.dataset.filename, config=cfg,
                transform=transform
            )
    print(f"Dataset length: {len(dataset)}")

    df = dataset.df

    # loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)
    # for batch in loader:
    #     print(batch)
    
    return (dataset, dataset.features, dataset.targets, 
            dataset.train_mask, dataset.val_mask, dataset.test_mask)



    
if __name__ == "__main__":
    create_slaps_dataset()