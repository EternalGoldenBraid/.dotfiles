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
    k = cfg.data.dataset.min_tag_occurrences
    tag_counts = df['tag_id'].explode().value_counts()
    df_ = df[df['tag_id'].apply(lambda x: len(set(x) & set(tag_counts[tag_counts >= k].index)) > 0)]

    # loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)
    # for batch in loader:
    #     print(batch)
    
    features = dataset[torch.arange(len(dataset))].x    
    targets = dataset[torch.arange(len(dataset))].y
    targets = targets.reshape(len(dataset), -1)

    assert len(features) == len(targets)
    assert (dataset[0].y == targets[0]).all()
    
    visualize_dataset(dataset=dataset, df=dataset.df, cfg=cfg, data_root=Path(data_root))

    if cfg.data.dataset.shuffle == True:
        idxs = torch.randperm(len(dataset))
        features = features[idxs]
        targets = targets[idxs]
    elif cfg.data.shuffle == False:
        pass
    else:
        raise ValueError(f"cfg.data.shuffle should be True or False, but got {cfg.data.shuffle}")


    train_mask = torch.zeros(len(dataset), dtype=torch.bool)
    train_mask[torch.arange(int(len(dataset) * cfg.data.dataset.train_ratio))] = True
    val_mask = torch.zeros(len(dataset), dtype=torch.bool)
    val_mask[torch.arange(int(len(dataset) * cfg.data.dataset.train_ratio),
            int(len(dataset) * (cfg.data.dataset.train_ratio + cfg.data.dataset.val_ratio)))] = True

    test_mask = torch.zeros(len(dataset), dtype=torch.bool)
    test_mask[torch.arange(int(len(dataset) * (cfg.data.dataset.train_ratio + cfg.data.dataset.val_ratio)), len(dataset))] = True

    return dataset, features, targets, train_mask, val_mask, test_mask



    
if __name__ == "__main__":
    create_slaps_dataset()