import os
from abc import ABC, abstractmethod
from pathlib import Path

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
import pandas as pd

from tqdm import tqdm

from src.utils.vis_utils import visualize_dataset
from src.utils.utils import is_truncated_precise

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class MLMEventFeatureDataset(Dataset):
    def __init__(self, root, config, filename, tokenizer: PreTrainedTokenizer):
        self.root = root
        self.config = config
        self.filename = filename
        self.tokenizer = tokenizer

        self.df = pd.read_hdf(os.path.join(root, filename), key=config.data.dataset.key)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        story = self.df.iloc[index]['story']
        input_ids = self.tokenizer.encode(story, add_special_tokens=True, return_tensors='pt').squeeze()

        # Create a mask for randomly selecting tokens to mask
        mask = torch.rand(input_ids.size(0)) < self.config.data.mask_probability

        # Replace the masked tokens with the mask token (e.g., 103 for BERT)
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask] = self.config.data.mask_token_id

        return masked_input_ids, input_ids

def create_mlm_dataset(cfg: DictConfig, tokenizer: PreTrainedTokenizer):
    data_root = hydra.utils.to_absolute_path(cfg.data.dataset.root)

    dataset = MLMEventFeatureDataset(
        root=data_root,
        filename=cfg.data.dataset.filename, config=cfg,
        tokenizer=tokenizer
    )
    print(f"Dataset length: {len(dataset)}")

    train_size = int(cfg.data.train_ratio * len(dataset))
    val_size = int(cfg.data.val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.data.mask_probability
    )

    return train_dataset, val_dataset, test_dataset, data_collator

