"""
Dataloader for finetuning the model.
"""

from pathlib import Path
import os 
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer

class VaalikoneDataset(Dataset):
    "Test class for dataloader on Lappi questions"
    def __init__(self, path: Path, config: Dict, tokenizer=None):
        """
        Load the data from the given path and relabel the columns.
        Remove rows with missing values and empty strings.
        
        Args:
            path: Path to the csv file.
            tokenizer: Tokenizer to be used for preprocessing.
            config: Dictionary containing the configuration parameters.
            
        Returns:
            None
        """
        self.config = config
        
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
        self.column_names: Dict[str, str] = {}

        self.data: pd.DataFrame = pd.read_csv(path)

        # Read in only the Lappi questions
        lappi_column_names = [col for col in self.data.columns if col[:6] == 'Lappi.']
        
        # Rename the columns
        self.data = self.data[lappi_column_names]
        self.data.columns = [f"L_{i}" for i in range(1, len(self.data.columns) + 1)]
        self.data = self.data[self.data[self.data.columns[5:]] == "-"]
        

        # Remove rows with missing values
        self.data = self.data.dropna()

        # Remove rows with empty strings
        self.data = self.data[self.data.apply(lambda x: x.str.len().gt(0).all(), axis=1)]

        # Replace any entry of "-" by 0 in the first 5 columns
        self.data.iloc[:, :5] = self.data.iloc[:, :5].replace("-", 0)

        # Cast first 5 columns to int
        self.data.iloc[:, :5] = self.data.iloc[:, :5].astype(int)

        # Store column name and new name in a dictionary
        self.column_names = {new_name: old_name for old_name, new_name in zip(lappi_column_names, self.data.columns)}
        
        print(f"Loaded data from {path} with shape, {self.data.shape}, and columns: {self.data.columns}")
        

        
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        """
        This is specific to lappi questions since we have 5 questions per row.
        We need to tokenize each question and then concatenate the tokens.
        We also need to create the attention mask and token type ids.
        
        Args:
            idx: Index of the row to be returned.
            
        Returns:
            sample: Dictionary containing the tokens, attention mask, token type ids and labels.
        """
        
        row = self.data.iloc[idx]
        
        # Onehot encode labels of size config["num_labels"]
        # labels = F.one_hot([int(row[f"L_{i}"]), num_classes=self.config["num_labels"]) for i in range(1, 6)]
        print(row)
        labels = torch.LongTensor([int(row[f"L_{i}"]) for i in range(1, 6)]) - 1
        print(labels)
        labels = F.one_hot(labels, num_classes=self.config["num_labels"])

        comments = [row[f"L_{i}"] for i in range(6, 11)]
        
        # Tokenize the comments
        tokenized_comments = self.tokenizer(comments,
                                            add_special_tokens=True,
                                            return_tensors='pt', 
                                            padding='max_length',
                                            truncation=True,
                                            max_length=self.config["max_length"])
        
        sample = {"input_ids": tokenized_comments["input_ids"],
                "attention_mask": tokenized_comments["attention_mask"],
                "token_type_ids": tokenized_comments["token_type_ids"],
                "labels": labels}
        
        return sample

class VaalikoneDataModule(pl.LightningDataModule):

    def __init__(self, train_path: Path, val_path: Path, test_path: Path, config: Dict, tokenizer=None):
        super().__init__()
        self.train_path: Path = train_path
        self.val_path: Path = val_path
        self.test_path: Path = test_path
        self.tokenizer = tokenizer

        self.config: Dict = config
        

    def setup(self, stage=None):
        

        if stage in (None, 'fit'):
            self.train_dataset = VaalikoneDataset(self.train_path, self.tokenizer, self.config)
            self.val_dataset = VaalikoneDataset(self.val_path, self.tokenizer, self.config)
        else:
            self.test_dataset = VaalikoneDataset(self.test_path, self.tokenizer, self.config)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config["batch_size"],
                            num_workers=self.config["num_workers"], shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config["batch_size"],
                            num_workers=self.config["num_workers"], shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config["batch_size"],
                            num_workers=self.config["num_workers"], shuffle=False)

if __name__ == "__main__":

    # Load both turku and roberta models. Turku seems more rigorous and documented.
    from transformers import RobertaTokenizer, RobertaModel, AutoModel, AutoTokenizer
    model_id: str = 'TurkuNLP/bert-large-finnish-cased-v1'
    model = AutoModel.from_pretrained(model_id)
    
    # Load data
    dl = VaalikoneDataset(
        path=Path("data/vaalit_2019.csv"),
        config=config)

    dl.__getitem__(1)