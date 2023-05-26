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
from torch.nn.functional import one_hot
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

class VaalikoneDataset(Dataset):
    "Dataset on global questions"
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
        self.data_conf = config.dataset
        self.training_conf = config.finetune
        
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.training_conf['model_id'])
        self.column_names: Dict[str, str] = {}

        self.data: pd.DataFrame = pd.read_csv(path, index_col=0)
        if (self.data.columns == ["question_id", "question", "label", "comment"]).all():
            # Add participant_id column from index
            self.data["participant_id"] = self.data.index
            self.data = self.data[["participant_id", "question_id", "question", "comment", "label"]]
        else:
            # Expecting the participant_id to be the first column
            raise ValueError("The columns of the data are not correct.")
        
        # TODO Incorporate onehot encoding of numeric answers to questions.
        # self.data = self.data[self.data["label"].apply(lambda x: str(x).isnumeric())]
        # one_hot_labels = one_hot(torch.tensor(self.data["label"].apply(lambda x: int(x)).values-1), num_classes=config["num_classes"])
        # self.data["label"] = one_hot_labels.tolist()
        self.data = self.data.drop(columns=["label"])

        self.parties = pd.read_csv("data/parties.csv", index_col=0, encoding_errors="ignore", header=None)
        self.parties.columns = ["party"]

        print(f"Loaded data from {path} with shape, {self.data.shape}, and columns: {self.data.columns}")
        
        # One hot encode parties using torch one_hot as label
        self.party_names = self.parties["party"].unique()
        self.party_id = {party: i for i, party in enumerate(self.party_names)}
        self.parties["party_id"] = self.parties["party"].apply(lambda x: self.party_id[x])
        self.parties["label"] = one_hot(torch.tensor(self.parties["party_id"].values), num_classes=self.training_conf["num_classes"]).tolist()
        
        # Merge the data and the parties by participant_id and party index
        self.data = self.data.merge(self.parties, left_on="participant_id", right_index=True)

        self.participants = self.data["participant_id"].unique()
        self.n_participants = len(self.participants)
        
    def __len__(self):
        return len(self.participants)
    
    # def encode()

    def __getitem__(self, idx):
        """
        Get the question and comment from the given index.
        Tokenize the question and comment and concatenate them.
        Create the attention mask and token type ids.

        Args:
            idx: Index of the row to be retrieved.

        Returns:
            Dictionary containing the input_ids, attention_mask and token_type_ids.

        TODO: Add a split token between question and comment.
        TODO 2: Add a special token for the question and comment?
        TODO 3: Add a token or learn a token for region.
        """
        #print(f'idx: {idx}')
        participant = self.participants[idx]
        #print(f'part: {participant}')

        participant_data = self.data[self.data["participant_id"] == participant]
        #print(participant_data)

        sample: Dict = {
            "participant_data": participant_data.to_dict("records")[0]
        }
        for row_idx in range(participant_data.shape[0]):

            row = participant_data.iloc[row_idx]
            question = row.question
            comment = row.comment
            question_id = row.question_id

            assert self.tokenizer != True
            # Tokenize question and comment
            tokenized_question = self.tokenizer(question,
                                                add_special_tokens=True,
                                                return_tensors='pt',
                                                padding='max_length',
                                                truncation=True,
                                                max_length=self.training_conf["max_length"])

            # Tokenize the comments
            tokenized_comment = self.tokenizer(comment,
                                                add_special_tokens=True,
                                                return_tensors='pt', 
                                                padding='max_length',
                                                truncation=True,
                                                max_length=self.training_conf["max_length"])

            # Concatenate the tokens and create the attention mask and token type ids
            input_ids = torch.cat((tokenized_question["input_ids"], tokenized_comment["input_ids"]), dim=1)
            attention_mask = torch.cat((tokenized_question["attention_mask"], tokenized_comment["attention_mask"]), dim=1)

            sample[question_id] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": tokenized_comment["token_type_ids"],
                    "labels": torch.Tensor(row["label"])
                    }

        #print(sample)
        #print()
        return sample

class VaalikoneDataModule(pl.LightningDataModule):

    def __init__(self, train_path: Path, val_path: Path, test_path: Path, config: Dict, tokenizer=None):
        super().__init__()
        self.train_path: Path = train_path
        self.val_path: Path = val_path
        self.test_path: Path = test_path
        self.tokenizer = tokenizer

        self.config: Dict = config
        self.data_conf = config.dataset
        self.training_conf = config.finetune


    def setup(self, stage=None):

        if stage in (None, 'fit'):
            self.train_dataset = VaalikoneDataset(path=self.train_path, 
                                            tokenizer=self.tokenizer, config=self.config)
            self.val_dataset = VaalikoneDataset(path=self.val_path, 
                                            tokenizer=self.tokenizer, config=self.config)
        else:
            self.test_dataset = VaalikoneDataset(path=self.test_path, tokenizer=self.tokenizer, config=self.config)
            
    def train_dataloader(self):

        dl = DataLoader(self.train_dataset, batch_size=self.training_conf["batch_size"],
                            num_workers=self.training_conf["num_workers"], shuffle=True)
        # dl.set_transform(dl.encode)
        
        return dl
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.training_conf["batch_size"],
                            num_workers=self.training_conf["num_workers"], shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.training_conf["batch_size"],
                            num_workers=self.training_conf["num_workers"], shuffle=False)


    # Load both turku and roberta models. Turku seems more rigorous and documented.

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):

    # Load data
    ds = VaalikoneDataset(
        path=Path("data/data_all_answers.csv"),
        config=config)

    for i in range(ds.n_participants):
        sample = ds.__getitem__(i)
    
    print(sample.keys())

if __name__ == "__main__":
    main()