"""
Dataloader for finetuning the model.
Batchmode refers to the fact that we are 
batching all the questions of a participant together in __get__item__.
TODO This is not the most efficient way to do this. Quick easy hack
"""

party_colors = {
    "Vasemmistoliitto": (50, 255, 50),
    "Vihreät": (50, 255, 50),
    "Keskusta": (200, 255, 200),
    "SDP": (200, 255, 200),
    "Perussuomalaiset": (200, 0, 200),
    "Kokoomus": (0, 0, 200),
    "Kristillisdemokraatit": (200, 200, 0),
    "Seitsemän tähden liike": (255, 255, 255),
    "Sininen tulevaisuus": (0, 0, 100),
    "Liike Nyt": (255, 200, 0),
    "RKP": (255, 200, 0),
    "Piraattipuolue": (100, 100, 100),
    "Suomen Kommunistinen Puolue": (255, 0, 0),
    "Kansalaispuolue": (200, 200, 200),
    "Itsenäisyyspuolue": (255, 100, 0),
    "Feministinen puolue": (255, 0, 255),
    "Liberaalipuolue": (0, 200, 200),
    "Suomen Kansa Ensin": (200, 0, 0),
    "Kommunistinen Työväenpuolue": (150, 0, 0),
    "Eläinoikeuspuolue": (0, 255, 255),
    "Sitoutumaton": (100, 100, 100),
    "Kansanliike Suomen Puolesta": (200, 200, 200)
}

from pathlib import Path
import os 
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.functional import one_hot
import torch
# import pytorch_lightning as pl
from transformers import AutoTokenizer

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from tqdm import tqdm


class VaalikoneDataset(Dataset):
    "Dataset on global questions"
    def __init__(self, config: Dict, split_data: pd.DataFrame=None, tokenizer=None, debug=False):

        """
        Load the data from the given path and relabel the columns.
        Remove rows with missing values and empty strings.
        
        NOTE: Relies heavily on the fact that self.data has row_ids of interest.
        
        Args:
            path: Path to the csv file.
            tokenizer: Tokenizer to be used for preprocessing.
            config: Dictionary containing the configuration parameters.
            
        Returns:
            None
            
        Attributes:
            TODO Add attributes
        """
        self.party_counts = None
        self.participants = None
        self.party_names = None
        self.n_nodes = None
        self.n_participants = None
        self.party_colors = party_colors

        self.config = config
        self.data_conf = config.dataset
        self.training_conf = config.finetune

        path = self.data_conf.data_path
        
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.slaps['model_id'])
        self.column_names: Dict[str, str] = {}

        self.data: pd.DataFrame = pd.read_csv(path, index_col=None) if split_data is None else split_data

        if config.debug is True:
            # sample ratio
            # ratio = 0.005
            ratio = 0.1
            # ratio = 0.01
            # ratio = 1.00
            pre_size = self.data.shape[0]
            self.data = self.data.sample(frac=ratio, random_state=config.seed)
            print("DEBUG: Sampling {} of {} rows".format(self.data.shape[0], pre_size))

        # Add participant_id column from index
        self.data = self.data.rename(columns={'candidate_id': 'participant_id'})
        self.data = self.data[["participant_id", "question_id", "question", "comment", "label"]]
                   
        # Drop label corresponding to the numerical answer.
        self.data = self.data.drop(columns=["label"])

        self.parties = pd.read_csv("data/parties.csv", index_col=0, encoding_errors="ignore", header=None)
        self.parties.columns = ["party"]

        # Retain data for top K parties
        party_sizes = self.parties["party"].value_counts()
        if self.config.top_k_parties != 0:
            top_k_parties = party_sizes.index[:self.config.top_k_parties]
            self.parties = self.parties[self.parties["party"].isin(top_k_parties)]
        else:
            top_k_parties = party_sizes.index

        # Sort by party size
        self.parties['party_size'] = self.parties['party'].apply(lambda x: party_sizes[x]) 
        self.parties = self.parties.sort_values(by=['party_size', 'party'], ascending=False)
            
        print(f"Loaded data from {path} with shape, {self.data.shape}, and columns: {self.data.columns}")
        
        # One hot encode parties using torch one_hot as label
        self.party_names = self.parties["party"].unique()
        self.party_id = {party: i for i, party in enumerate(self.party_names)}
        self.parties["party_id"] = self.parties["party"].apply(lambda x: self.party_id[x])
        self.parties["label"] = one_hot(torch.tensor(self.parties["party_id"].values),
                                num_classes=self.parties["party_id"].nunique()).tolist()
        
        # Merge the data and the parties by participant_id and party index
        # Retain sort order of the parties based on party size
        how='inner'
        assert self.data.index.isna().sum() == self.parties.index.isna().sum() == 0
        assert self.data['participant_id'].dtype == self.parties.index.dtype
        self.data = self.parties.merge(self.data, left_index=True, right_on="participant_id",
                                    indicator='one-to-many', how=how, sort=False)  
        assert self.data.index.dtype == "int64"
        assert len(self.data['participant_id'].unique()) == min(len(self.data['participant_id'].unique()), len(self.parties.index.unique()))

        # Preserve order of the parties based on party size.
        assert (self.data['party'].unique() == self.party_names).sum() == len(self.party_names)
            
        ### This should be unnecessary if above assertion is correct
        # self.data_ = self.data.sort_values(by=['party_size', 'party_id'], ascending=False)
        # print(self.data_.head())
        # print(self.data.head())

        # Store questions per participant
        self.questions_per_participant = self.data.groupby("participant_id").count()["question_id"]
        
        self.participants = pd.DataFrame(self.data["participant_id"].unique(),
                                            columns=["participant_id"])
        self.participants.set_index(self.participants['participant_id'], inplace=True)
        
        # Add party id to self.participants
        self.participants = self.participants.merge(self.parties, how='inner',
                    left_index=True, right_index=True, sort=False)
        
        self.party_counts = self.participants[['party','party_id']].value_counts()

        self.n_participants = len(self.participants)
        self.n_nodes = len(self.participants)
        
        # If this is not true we did not find a participant_id for each party
        assert len(self.participants['participant_id']) == len(self.data['participant_id'].unique())

        self.party_label = torch.tensor(self.participants["party_id"].values.tolist())
        
        # Compute the maximum number of questions 
        self.max_questions = self.data.groupby("participant_id").count().max()[0]
        
        
    def __len__(self):
        # return self.n_participants
        return len(self.data)
    
        
        
    def tokenize_participant_data_(self, participant_data: pd.DataFrame) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Tokenize the participant data and return the input_ids, attention_masks and token_type_ids.
        input_ids, attention_masks, token_type_ids are of size (max_questions, max_token_length)
        mask is of size (max_questions, 1) and contains True if the question is not empty and False otherwise.

        Args:
            participant_data: Dataframe containing the participant data.
        Returns:
            Dict containing the tokenized data. 
            Keys: "input_ids", "attention_masks", "token_type_ids", 
            "participant_id", "party_id", "party_label", "mask"
        """
    
        input_ids = torch.zeros(
            (self.max_questions, 2*self.config.bert["max_token_length"]), dtype=torch.int32)
        attention_masks = torch.zeros(
            (self.max_questions, 2*self.config.bert["max_token_length"]), dtype=torch.int32)
        token_type_ids = torch.zeros(
            (self.max_questions, 2*self.config.bert["max_token_length"]), dtype=torch.int32)

        participant_id = participant_data["participant_id"].values[0]
        sample: Dict = {}
        questions = participant_data['question'].values.tolist()
        comments = participant_data['comment'].values.tolist()
        question_ids = participant_data['question_id'].values
        
        assert self.tokenizer != True
        # Tokenize question and comment
        tokenized_questions = self.tokenizer(questions,
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            padding='max_length',
                                           truncation=True,
                                            max_length=self.config.bert["max_token_length"])

        # Tokenize the comments
        tokenized_comments = self.tokenizer(comments,
                                            add_special_tokens=True,
                                            return_tensors='pt', 
                                            padding='max_length',
                                            truncation=True,
                                            max_length=self.config.bert["max_token_length"])
        
        # Concatenate the tokens and create the attention mask and token type ids
        input_ids_ = torch.cat((tokenized_questions["input_ids"], 
                                tokenized_comments["input_ids"]), dim=1)
        attention_masks_ = torch.cat((tokenized_questions["attention_mask"],
                                tokenized_comments["attention_mask"]), dim=1)
        token_type_ids_ = torch.cat((tokenized_questions["token_type_ids"],
                                tokenized_comments["token_type_ids"]), dim=1)
        
        input_ids[:len(input_ids_)] = input_ids_
        attention_masks[:len(attention_masks_)] = attention_masks_
        token_type_ids[:len(token_type_ids_)] = token_type_ids_
        mask = torch.zeros(self.max_questions)
        n_questions = input_ids_.shape[0]
        mask[:n_questions] = 1
        mask = mask.bool()
        
        assert input_ids.dtype == torch.int
        assert attention_masks.dtype == torch.int
        assert token_type_ids.dtype == torch.int
        assert mask.dtype == torch.bool
        
        # sample = {
        #         "input_ids": input_ids,
        #         "attention_mask": attention_masks,
        #         "token_type_ids": token_type_ids,
        #         "labels": torch.Tensor(participant_data["label"].values.tolist()),
        #         "participant_id": participant_id,
        #         "n_questions": n_questions,
        #         "mask": mask,
        #         "party_id": participant_data["party_id"].values[0],
        #         }

        sample = (
                input_ids,
                attention_masks,
                token_type_ids,
                torch.Tensor(participant_data["label"].values.tolist()),
                participant_id,
                n_questions,
                mask,
                participant_data["party_id"].values[0],
                )
    
        return sampleo
    
    # def create_users_data(self):
    #     users_data = {}
    #     for _, row in self.data.iterrows():
    #         participant_id = row["participant_id"]
    #         input_ids, attention_mask, _, _, _ = self.__getitem__(row.name)
    #         text_tensor = torch.cat([input_ids, attention_mask], dim=-1)
    #         if participant_id not in users_data:
    #             users_data[participant_id] = []
    #         users_data[participant_id].append(text_tensor)
    #     return users_data
    
    def create_users_data(self):

        def concat_tensors(group):
            tensors = []
            for idx in group.index:
                input_ids, attention_mask, _, _, _ = self.__getitem__(idx)
                text_tensor = torch.cat([input_ids, attention_mask], dim=-1)
                tensors.append(text_tensor)
            return tensors

        grouped_data = self.data.groupby("participant_id")
        users_data = {participant_id: concat_tensors(group) for participant_id, group in tqdm(grouped_data, desc="Creating users data")}

        return users_data

    def __getitem__(self, idx):


        data = self.data.loc[idx]
        comment = data['comment']
        question = data['question']
        # tokens = self.tokenizer.encode(question, comment,
        tokens = self.tokenizer(question, comment,
                                            add_special_tokens=True, 
                                            return_tensors='pt',
                                            padding='max_length',
                                            truncation=True,
                                            max_length=self.config.bert["max_token_length"],
        )
        
        input_ids = tokens.input_ids.squeeze()
        attention_mask = tokens.attention_mask.squeeze()

        return input_ids, attention_mask, data.party_id, data.participant_id, idx


    def get_idx_split(self, stratified: bool = False):
        """
        Get the indices of the train, validation and test sets.
        """
        
        # Train-test split
        
        train_size = self.config["train_size"]
        val_size = self.config["val_size"]
        test_size = self.config["test_size"]

        if self.config.shuffle:
            idxs = torch.randperm(self.n_participants)
        else:
            idxs = torch.arange(self.n_participants)
        train_idxs = idxs[:int(train_size*self.n_participants)]
        val_idxs = idxs[int(train_size*self.n_participants):int((train_size+val_size)*self.n_participants)]
        test_idxs = idxs[int((train_size+val_size)*self.n_participants):]

        return {"train": train_idxs, "val": val_idxs, "test": test_idxs}
                                         
                                         
    def get_data_split(self, stratified: bool = False):
        """
        Get the train, validation, and test splits.
        """

        train_size = self.config["train_size"]
        val_size = self.config["val_size"]
        test_size = self.config["test_size"]

        unique_participants = self.data["participant_id"].unique()
        n_participants = len(unique_participants)

        if self.config.shuffle:
            np.random.shuffle(unique_participants)

        train_participants = unique_participants[:int(train_size * n_participants)]
        val_participants = unique_participants[int(train_size * n_participants):int((train_size + val_size) * n_participants)]
        test_participants = unique_participants[int((train_size + val_size) * n_participants):]

        train_data = self.data[self.data["participant_id"].isin(train_participants)]
        val_data = self.data[self.data["participant_id"].isin(val_participants)]
        test_data = self.data[self.data["participant_id"].isin(test_participants)]

        return {"train": train_data, "val": val_data, "test": test_data}

        

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):

    # Load data
    # Remove data/ from the path
    # if config.dataset.data_path[:5] == "data/":
    #     config.dataset.data_path = config.dataset.data_path[5:]
    config.debug = False
    ds = VaalikoneDataset(config=config)
    
    splits = ds.get_idx_split()
    test_idxs = splits["test"]
    test_ds = torch.utils.data.Subset(ds, test_idxs)

    train_idxs = splits["train"]
    train_ds = torch.utils.data.Subset(ds, train_idxs)

    val_idxs = splits["val"]
    val_ds = torch.utils.data.Subset(ds, val_idxs)

    print("Number of participants in train set:", len(train_ds))
    print("Number of participants in val set:", len(val_ds))
    print("Number of participants in test set:", len(test_ds))

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=config.shuffle)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=config.shuffle)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=config.shuffle)


    for dl in [train_dl, val_dl, test_dl]:
        classes = {}
        for sample in dl:
            party_id = sample['party_id'].item()
            if party_id not in classes:
                classes[party_id] = 1
            else:
                classes[party_id] += 1
            
        print(classes)

if __name__ == "__main__":
    main()