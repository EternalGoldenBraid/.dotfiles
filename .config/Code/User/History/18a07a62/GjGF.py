# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import warnings

from typing import Dict, List
import pickle
import torch
from omegaconf import DictConfig
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

import hydra


warnings.simplefilter("ignore")

# from citation_networks import load_citation_network, sample_mask
from transformers import (AutoModel, BertForMaskedLM,
                         get_cosine_schedule_with_warmup, AutoTokenizer)

        # super(BertEncoder, self).__init__()
# class BertEncoder():
#     def __init__(self, config: DictConfig):
        # super(BertEncoder, self).__init__()
        
class DebugMLP(nn.Module):
    """
    A simple MLP for debugging purposes.
    Replaces the BERT encoder.
    """

    def __init__(self, config, output_dim):

        super(DebugMLP, self).__init__()

        self.fc1 = nn.Linear(2*config.bert['max_token_length'], 768)
        self.fc2 = nn.Linear(768, 768)
        self.fc3 = nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask):
        """
        Forward like in bert.
        Also includes the .pooler_output attribute.
        """
        x = input_ids.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        self.pooler_output = x

        return self


class BertEncoder():
    def __init__(self, config: DictConfig, max_n_questions=None, device='cpu'):
        # super(BertEncoder, self).__init__()

        self.config = config
        self.nfeats = None
        self.bert_dim = None
        self.bert_model = None
        self.device = device
        self.outputs = None
        self.max_n_questions = max_n_questions

        # self.bert_model = AutoModel.from_pretrained(config.slaps["model_id"], return_dict=True)
        # self.bert_model = AutoModel.from_pretrained("checkpoints/mlm",
        #     return_dict=True)
        # self.tokenizer = AutoTokenizer.from_pretrained("checkpoints/mlm")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.slaps['model_id'])
        
        ### DEBUG ###
        if True:
            nfeats_ = 768
            self.bert_model = DebugMLP(config, output_dim=nfeats_)
            self.model = self.bert_model
            self.nfeats = nfeats_
        ### END DEBUG ###
        else:

            self.model = self.bert_model
            self.bert_model.to(config.bert_device)
            
            if config.bert.freeze == 'full':
                for param in self.bert_model.parameters():
                    param.requires_grad = False
            elif config.bert.freeze == 'pooler':
                for param in self.bert_model.pooler.parameters():
                    param.requires_grad = False
                    # assert param.requires_grad == True
            elif config.bert.freeze == 'partial':

                # Freeze all layers
                for n, p in self.model.named_parameters():
                    p.requires_grad = False

                # Unfreeze the last n encoder layers and the prediction head
                n_last_layers = 1  # You can change this number based on your memory constraints
                num_layers = len(self.model.encoder.layer)
                for i in range(num_layers - n_last_layers, num_layers):
                    for param in self.model.encoder.layer[i].parameters():
                        param.requires_grad = True

                # # Unfreeze the prediction head
                # for param in self.model.pooler.parameters():
                #     param.requires_grad = True

            elif config.bert.freeze == 'none':
                for param in self.bert_model.parameters():
                    param.requires_grad = True
                    
            total_params = sum(p.numel() for p in self.model.parameters())
            total_params_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"Model size ~{100*total_params_trainable/total_params:.2f}%")
            
            # self.nfeats = self.bert_model.confs = ig.hidden_size
            self.nfeats = self.model.pooler.dense.out_features
        
    def init_outputs(self, max_n_questions):
        """
        Initialize the outputs tensor.
        A hack to avoid allocating memory for the outputs tensor in the forward pass.
        """
        self.outputs: torch.tensor = torch.zeros((max_n_questions, self.nfeats), 
                                    dtype=torch.float32, device=self.device, requires_grad=True)
    
    # def __call__(self, input_ids, attention_mask, labels, mask_prob=0.15):

    #     # Predict the masked tokens
    #     outputs = self.model(input_ids, labels=labels)
    #     return outputs 

    #     logits = outputs.logits

    #     return logits

    def to(self, device):
        self.model.to(device)
        return self
    
    def train(self):
        self.model.train()
        return self
    
    def eval(self):
        self.model.eval()
        return self
    
    def parameters(self):
        return self.model.parameters()
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        return self
    
    def save_pretrained(self, path):
        self.model.save_pretrained(path)

    
    def encode(self, input_ids: torch.tensor, attention_mask: torch.tensor,
                token_type_ids: torch.tensor):

        """
        Encode the questions for all participants.
        Pool over the questions to get a node representation for each participant.
        
        Number of participants is batch size.
        
        Args:

        Returns:
            features: torch.Tensor of shape (n_participants, bert_dim)

        """
        
        if len(input_ids.shape) != 2:
            raise ValueError("input_ids.shape should be (n_questions, n_tokens), n_questions = batch_size")

        output = self.bert_model(input_ids=input_ids,
            attention_mask=attention_mask).pooler_output

        return output 

    def bert_step(self, config, dataloader, features=None):
        # Encode features with BERT
        # for participant_idx in tqdm(range(len(dataset))):
        
        # if features is None:
        if False:
            features = torch.zeros((len(dataloader), self.nfeats),
                    dtype=torch.float32, requires_grad=False)
            # features = features.cuda(config.slaps_gpu)
            features = features.to('cuda')
        
        # foo = [next(iter(dataloader)) for _ in range(10)]
        # for participant in tqdm(foo, desc='Participant', leave=False):
        
        for participant in tqdm(dataloader, desc='Participant', leave=False):
            # participant_id = participant['participant_id']
            # participant_idx = participant['participant_idx']
            # mask: torch.tensor[bool] = participant['mask'].to(self.device)
            # n_questions: int = participant['n_questions']
            # # input_ids = participant['input_ids'][mask].cuda(config.bert_gpu)
            # # attention_mask = participant['attention_mask'][mask].cuda(config.bert_gpu)
            # # token_type_ids = participant['token_type_ids'][mask].cuda(config.bert_gpu)
            # input_ids = participant['input_ids'].to(self.device)[mask]
            # attention_mask = participant['attention_mask'].to(self.device)[mask]
            # token_type_ids = participant['token_type_ids'].to(self.device)[mask]
            
            ### As tuple
            mask: torch.tensor[bool] = participant[6].to(self.device)
            input_ids = participant[0].to(self.device)[mask]
            attention_mask = participant[1].to(self.device)[mask]
            token_type_ids = participant[2].to(self.device)[mask]
            # label = participant[3].to(self.device)[mask]
            participant_id = participant[4]
            n_questions: int = participant[5]
            # party_id = participant[7]
            participant_idx = participant[8]

            # assert input_ids.shape == attention_mask.shape == token_type_ids.shape
            assert (input_ids.requires_grad == attention_mask.requires_grad == \
                     token_type_ids.requires_grad == False)
            
            # input_ids = torch.randint_like(input_ids, input_ids.max())
            # attention_mask = torch.randint_like(attention_mask, 1)
            # token_type_ids = torch.randint_like(token_type_ids, 1)
            
            # return features
            
            # total_batch_size = input_ids.shape[0] # Dataloader batch size == n_participants at a time
            if n_questions > config.bert.batch_size: 
                input_ids_batch = torch.split(input_ids, config.bert.batch_size, dim=0)
                attention_mask_batch = torch.split(attention_mask, config.bert.batch_size, dim=0)
                token_type_ids_batch = torch.split(token_type_ids, config.bert.batch_size, dim=0)
                
                # for i_ in range(len(input_ids_batch)):
                for batch_idx, (i_ids, i_mask, t_t_id) in enumerate(zip(input_ids_batch, attention_mask_batch, token_type_ids_batch)):
                    start = batch_idx * config.bert.batch_size
                    end = start + config.bert.batch_size
                    self.outputs[start:end] = self.encode(i_ids, i_mask, t_t_id)
                    
                features[participant_idx] = self.outputs.mean(dim=0)
                self.outputs[:] = 0.
            else:
                features[participant_idx] = self.encode(input_ids, attention_mask,
                                            token_type_ids).mean(dim=0)
            
        return features

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    # config.dataset.data_path = 'data/data_global_questions.csv'
    # config.dataset.data_path = 'data/data_all_answers.csv'
    
    # config.slaps.load_saved_data=False
    # config.slaps.save_data=True
    # config.slaps.model_id="TurkuNLP/bert-base-finnish-cased-v1"
    # config.dataset.data_path = "data/data_no_empty_answers.csv"
    load_vaaliperttu_data(config)


if __name__ == '__main__':
    # load_data(None, dataset_str_='ogbn-arxiv')
    main()

