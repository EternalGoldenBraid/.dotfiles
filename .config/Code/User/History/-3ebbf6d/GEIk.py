# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

import torch
from torch import nn
from torch.nn import functional as F

from transformers import BertTokenizer, BertForMaskedLM

class BertMLMEncoder:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.model.language_encoders.bert.model_name)
        self.model = BertForMaskedLM.from_pretrained(config.model.language_encoders.bert.model_name)

        # if config.model.language_encoders.bert.freeze:
        if True:
            # Freeze all layers
            for n, p in self.model.named_parameters():
                p.requires_grad = False

            # Unfreeze the last n encoder layers and the prediction head
            n_last_layers = 2  # You can change this number based on your memory constraints
            num_layers = len(self.model.bert.encoder.layer)
            for i in range(num_layers - n_last_layers, num_layers):
                for param in self.model.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

            # Unfreeze the prediction head
            for param in self.model.cls.parameters():
                param.requires_grad = True

    def mask_tokens(self, inputs, mask_prob):
        """
        Create a mask for a certain percentage of tokens in the input sequence.
        """
        mask = torch.bernoulli(torch.full(inputs.shape, mask_prob)).bool()
        labels = torch.where(mask, inputs, -100)
        masked_inputs = inputs.clone()
        masked_inputs[mask] = self.tokenizer.mask_token_id
        return masked_inputs, labels

    def __call__(self, input_ids, attention_mask, labels, mask_prob=0.15):

        # Predict the masked tokens
        outputs = self.model(input_ids, labels=labels)
        return outputs 

        logits = outputs.logits

        return logits

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
    
    @property
    def state_dict(self):
        return self.model.state_dict()
    