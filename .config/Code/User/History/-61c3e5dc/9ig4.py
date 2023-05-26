from typing import List, Tuple, Dict

from transformers import AutoModel, get_cosine_schedule_with_warmup, AutoTokenizer
from torch.optim.adamw import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchmetrics.classification import auroc
import pytorch_lightning as pl
import pandas as pd
import numpy as np

class VaalikoneClassifier(pl.LightningModule):

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        self.bert_model = AutoModel.from_pretrained(self.config["model_id"], return_dict=True)

        self.hidden = nn.Linear(self.bert_model.config.hidden_size, self.config["hidden_size"])
        self.classifier = nn.Linear(self.config["hidden_size"], config['num_classes'])

        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.dropout = nn.Dropout(0.1)
        # self.auroc = auroc()


    def forward(self, question_dict: dict): # input_ids, attention_mask, labels = None):

        bert_outputs: None | np.ndarray = None

        for question_id, sample in question_dict.items():
            if not (0 <= question_id <= 29):
                print("Forward got participant_data by accident")
                continue 
            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]

            output: torch.Tensor = self.bert_model(input_ids=input_ids.squeeze(), attention_mask=attention_mask.squeeze()).pooler_output

            # Detach Tensor from gpu memory
            output = output.cpu().detach()

            if bert_outputs is None:
                # Initialize 3d-matrix
                bert_outputs = output.numpy()
                bert_outputs = bert_outputs[..., np.newaxis]
            else:
                # append to 3d-matrix
                #bert_outputs = np.stack([bert_outputs, output], axis=-1)
                bert_outputs = np.append(bert_outputs, np.atleast_3d(output), axis=2)
        
        # Calculate mean in the correct form
        bert_mean = bert_outputs.mean(axis=2)

        bert_mean = torch.from_numpy(bert_mean)
        bert_mean = bert_mean.cuda()

        output = self.dropout(bert_mean)
        output = self.hidden(output)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.classifier(output)

        return output

    def training_step(self, batch: dict, batch_idx):
        participant_data = batch.pop("participant_data")

        one_hot_party = participant_data["label"]
        one_hot_party = torch.stack(one_hot_party).transpose(0, 1).to(torch.int)

        model_output = self(batch)
        loss = self.loss(model_output, one_hot_party)

        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": model_output, "labels": one_hot_party}


    def validation_step(self, batch: dict, batch_idx):
        participant_data = batch.pop("participant_data")

        one_hot_party = participant_data["label"]
        one_hot_party = torch.stack(one_hot_party).transpose(0, 1).to(torch.int)

        model_output = self(batch)
        loss = self.loss(model_output, one_hot_party)

        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": model_output, "labels": one_hot_party}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        _, logits = self(input_ids, attention_mask, labels)
        return {"predictions": logits, "labels": labels}
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        warmup_steps = math.ceil(self.config["train_size"] * self.config["epochs"] * self.config["warmup_proportion"])
        total_steps = math.ceil(self.config["train_size"] * self.config["epochs"])
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        return [optimizer], [scheduler]
