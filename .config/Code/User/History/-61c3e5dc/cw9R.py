from typing import List, Tuple, Dict

from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchmetrics.classification import auroc
import pytorch_lightning as pl

class VaalikoneClassifier(pl.LightningModule):

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.bert_model = AutoModel.from_pretrained(config["model_id"], return_dict=True)

        self.hidden = nn.Linear(self.bert_model.config.hidden_size, self.config["hidden_size"])
        self.classifier = nn.Linear(self.config["hidden_size"], config['num_labels'])

        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.dropout = nn.Dropout(0.1)
        # self.auroc = auroc()
        

    def forward(self, input_ids, attention_mask, labels = None):
        
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        
        output = self.dropout(output)
        output = self.hidden(output)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.classifier(output)

        loss = 0
        if labels is not None:
            loss = self.loss(output, labels)
            self.log('train_loss', loss)

        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, logits = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": logits, "labels": labels}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, logits = self(input_ids, attention_mask, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": logits, "labels": labels}

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
