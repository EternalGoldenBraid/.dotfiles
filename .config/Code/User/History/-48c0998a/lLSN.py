from pathlib import Path
import os

from transformers import pipeline
from transformers import RobertaTokenizer, RobertaModel, AutoModel, AutoTokenizer
import pandas as pd
import torch

# from data import Dataloader_finetuning
from data.dataloader_finetuning import VaalikoneDataset, VaalikoneDataModule
from models.model import VaalikoneClassifier
from config.config import finetune_config

# Clean vaalit_2019.csv and save it to data/finetune.csv if it doesn't exist
data_path = "data/finetune.csv"
if not os.path.exists(data_path):
    print("Cleaning data")
    df = pd.read_csv("data/vaalit_2019.csv")
    
    # Remove rows with empty strings
    lappi_column_names = [col for col in df.columns if col[:6] == 'Lappi.']
    df = df.dropna(subset=lappi_column_names)
    # df = df[df.apply(lambda x: x.str.len().gt(0).all(), axis=1)]

    # df.to_csv(data_path, index=False)
    df.to_csv(data_path)

# Create training, validation and test sets if they don't exist
train_path = "data/finetune_train.csv"
val_path = "data/finetune_val.csv"
test_path = "data/finetune_test.csv"

if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
    print("Creating train, val and test sets")
    df = pd.read_csv(data_path)
    # df = df.sample(frac=1).reset_index(drop=True)
    
    # Split the data into train, val and test sets
    train_df = df.iloc[:int(len(df) * 0.8)]
    val_df = df.iloc[int(len(df) * 0.8):int(len(df) * 0.9)]
    test_df = df.iloc[int(len(df) * 0.9):]
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

# Load both turku and roberta models. Turku seems more rigorous and documented.
model_id: str = 'TurkuNLP/bert-large-finnish-cased-v1'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)



### Test dataset module
# Load data
ds_train = VaalikoneDataset(
    path=config["train_path"],
    config=config)

# dl_train.__getitem__(1)

# ### Test pytorch-lightning module
# datamodule = VaalikoneDataModule(
#     train_path=config["train_path"],
#     val_path=config["val_path"],
#     test_path=config["test_path"],
#     tokenizer=tokenizer,
#     config=config)
# datamodule.setup()
# dl_train2 = datamodule.train_dataloader()

# print(len(dl_train2))
# print(len(dl_train))

classifier = VaalikoneClassifier(config=config)

idx = 1
item = ds_train.__getitem__(idx)
input_ids = item["input_ids"]
attention_mask = item["attention_mask"]
labels = item["labels"]
loss, logits = classifier(input_ids, attention_mask)

# Softmax
probs = torch.softmax(logits, dim=1)
print(probs)
print(ds_train.data.iloc[idx])