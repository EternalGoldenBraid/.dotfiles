from pathlib import Path
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
import omegaconf
import wandb
from transformers import pipeline
from transformers import RobertaTokenizer, RobertaModel, AutoModel, AutoTokenizer
from transformers import logging
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.accelerators import find_usable_cuda_devices

# from data import Dataloader_finetuning
from data.dataloader_global_questions import VaalikoneDataset, VaalikoneDataModule
from models.model import VaalikoneClassifier
from data.create_tiny_data import create_tiny_data

from utils.utils import initialize

logging.set_verbosity_error()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    config = config
    # initialize(config)

    # Clean vaalit_2019.csv and save it to data/finetune.csv if it doesn't exist
    # Create training, validation and test sets if they don't exist
    data_conf = config.dataset
    data_path = data_conf.data_path

    if not os.path.exists(data_path):
        raise Exception("data/data_all_answers.csv does not exist. Run data/clean_data.py first.")
        
    if not os.path.exists(data_conf.train_path) or not os.path.exists(data_conf.val_path) or not os.path.exists(data_conf.test_path):
        print("Creating train, val and test sets")
        df = pd.read_csv(data_path)

        # Split the data into train, val and test sets
        train_end = config.dataset.train_data_size
        val_end = config.dataset.train_data_size + config.dataset.val_data_size
        test_end = config.dataset.val_data_size + config.dataset.test_data_size
        train_df = df.iloc[:int(len(df) * train_end)]
        val_df = df.iloc[int(len(df) * train_end):int(len(df) * val_end)]
        test_df = df.iloc[int(len(df) * val_end):int(len(df) * test_end)]

        train_df.reset_index()
        val_df.reset_index()
        test_df.reset_index()
        
        train_df.to_csv(data_conf.train_path, index=False)
        val_df.to_csv(data_conf.val_path, index=False)
        test_df.to_csv(data_conf.test_path, index=False)


    #### DEBUG ####

    ### Test dataset module
    # Load data
    #ds_train = VaalikoneDataset(
    #    path=config["train_path"],
    #    config=config)
    
        # dl_train.__getitem__(1)
    
    # ds = VaalikoneDataset(
    #     # path=config["val_path"],
    #     path=config["train_path"],
    #     config=config)
    
    # idx = 1
    # item = ds.__getitem__(idx)
    # input_ids = item["input_ids"]
    # attention_mask = item["attention_mask"]
    # labels = item["labels"]

    # classifier = VaalikoneClassifier(config=config)
    # loss, logits = classifier(input_ids, attention_mask)

    # # Softmax
    # probs = torch.softmax(logits, dim=1)
    # print(probs)
    # print(ds.data.iloc[idx])

    #### END DEBUG ####

    # Train loop
    datamodule = VaalikoneDataModule(
        train_path=data_conf.train_path,
        val_path=data_conf.val_path,
        test_path=data_conf.test_path,
        config=config)
    datamodule.setup()

    # sample = (datamodule.train_dataloader()).dataset[1]
    with open_dict(config):
        config["finetune"]["train_size"] = len(datamodule.train_dataloader())

    classifier = VaalikoneClassifier(config=config.finetune)
    #classifier = classifier.cuda(CUDA_DEVICE_NR)


    # True makes the logger default
    # logger = pl.loggers.WandbLogger(project=config.wandb.project_name, save_dir="logs") if config.wandb.use_wandb else True


    trainer = pl.Trainer(max_epochs=config.finetune.epochs,
#                        num_sanity_val_steps=50,
                        accelerator=config.finetune.device,
                        devices=config.finetune.nr_of_gpus,
                        # logger=logger,
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min")]
                    )

    trainer.fit(classifier, datamodule)


if __name__ == "__main__":
    main()