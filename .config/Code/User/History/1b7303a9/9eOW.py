import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from transformers import DataCollatorForLanguageModeling
import hydra
from omegaconf import DictConfig
import omegaconf

import wandb

from src.models.bertmlm import BertMLMEncoder
from src.data.daily_mlm_dataset import MLMEventFeatureDataset
from src.utils.train_utils import train
from src.utils.utils import evaluate, evaluate_mlm

@hydra.main(config_path="conf", config_name="config")
def train_mlm(cfg: DictConfig) -> None:
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    os.environ["WANDB_MODE"] = cfg.wandb.wandb_mode
    
    run = wandb.init(
        entity=cfg.wandb.entity, 
        project=cfg.wandb.project_name,
        config = wandb.config,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the tokenizer and model
    # model_conf = cfg.model.language_encoders.minilm
    model_conf = cfg.model.language_encoders.bert
    model = BertMLMEncoder(config=model_conf).to(device)
    # model = BertMLMEncoder(config=cfg)
    # model = model.to(device)
    tokenizer = model.tokenizer

    # Create the dataset
    dataset = MLMEventFeatureDataset(
        root=hydra.utils.to_absolute_path(cfg.data.dataset.root),
        filename=cfg.data.dataset.filename,
        config=cfg, tokenizer=tokenizer
    )
    df = dataset.df

    # Split the dataset into train, validation, and test datasets
    train_len = int(len(dataset) * cfg.data.dataset.train_ratio)
    val_len = int(len(dataset) * cfg.data.dataset.val_ratio)
    test_len = len(dataset) - train_len - val_len
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])


    # Create data collator for masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.data.dataset.mask_probability
    )

    # Create data loaders for train, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=model_conf.batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=model_conf.batch_size, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=model_conf.batch_size, shuffle=False, collate_fn=data_collator)
    
    # Train the model
    model_save_path = Path(cfg.paths.ckpt_dir, 'mlm', "best_model_mlm.pt")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    train(config=cfg, model=model, train_dataloader=train_loader,
            val_dataloader=val_loader, device=device, model_save_path=model_save_path)

    # Evaluate the model on the test dataset
    # model.state_dict(torch.load(model_save_path))
    model.load_state_dict(torch.load(model_save_path))
    # test_metrics = evaluate_mlm(cfg=cfg, model=model, dataloader=None, dataset=dataset, device=device)
    # print(f"Test metrics: {test_metrics}")

    max_samples = 7
    val_metrics = evaluate_mlm(cfg=cfg, model=model, dataset=dataset,
                                device=device, max_samples=max_samples,
                                collator=data_collator)
    print(f"Val metrics: {val_metrics}")

if __name__ == "__main__":
    train_mlm()
