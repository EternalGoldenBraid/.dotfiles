import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from transformers import DataCollatorForLanguageModeling
import hydra
from omegaconf import DictConfig

from src.models.bertmlm import BertMLMEncoder
from src.data.daily_mlm_dataset import MLMEventFeatureDataset
from src.utils.train_utils import train
from src.utils.utils import evaluate

@hydra.main(config_path="conf", config_name="config")
def train_mlm(cfg: DictConfig) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the tokenizer and model
    model = BertMLMEncoder(config=cfg).to(device)
    # model = BertMLMEncoder(config=cfg)
    # model = model.to(device)
    tokenizer = model.tokenizer

    # Create the dataset
    dataset = MLMEventFeatureDataset(
        root=hydra.utils.to_absolute_path(cfg.data.dataset.root),
        filename=cfg.data.dataset.filename,
        config=cfg, tokenizer=tokenizer
    )

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
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=data_collator)
    
    ### DEBUG
    for batch in train_loader:
        input_ids, labels = batch['input_ids'], batch['labels']
        # attention_mask = (input_ids != model.tokenizer.pad_token_id).long()
        attention_mask = (input_ids != model.tokenizer.pad_token_id)
        input_ids = input_ids.to(device)
        pass

    # Train the model
    train(config=cfg, model=model, train_dataloader=train_loader,
            val_dataloader=val_loader, device=device)

    # Evaluate the model on the test dataset
    # test_metrics = evaluate(cfg, model, test_loader, device)
    # print(f"Test metrics: {test_metrics}")

if __name__ == "__main__":
    train_mlm()
