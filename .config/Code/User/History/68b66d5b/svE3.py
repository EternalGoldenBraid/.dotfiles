import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

def train(model, train_dataloader, val_dataloader, config, device, model_save_path):
    model = model.to(device)
    model.train()
    
    # Set up the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)
    total_steps = len(train_dataloader) * config.training.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.training.warmup_steps, num_training_steps=total_steps)
    
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(config.training.epochs):
        print(f"Epoch {epoch + 1}/{config.training.epochs}")
        epoch_loss = 0
        
        for batch in tqdm(train_dataloader, desc="Iteration"):
            input_ids, labels = batch['input_ids'], batch['labels']
            attention_mask = (input_ids != model.tokenizer.pad_token_id).long()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # if 103 not in input_ids:
            #     print("103 not in input_ids")
            #     continue

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # loss = F.cross_entropy(input=logits.view(-1, logits.size(-1)),
                                    # target=labels.view(-1), ignore_index=-100
                                # )
            loss = outputs.loss
            
            if loss.item() == torch.nan:
                print("loss is nan")
                continue

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1
            
        avg_train_loss = epoch_loss / len(train_dataloader)
        print(f"Average train loss: {avg_train_loss}")

        val_loss = validate(model, val_dataloader, device)
        print(f"Validation loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)


    return global_step


def validate(model, dataloader, device):
    model.eval()

    total_loss = 0
    criterion = CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels = batch['input_ids'], batch['labels']
            attention_mask = (input_ids != model.tokenizer.pad_token_id).long()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)
