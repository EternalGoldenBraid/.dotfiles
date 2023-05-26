import hydra
from hydra.core.hydra_config import HydraConfig
import pandas as pd

import omegaconf
from omegaconf import DictConfig, open_dict
import torch
import torch.nn.functional as F
import wandb

def initialize(config: DictConfig):
    if config.wandb.use_wandb:
        # Set config for the training
        wandb.config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )   # track hyperparameters and run metadata

        # Set other parameters
        wandb.init(
            project=config.wandb.project_name,  # Project name
        )

    if config.finetune.use_tensor_cores:
        torch.set_float32_matmul_precision("medium")
        
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = torch.inf

    def early_stop(self, validation_loss):
        # print("counter:",self.counter, validation_loss, self.min_validation_loss, self.min_validation_loss + self.min_delta)
        if validation_loss != torch.inf:
            pass
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
from tqdm import tqdm
# def evaluate_mlm(model, dataloader, cfg, device='cuda'):

#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Iteration"):
#             input_ids, labels = batch['input_ids'], batch['labels']
#             attention_mask = (input_ids != model.tokenizer.pad_token_id).long()
#             input_ids = input_ids.to(device)
#             attention_mask = attention_mask.to(device)
#             labels = labels.to(device)

#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
#             predicted_token_ids = outputs.logits.argmax(dim=-1)
#             predicted_text = model.tokenizer.batch_decode(predicted_token_ids)

#             print(30 * "-")
#             print("Predicted:" ,predicted_text)
#             print()
#             print("Actual:", model.tokenizer.batch_decode(input_ids))
#             print(30 * "-")
#             pass
        
def evaluate_mlm(model,  cfg, dataloader=None, dataset=None, device='cuda', max_samples=10,
                    collator=None):
    
    if dataloader is None and dataset is None:
        raise ValueError("Either dataloader or dataset must be provided")

    with torch.no_grad():
        
        if dataset is None:
            for batch in tqdm(dataloader, desc="Iteration"):
                input_ids, labels = batch['input_ids'], batch['labels']
                attention_mask = (input_ids != model.tokenizer.pad_token_id).long()
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                predicted_token_ids = outputs.logits.argmax(dim=-1)
                predicted_texts = model.tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

                actual_texts = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

                print(30 * "-")
                # for predicted_text, actual_text in zip(predicted_texts, actual_texts):
                for idx, (predicted_text, actual_text) in enumerate(zip(predicted_texts, actual_texts)):
                    log_entry = {
                        f'test_{idx}_predicted': predicted_text,
                        f'test_{idx}_actual': actual_text
                    }
                    wandb.log(log_entry)
                    print("Predicted:", predicted_text)
                    print()
                    print("Actual:", actual_text)
                    print(30 * "-")
                    # print("Predicted:", model.tokenizer.convert_ids_to_tokens(predicted_token_ids))
                    # print()
                    # print("Actual:", model.tokenizer.convert_ids_to_tokens(input_ids))
                    # print(30 * "-")
        else:
            idxs = torch.randperm(len(dataset))[:max_samples]
            # for batch in tqdm(dataloader, desc="Iteration"):
            for batch_idx in idxs:
                batch_idx = batch_idx.item()
                input_ids = dataset[batch_idx][None,...]
                # input_ids, labels = model.mask_tokens(input_ids, cfg.data.dataset.mask_probability)
                input_ids, labels = collator.torch_mask_tokens(input_ids)
                attention_mask = (input_ids != model.tokenizer.pad_token_id).long()

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                predicted_token_ids = outputs.logits.argmax(dim=-1)
                predicted_text = model.tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=False)

                actual_text = model.tokenizer.batch_decode(input_ids, skip_special_tokens=False)

                print(30 * "-")
                # for idx, (predicted_text, actual_text) in enumerate(zip(predicted_texts, actual_texts)):
                #     log_entry = {
                #         f'test_{idx}_story': dataset.df.iloc[batch_idx]['story'],
                #         f'test_{idx}_actual': actual_text,
                #         f'test_{idx}_predicted': predicted_text,
                #     }
                #     wandb.log(log_entry)
                #     print("Story:", dataset.df.iloc[idx]['story'])
                #     print()
                #     print("Actual:", actual_text)
                #     print()
                #     print("Predicted:", predicted_text)
                #     print(30 * "-")
                story = dataset.df.iloc[batch_idx]['story']
                log_entry = {
                    f'test_{batch_idx}_story': story,
                    f'test_{batch_idx}_actual': actual_text,
                    f'test_{batch_idx}_predicted': predicted_text,
                }
                wandb.log(log_entry)
                print("Story:", story)
                # print()
                # print("Actual:", actual_text)
                # print()
                # print("Predicted:", predicted_text)
                # print(30 * "-")
                print("Actual:", model.tokenizer.convert_ids_to_tokens(input_ids[0]))
                print()
                print("Predicted:", model.tokenizer.convert_ids_to_tokens(predicted_token_ids[0]))
                print(30 * "-")

    
def evaluate(model, dataset, cfg, k=3, features=None, adj=None):
    features = dataset.features if features is None else features
    labels = dataset.targets
    mask = dataset.train_mask
    
    df = dataset.df.copy()
    
    idx_to_name = dataset.id_to_tag

    model.eval()
    with torch.no_grad():
        # model.to(features.device)
        features = features.to('cuda')
        if adj is None:
            out = model(features[mask])
        else:
            out = model(x=features, adj_t=adj)[mask]
        
        probs = F.sigmoid(out)
        
        # print(probs)

        # Get indices of entries over the threshold
        vals, indices = torch.topk(probs, k=k, dim=1)
        # print(vals)
        # print(indices)

        # Get the names of the tags
        mapped_names = [[None] * k for i in range(indices.shape[0])]
        for i in range(indices.shape[0]):
            # mapped_names.append([idx_to_name[idx.item()] for idx in indices[i]])
            mapped_names[i] = [idx_to_name[idx.item()] for idx in indices[i]]

        # print(mapped_names)
        # df.iloc[dataset.test_mask, 'predicted_tags'] = mapped_names
        # Create a new Series for the specified subset of rows
        # predicted_tags_series = pd.Series(index=df.index[dataset.test_mask], data=mapped_names, name='predicted_tags')
        # df = df.merge(predicted_tags_series, left_index=True, right_index=True, how='left')

        df = df.merge(pd.Series(index=df.index[mask], data=mapped_names, name='predicted_tags'),
                     left_index=True, right_index=True, how='left')
        
        print(df[['tag_name', 'predicted_tags', 'story']].loc[mask.numpy()])
        
def is_truncated_precise(original_story, tokenizer, max_seq_length):
    # Tokenize the original story without truncation or padding
    input_ids = tokenizer.encode(original_story, return_tensors='pt', add_special_tokens=False)

    # Calculate the number of tokens (excluding special tokens)
    num_tokens = input_ids.size(-1)

    # Check if the tokenized story would be truncated with truncation enabled
    return num_tokens > max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)