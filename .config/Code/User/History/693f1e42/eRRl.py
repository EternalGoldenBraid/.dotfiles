# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import warnings
import os

from tqdm import tqdm
import pickle
import torch
from omegaconf import DictConfig
from data.dataloader_slaps import VaalikoneDataset
from data.dataloader_slaps_batch_mode import VaalikoneDataset as VaalikoneDatasetBatch
from pathlib import Path

from slaps.bert_encoder import BertEncoder

import numpy as np


warnings.simplefilter("ignore")

# from citation_networks import load_citation_network, sample_mask
from transformers import (AutoModel, BertForMaskedLM,
                         get_cosine_schedule_with_warmup, AutoTokenizer)

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)

def load_vaaliperttu_data(config: DictConfig):

    
    # Load data and access to tokenized data via __getitem__
    ds = VaalikoneDatasetBatch(config=config, debug=config.debug)
    
    # TODO Batch mode is not working yet due to data loader __getitem__.
    # dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    # for ds_, dl_ in zip(ds, dataloader):
    # for idx in range(len(ds)):
    #     ds_ = ds[idx]
    #     dl_ = next(iter(dataloader))
    #     # print(len(ds_), len(dl_))
    #     # print('input_ids:', ds_['input_ids'].shape, dl_['input_ids'].shape)
    #     # print('labels:', ds_['labels'].shape, dl_['labels'].shape)
        
    #     print()
        
    split_idx = ds.get_idx_split()
    train_mask = sample_mask(split_idx['train'], len(ds))
    val_mask = sample_mask(split_idx['val'], len(ds))
    test_mask = sample_mask(split_idx['test'], len(ds))

    nclasses = config.finetune.num_classes

    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    return ds, nclasses, train_mask, val_mask, test_mask


def load_vaaliperttu_data_frozen(config: DictConfig, tokenizer=None):

    # Load data and access to tokenized data via __getitem__
    # ds = VaalikoneDataset(config=config, debug=config.debug)
    ds = VaalikoneDatasetBatch(config=config, debug=config.debug, tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, 
                    shuffle=False, num_workers=0)
    

    # TODO This should be done in the dataset class? Streamline this saving logic.
    save_dir = Path(config.outputs_dir, config.slaps.slaps_save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    # if ((not save_dir / "slaps_features.pkl" or config.slaps.save_data)
        #  and config.slaps.load_saved_data != True):
    if True:
        print("Encoding data with frozen bert...")
        # Encode with frozen bert (JääPerttu)
        encoder = BertEncoder(config=config)
        features = encoder.bert_step(config, dataloader)
        
        labels = ds.party_label

        split_idx = ds.get_idx_split()
        train_mask = sample_mask(split_idx['train'], features.shape[0])
        val_mask = sample_mask(split_idx['val'], features.shape[0])
        test_mask = sample_mask(split_idx['test'], features.shape[0])

        print("Train_idxs:",np.where(train_mask)[0])
        print("Val_idxs:", np.where(val_mask)[0])
        print("Test_idxs:", np.where(test_mask)[0])
    elif config.slaps.load_saved_data and save_dir.exists():
        
        if not (save_dir / "slaps_features.pkl").exists():
            raise ValueError("Saved data not found in ", save_dir)
        
        with open(save_dir / "slaps_features.pkl", "rb") as f:
            features = pickle.load(f)

        with open(save_dir / "slaps_nfeats.pkl", "rb") as f:
            nfeats = pickle.load(f)

        with open(save_dir / "slaps_labels.pkl", "rb") as f:
            labels = pickle.load(f)

        with open(save_dir / "slaps_nclasses.pkl", "rb") as f:
            nclasses = pickle.load(f)

        with open(save_dir / "slaps_train_mask.pkl", "rb") as f:
            train_mask = pickle.load(f)

        with open(save_dir / "slaps_val_mask.pkl", "rb") as f:
            val_mask = pickle.load(f)

        with open(save_dir / "slaps_test_mask.pkl", "rb") as f:
            test_mask = pickle.load(f)
            
    else:
        raise ValueError("`save_data` and `load_saved_data` cannot both be False")

    print("Loaded participant data for ", ds.n_participants, " participants")
        
    nfeats = features.shape[1]
    nclasses = len(labels.unique())

    
    # print("Percentage:", split_idx["test"], "test, ", split_idx["val"], "val, ", split_idx["train"], "train")

    # labels = torch.LongTensor(labels).view(-1)
    # labels = labels.to(torch.long, requires_grad=False)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    
    # Pickle the data
    if config.slaps.save_data:

        with open(save_dir / "slaps_features.pkl", "wb") as f:
            pickle.dump(features, f)
            
        with open(save_dir / "slaps_nfeats.pkl", "wb") as f:
            pickle.dump(nfeats, f)

        with open(save_dir / "slaps_labels.pkl", "wb") as f:
            pickle.dump(labels, f)

        with open(save_dir / "slaps_nclasses.pkl", "wb") as f:
            pickle.dump(nclasses, f)

        with open(save_dir / "slaps_train_mask.pkl", "wb") as f:
            pickle.dump(train_mask, f)

        with open(save_dir / "slaps_val_mask.pkl", "wb") as f:
            pickle.dump(val_mask, f)

        with open(save_dir / "slaps_test_mask.pkl", "wb") as f:
            pickle.dump(test_mask, f)
            
            
        print("Saved data to ", save_dir)

    # data = {
    #     "features": features,
    #     "nfeats": nfeats,
    #     "labels": labels,
    #     "nclasses": nclasses,
    #     "train_mask": train_mask,
    #     "val_mask": val_mask,
    #     "test_mask": test_mask
    #     "party_names": ds.party_names,
    # }
    return ds, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask

def load_ogb_data(dataset_str):
    """ For legacy """
    from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(dataset_str)

    data = dataset[0]
    features = data.x
    nfeats = data.num_features
    nclasses = dataset.num_classes
    labels = data.y

    split_idx = dataset.get_idx_split()
    
    # DEBUG
    split_idx['train'] = split_idx['train']
    split_idx['valid'] = split_idx['valid']
    split_idx['test'] = split_idx['test']
    new_idxs = np.concatenate([split_idx['train'], split_idx['valid'], split_idx['test']])
    features = features[new_idxs]
    labels = labels[new_idxs]
    nfeats = features.shape[1]
    train_mask = sample_mask(torch.arange(len(split_idx['train'])), features.shape[0])
    val_mask = sample_mask(torch.arange(len(split_idx['train']), 
                            len(split_idx['train']) + len(split_idx['valid'])), features.shape[0])
    test_mask = sample_mask(torch.arange(len(split_idx['train']) + len(split_idx['valid']),
                            len(split_idx['train']) + len(split_idx['valid']) + len(split_idx['test'])), features.shape[0])
    # END DEBUG

    # train_mask = sample_mask(split_idx['train'], data.x.shape[0])
    # val_mask = sample_mask(split_idx['valid'], data.x.shape[0])
    # test_mask = sample_mask(split_idx['test'], data.x.shape[0])

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels).view(-1)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask


def load_data(args, dataset_str_ = None):
    
    if dataset_str_ != None:
        dataset_str = dataset_str_
    else:
        dataset_str = args.dataset

    if dataset_str.startswith('ogb'):
        return load_ogb_data(dataset_str)

    return load_citation_network(dataset_str)

def load_inference_data(config):

    save_dir = Path(config.outputs_dir, config.slaps.slaps_save_dir)
    
    if config.slaps.load_saved_data and save_dir.exists():
        
        if not (save_dir / "slaps_features.pkl").exists():
            raise ValueError("Saved data not found in ", save_dir)
        
        with open(save_dir / "slaps_features.pkl", "rb") as f:
            features = pickle.load(f)

        with open(save_dir / "slaps_nfeats.pkl", "rb") as f:
            nfeats = pickle.load(f)

        with open(save_dir / "slaps_labels.pkl", "rb") as f:
            labels = pickle.load(f)

        with open(save_dir / "slaps_nclasses.pkl", "rb") as f:
            nclasses = pickle.load(f)

        with open(save_dir / "slaps_train_mask.pkl", "rb") as f:
            train_mask = pickle.load(f)

        with open(save_dir / "slaps_val_mask.pkl", "rb") as f:
            val_mask = pickle.load(f)

        with open(save_dir / "slaps_test_mask.pkl", "rb") as f:
            test_mask = pickle.load(f)
            
        fname = 'adjacency_best_val_acc_trial_'
        existing_adjacencies = [f for f in os.listdir(save_dir) if f.startswith(fname)]

        if len(existing_adjacencies) > 0:
            Adjs = [None] * len(existing_adjacencies)
            for i, adj in enumerate(existing_adjacencies):
                with open(save_dir / adj, "rb") as f:
                    Adjs[i] = pickle.load(f)
        else:
            raise ValueError("No saved adjacencies found in ", save_dir)
    else:
        raise ValueError("`save_data` and `load_saved_data` cannot both be False")
    
    data = {
        # 'dataset': dataset,
        'features': features,
        'nfeats': nfeats,
        'labels': labels,
        'nclasses': nclasses,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'Adjs': Adjs
    }
    
    return data


import hydra
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    # config.dataset.data_path = 'data/data_global_questions.csv'
    # config.dataset.data_path = 'data/data_all_answers.csv'
    
    # config.slaps.load_saved_data=False
    # config.slaps.save_data=True
    # config.slaps.model_id="TurkuNLP/bert-base-finnish-cased-v1"
    # config.dataset.data_path = "data/data_no_empty_answers.csv"
    
    config.debug = False
    load_vaaliperttu_data(config)


if __name__ == '__main__':
    # load_data(None, dataset_str_='ogbn-arxiv')
    main()

