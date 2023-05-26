import os
from abc import ABC, abstractmethod
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, InMemoryDataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from src.utils.vis_utils import visualize_dataset


class EventFeatureDataset(InMemoryDataset):
    def __init__(self, root, config, filename,
                transform=None, pre_transform=None, pre_filter=None):
        self.filename = filename
        self.config = config
        
        self._processed_dir = hydra.utils.to_absolute_path(config.data.dataset.processed_dir)
        self._raw_dir = hydra.utils.to_absolute_path(config.data.dataset.raw_dir)
        

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # self.unique_tags = None
        # self.df = None
        # self.tag_to_id = None
        self.unique_tags = self.data.unique_tags
        self.df = self.data.df
        self.tag_to_id = self.data.tag_to_id
        self.id_to_tag = self.data.id_to_tag
    
        # Create train, val, test splits
        dataset = self
        features = dataset[torch.arange(len(dataset))].x    
        targets = dataset[torch.arange(len(dataset))].y
        targets = targets.reshape(len(dataset), -1)

        assert len(features) == len(targets)
        assert (dataset[0].y == targets[0]).all()
        
        visualize_dataset(dataset=dataset, df=dataset.df, cfg=config, data_root=Path(self.root))

        if self.config.data.dataset.shuffle == True:
            idxs = torch.randperm(len(dataset))
            features = features[idxs]
            targets = targets[idxs]
        elif self.config.data.shuffle == False:
            pass
        else:
            raise ValueError(f"self.config.data.shuffle should be True or False, but got {self.config.data.shuffle}")


        train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        train_mask[torch.arange(int(len(dataset) * self.config.data.dataset.train_ratio))] = True
        val_mask = torch.zeros(len(dataset), dtype=torch.bool)
        val_mask[torch.arange(int(len(dataset) * self.config.data.dataset.train_ratio),
                int(len(dataset) * (self.config.data.dataset.train_ratio + self.config.data.dataset.val_ratio)))] = True

        test_mask = torch.zeros(len(dataset), dtype=torch.bool)
        test_mask[torch.arange(int(len(dataset) * (self.config.data.dataset.train_ratio + self.config.data.dataset.val_ratio)), len(dataset))] = True
        
        dataset.data.train_mask = train_mask
        dataset.data.val_mask = val_mask
        dataset.data.test_mask = test_mask

        dataset.features = features
        dataset.targets = targets

    @property
    def raw_file_names(self):
        return [self.filename]

    @property
    def processed_file_names(self):
        return [f'data.pt']
        # return [os.path.join(self._processed_dir, 'data.pt')]
        # return [f'{self.processed_dir}/data.pt']

    def download(self):
        pass

    def process(self):

        # story_encoder = BertEncoder(self.config)
        story_encoder = MiniLMEncoder(self.config)

        self.df = pd.read_hdf(self.raw_paths[0], key=self.config.data.dataset.key)
        
        if self.config.mode.debug == True:
            df_ = self.df.copy()
            # self.df = self.df.iloc[-1000:]
            # self.df = self.df.iloc[-100:]
            
        # Filter tags with too few occurences
        k = self.config.data.dataset.min_tag_occurrences
        tag_counts = self.df['tag_name'].explode().value_counts()
        valid_tag_names = tag_counts[tag_counts >= k].index.tolist()
        self.df['tag_name'] = self.df['tag_name'].apply(
                                lambda x: [tag for tag in x if tag in valid_tag_names]
                                )
        self.df = self.df[self.df['tag_name'].apply(lambda x: len(x) > 0)]

        if len(self.df) == 0:
            raise ValueError("No valid data points found. Check min_tag_occurrences")
        # k = cfg.data.dataset.min_tag_occurrences
        # tag_counts = df['tag_id'].explode().value_counts()
        # df = df[df['tag_id'].apply(lambda x: len(set(x) & set(tag_counts[tag_counts >= k].index)) > 0)]
        data_list = []
        
        # TODO How sparse and high are the labels?
        # unique_tags = sorted(set(tag for tag_list in self.df['tag_id'] for tag in tag_list))
        unique_tags = sorted(set(tag for tag_list in self.df['tag_name'] for tag in tag_list))
        self.tag_to_id = {tag: i for i, tag in enumerate(unique_tags)}
        self.id_to_tag = {i: tag for i, tag in enumerate(unique_tags)}
        self.df['tag_id'] = self.df['tag_name'].apply(lambda x: [self.tag_to_id[tag] for tag in x])
        # self.unique_tags = unique_tags
        self.unique_tags = [self.tag_to_id[tag] for tag in unique_tags]
        num_tags = len(unique_tags)
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            
            with torch.no_grad():
                # embeddings = story_encoder.model(input_ids)[0][:, 0, :]
                # embeddings = story_encoder.model(input_ids)
                embeddings = story_encoder(row['story'])

                decoded = story_encoder.decode(row['story'])
                story = row['story']
                story = ' '.join(story.split()).lower()
                assert story == decoded or abs(len(story) - len(decoded)) < 10, f"""
                    {story}
                    !=
                    {decoded}
                    """

            if len(self.config.data.dataset.other_features) > 0:
                x = torch.tensor(row[self.config.data.dataset.other_features], dtype=torch.float)[None,...]
                x = torch.cat([x, embeddings], dim=1)
            else:
                x = embeddings

            y = np.zeros(num_tags, dtype=int)
            # y[row['tag_id']] = 1
            # for tag in row['tag_id']:
            for tag in row['tag_name']:
                y[self.tag_to_id[tag]] = 1
            y = torch.tensor(y, dtype=torch.float)

            data_ = Data(x=x, y=y, 
                        story=row['story'],
                        date=row['date'],
                        tag_name=row['tag_name'])
            data_list.append(data_)

        data, slices = self.collate(data_list)
        
        # TODO Where to store these?
        data.tag_to_id = self.tag_to_id
        data.id_to_tag = self.id_to_tag
        data.unique_tags = self.unique_tags
        data.df = self.df
        
        torch.save((data, slices), self.processed_paths[0])
        
        # self.num_classes = num_tags

    def __repr__(self):
        return f"{self.__class__.__name__}(len={len(self)}, root='{self.root}')"

    def set_transform(self, transform):
        self.transform = transform
        
class LanguageEncoder(ABC):
    def __init__(self, config, model_name):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        for n, p in self.model.named_parameters():
            p.requires_grad = False

    @abstractmethod
    def __call__(self, data):
        pass

    @abstractmethod
    def decode(self, data):
        pass
    
class BertEncoder(LanguageEncoder):
    def __init__(self, config):
        super().__init__(config, config.model.language_encoders.bert.model_name)

    def __call__(self, data):
        input_ids = self.tokenizer.encode(data, return_tensors='pt',
                            max_length=self.config.model.language_encoders.bert.max_seq_length,
                            truncation=True, padding='max_length')
        with torch.no_grad():
            embeddings = self.model(input_ids)[0][:, 0, :]
        return embeddings

    def decode(self, data):
        input_ids = self.tokenizer.encode(data, return_tensors='pt',
                            max_length=self.config.model.language_encoders.bert.max_seq_length,
                            truncation=True, padding='max_length')
        decoded = self.tokenizer.decode(input_ids[0])
        decoded = decoded.replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]', '')
        decoded = ' '.join(decoded.split())
        return decoded

class MiniLMEncoder(LanguageEncoder):
    def __init__(self, config):
        super().__init__(config, config.model.language_encoders.minilm.model_name)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def __call__(self, data):
        encoded_input = self.tokenizer(data, return_tensors='pt',
                                        max_length=self.config.model.language_encoders.minilm.max_seq_length,
                                        truncation=True, padding='max_length')

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def decode(self, data):
        input_ids = self.tokenizer.encode(data, return_tensors='pt',
                            max_length=self.config.model.language_encoders.bert.max_seq_length,
                            truncation=True, padding='max_length')
        decoded = self.tokenizer.decode(input_ids[0])
        decoded = decoded.replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]', '')
        decoded = ' '.join(decoded.split())
        return decoded