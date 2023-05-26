import os
from abc import ABC, abstractmethod
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, InMemoryDataset
import pandas as pd
from transformers import (AutoTokenizer, AutoModel,
                            BertTokenizer, BertForMaskedLM)
from tqdm import tqdm

from src.utils.vis_utils import visualize_dataset
from src.utils.utils import is_truncated_precise

