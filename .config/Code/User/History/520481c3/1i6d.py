import os
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from wordcloud import WordCloud, STOPWORDS
import torch
from torch_geometric.data import Dataset

def visualize_dataset(df: pd.DataFrame, dataset: Dataset, 
                        cfg: DictConfig, data_root: Path):

    # df = pd.read_hdf('data/data_dump.h5', key='df')
    # dataset = EventFeatureDataset(root='data', filename='data_dump.h5')

    features = dataset[torch.arange(len(dataset))].x
    targets = dataset[torch.arange(len(dataset))].y # Onehot encoded
    targets = targets.reshape(len(dataset), -1)

    fig_labels, ax_labels = plt.subplots()
    ax_labels.set_title("Labels")
    x=list(dataset.tag_to_id.keys())
    y=targets.sum(axis=0).numpy()
    sns.barplot(
        x=x,
        y=y, ax=ax_labels
        )
    ax_labels.set_xticklabels(ax_labels.get_xticklabels(), rotation=45, horizontalalignment='right')
    
    save_dir = data_root/Path(cfg.paths.outputs_dir,cfg.paths.vis_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_labels.savefig(save_dir/"labels.png")

    fig_wordcloud, ax_wordcloud = plt.subplots()
    ax_wordcloud.set_title("Wordcloud")
    tags = ", ".join([t for tags in df['tag_name'] for t in tags])
    wordcloud = WordCloud(
        background_color='black',
        stopwords=STOPWORDS,
        max_words=100,
        max_font_size=50, 
        random_state=42
        ).generate(tags)

    ax_wordcloud.imshow(wordcloud)
    # fig_wordcloud.savefig("wordcloud.png")
    fig_wordcloud.savefig(save_dir/"wordcloud.png")
    
    print("current working directory: ", os.getcwd())
    print("save_dir: ", save_dir)
    print("contents of save_dir: ", os.listdir(save_dir))

    print("Saved visualizations to: ", save_dir)