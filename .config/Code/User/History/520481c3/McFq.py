import pandas as pd
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from wordcloud import WordCloud, STOPWORDS

def visualize_dataset(df: pd.DataFrame, dataset: EventFeatureDataset, cfg: DictConfig):

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
    fig_labels.savefig(cfg.paths.outputs_dir/"visualizations"/"labels.png")

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
    fig_wordcloud.savefig(cfg.paths.outputs_dir/"visualizations"/"wordcloud.png")