
from pathlib import Path
import glob
import os
import hydra
from omegaconf import DictConfig

from src.utils import make_probs_movie

@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config):
    attention_save_dir = Path(config.visualize_dir, 'attention','val','images')
    attention_save_dir.mkdir(parents=True, exist_ok=True)
    make_probs_movie(path=attention_save_dir, config=config)

    
if __name__ == '__main__':
    main() 