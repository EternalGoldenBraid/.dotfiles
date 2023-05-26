from pathlib import Path
import torch

import hydra
from omegaconf import DictConfig

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):



if __name__ == '__main__':
    main()