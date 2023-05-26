from pathlib import Path
import torch

import hydra
from omegaconf import DictConfig

from src.slaps.train_slaps_frozen import Experiment as SlapsFrozenExperiment

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):



if __name__ == '__main__':
    main()