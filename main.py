import os

import torch
import yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from network import ResNet

# import pytorch_lightning as pl
# from dataset import MNIST
# from network import Net, ResidualBlock
# from pytorch_lightning import Trainer, loggers
# from pytorch_lightning.callbacks import ModelCheckpoint


def get_config(config_file: str = './config.yml') -> edict:
    """
    Load the parameters from the config file given in parameters.

    In:
        config_file (str)

    Out:
        (edict) with the config parameters
    """
    with open(config_file) as file_:
        cfg = yaml.load(file_, Loader=yaml.FullLoader)

    return edict(cfg)


def main():
    cfg = get_config()

    for i in range(1, 5):
        net = ResNet(cfg, **cfg.net)

        ######
        # Train
        ######
        filename = f'save_{cfg.net.img_size}X{cfg.net.img_size}.pth'
        torch.save(net.state_dict(),
                   os.path.join(cfg.root_model, filename))

        cfg.net.layers.extend([2])
        cfg.net.img_size *= 2

    print(cfg)


if __name__ == '__main__':
    main()
