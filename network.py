from dataset import DatasetFactory
import os
from math import ceil
from typing import List, Dict

import torch
import torch.nn as nn


def conv3x3(in_channels: int, out_channels: int, stride: int = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                     stride=(stride, stride), padding=(1, 1))


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, nb_layers: int = 3,
                 downsample: bool = True) -> None:
        super(ResidualBlock, self).__init__()
        self.downsample = conv3x3(
            in_channels, out_channels, stride=2) if downsample else lambda x: x

        layers = []
        for i in range(nb_layers):
            stride = 1 if i + 1 < nb_layers and downsample else 2
            layers.extend([conv3x3(in_channels, out_channels, stride=stride),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU()])
            in_channels = out_channels

        self.block = nn.Sequential(*layers)

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.downsample(x)
        output = self.block(x)
        output += residual
        output = self.relu(output)

        return output


class BasicConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, nb_layers: int = 3) -> None:
        super(BasicConvBlock, self).__init__()
        # Create the layers
        layers = []
        for i in range(nb_layers):
            stride = 1 if i + 1 < nb_layers else 2
            layers.extend([nn.Conv2d(in_channels, out_channels,
                                     kernel_size=3, stride=stride, padding=1),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU()])
            in_channels = out_channels

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BasicMLPBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, dropout: int = 0.5,
                 last: bool = False) -> None:
        super(BasicMLPBlock, self).__init__()

        layer = [nn.Dropout(p=dropout),
                 nn.Linear(in_channels, out_channels), ]
        if not last:
            layer += [nn.BatchNorm1d(out_channels),
                      nn.ReLU(), ]
        self.block = nn.Sequential(*layer)

    def forward(self, x):
        return self.block(x)


class ResNet(nn.Module):

    _BLOCK = {'ResidualBlock': ResidualBlock}

    def __init__(self, cfg, block: str, layers: List[int], img_size: int) -> None:
        super(ResNet, self).__init__()
        self.block = self._BLOCK[block]
        self.size_img = img_size
        self.root_model = cfg.root_model
        _, in_channels, num_classes, * \
            _ = DatasetFactory.DATASETS[cfg.data.dataset]

        self.create_conv_layers(self.block, layers, in_channels)

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            BasicMLPBlock(self.classifer_dim, 512),
            BasicMLPBlock(512, 512),
            BasicMLPBlock(512, 256),
            BasicMLPBlock(256, num_classes, last=True)
        )

        self.load_pretrained_dict()

    def load_pretrained_dict(self):
        last_img_size = self.size_img // 2
        load_file = os.path.join(
            self.root_model, f'save_{last_img_size}X{last_img_size}.pth')
        print(load_file)
        if os.path.exists(load_file):
            print('Load pretrained network.')
            pretrained_model = torch.load(load_file)
            pretrained_dict = {
                k: v for k, v in pretrained_model.items() if 'classifier.1' not in k}
            self.load_state_dict(pretrained_dict, strict=False)

    def create_conv_layers(self, block: nn.Module, layers: List[int],
                           in_channels: int) -> None:
        size_img_out = ceil(self.size_img / 2**(len(layers)))
        assert size_img_out > 2,\
            'The image size is too small for the number of given layers'
        out_channels = 16
        blocks = []
        for nb_layer in layers:
            blocks.append(block(in_channels, out_channels, nb_layer))
            in_channels = out_channels
            out_channels *= 2

        self.features = nn.Sequential(*blocks)
        self.classifer_dim = size_img_out**2 * (out_channels//2)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)

        return output


def main():
    params = {'layers': [2],
              'img_size': 8}
    for i in range(1, 5):
        net = ResNet(ResidualBlock, **params)

        # load pretrained network
        if i > 1:
            pretrained_model = torch.load(
                f'./save_{params["img_size"]//2}X{params["img_size"]//2}.pth')
            pretrained_dict = {
                k: v for k, v in pretrained_model.items() if 'classifier.1' not in k}
            net.load_state_dict(pretrained_dict, strict=False)

        ######
        # Train
        ######
        torch.save(net.state_dict(),
                   f'./save_{params["img_size"]}X{params["img_size"]}.pth')

        params['layers'].extend([2])
        params['img_size'] *= 2


if __name__ == '__main__':
    main()
