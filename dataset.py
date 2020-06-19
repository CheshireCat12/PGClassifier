from typing import Dict, Tuple

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST


class DatasetFactory:

    # (dataset, in_channels, nb_classes, img_size, mean, std)
    DATASETS: Dict[str, Tuple[Dataset, int, int, int]] = {
        'mnist': (MNIST, 1, 10, 28, [0.5, ], [0.5, ]),
        'cifar10': (CIFAR10, 3, 10, 32,
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        'cifar100': (CIFAR100, 3, 100, 32,
                     (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    }

    def __init__(self, cfg):
        self._set_attribute_from_cfg(cfg)
        self.prepare_dataset()

    def _set_attribute_from_cfg(self, cfg):
        """
        set the attributes from cfg.data to this class
        """
        for new_attr, value in cfg.data.items():
            setattr(self, new_attr, value)

    def prepare_dataset(self):
        assert self.dataset_name in self.DATASETS.keys(
        ), f'{self.dataset_name} is not available!'

        self.dataset, self.img_size, self.in_channels, self.out_channels, self.mean, self.std = self.DATASETS[
            self.dataset]
        self.rectify_img_size()
        self.dataset(self.root_data, train=True, download=True)
        self.dataset(self.root_data, train=False, download=True)

    def rectify_img_size(self):
        if self.new_img_size < self.img_size:
            self.img_size = self.new_img_size

    def train_loader(self):
        transform = transforms.Compose([
            transforms.RandomAffine(30, translate=(0.1, 0.1)),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        data_train = self.dataset(self.root_data, train=True,
                                  transform=transform, download=False)

        return DataLoader(data_train, batch_size=self.batch_size_train, shuffle=True, num_workers=8)

    def val_loader(self):
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        data_val = self.dataset(self.root_data, train=False,
                                transform=transform, download=False)

        return DataLoader(data_val, batch_size=self.batch_size_val, shuffle=False, num_workers=8)
