from typing import Dict

import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler

import pytorch_lightning as pl
from dataset import DatasetFactory
from network import ResNet


class TrainNet(pl.LightningModule):

    def __init__(self, cfg) -> None:
        super(TrainNet, self).__init__()
        self.cfg = cfg
        self.dataset = DatasetFactory(cfg)
        self.net = ResNet(cfg, **cfg.net)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def prepare_data(self):
        self.dataset.prepare_dataset()

    def train_dataloader(self):
        return self.dataset.train_loader()

    def val_dataloader(self):
        return self.dataset.val_loader()

    def configure_optimizers(self):
        optimizer = SGD(self.net.parameters(),
                        **self.cfg.training.optimizer)
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, **self.cfg.training.scheduler)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        self.train()
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        predicted_labels = logits.argmax(dim=1)
        accuracy = (predicted_labels == y).sum().item() / len(y)

        tensorboard_logs = {'train_loss': loss,
                            'train_acc': accuracy}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        self.eval()
        x, y = batch
        y_hat = self(x)
        predicted_labels = y_hat.argmax(dim=1)
        accuracy = (predicted_labels == y).sum().float() / len(y)

        return {'val_loss': self.loss_fn(y_hat, y), 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': tensorboard_logs}
