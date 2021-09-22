import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import DictConfig
from torch import nn


class MyResnet(pl.LightningModule):
    def __init__(self, cnf: DictConfig, num_classes: int):
        super().__init__()
        self.cnf = cnf
        self.resnet = self.make_model(num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def make_model(self, num_classes):
        # self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)  # Causes timeout error
        model = torchvision.models.resnet50(pretrained=True, progress=True)
        # print(model.fc) to see how many units we need.
        model.fc = nn.Linear(2048, num_classes)  # Replace the classification layer
        return model

    def forward(self, x):
        return self.resnet.forward(x)

    def _calc_loss(self, batch):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calc_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calc_loss(batch)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cnf.learning_rate)
