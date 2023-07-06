import torchmetrics
import torch.optim
import torch.optim.lr_scheduler

import torch
from torch import nn
from lightning import LightningModule


class UnetLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["net"])

        self.net = net

        self.criterion = nn.PoissonNLLLoss(log_input=True, full=True)
        self.train_loss = torchmetrics.MeanMetric()

    def model_step(self, batch):
        X, y = batch
        y_hat = self.net(X)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def on_train_start(self):
        self.train_loss.reset()

    def forward(self, X: torch.Tensor):
        return self.net(X)

    def training_step(self, batch, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mean_preds", preds.mean())
        self.log("train/mean_targets", targets.mean())

        # return loss or backpropagation will fail
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters, self.hparams.lr, amsgrad=True)
        return {
            "optimizer": optimizer,
        }
