import torch
import torch.optim
import torch.optim.lr_scheduler
import torchmetrics
from lightning import LightningModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from munch import munchify
from torch import nn
from torch.utils.data import DataLoader

from src import mask
from src.dataset import SpikesDataset


class UnetLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["net"])

        self.net = net

    def setup(self, stage: str):
        self.criterion = nn.PoissonNLLLoss(log_input=True, full=True)
        self.masker = mask.Masker(mask.MaskParams(), self.device)
        self.train_loss = torchmetrics.MeanMetric()
        self.train_dataset = SpikesDataset("../data/lorenz.yaml")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def model_step(self, batch):
        # Use a masked langueage model and predict the missing values
        X, _, _, _ = batch
        mask, X_true = self.masker.mask_batch(X)
        X_smoothed = self.net(X * mask)
        loss = self.criterion(X_smoothed[mask == 0], X_true[mask == 0])
        return loss, X_smoothed, X_true

    def on_train_start(self):
        self.train_loss.reset()

    def forward(self, X: torch.Tensor):
        return self.net(X)

    def training_step(self, batch, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mean_preds", preds.mean())
        self.log("train/mean_targets", targets.mean())
        self.log("train/std_preds", preds.std())
        self.log("train/std_targets", preds.mean())

        # return loss or backpropagation will fail
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.net.parameters(), self.hparams.lr, amsgrad=True
        )
        return {
            "optimizer": optimizer,
        }
