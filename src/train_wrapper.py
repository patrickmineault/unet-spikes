from __future__ import annotations

import numpy as np
import sklearn
import sklearn.metrics
import torch
import torch.optim
import torch.optim.lr_scheduler
import torchmetrics
from lightning import LightningModule
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from src import mask
from src.dataset import DATASET_MODES, SpikesDataset


class Exponential(nn.Module):
    """For consistency with other inverse link functions, we implement this as a module rather than as a lambda."""

    def forward(self, x):
        return torch.exp(x)


canonical_links = {
    "mse": "identity",
    "poisson": "exp",
}


class TrainWrapper(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        data_source: str,
        lr: float = 1e-3,
        ilink: str = "canonical",
        beta: int = 1,
        masking_strategy: str = "zeros",
        loss: str = "mse",
        batch_size: int = 32,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["net"])

        self.net = net
        self.data_source = data_source

        if ilink == "canonical":
            ilink = canonical_links[loss]

        if ilink == "identity":
            self.nl = nn.Identity()
        elif ilink == "exp":
            self.nl = Exponential()
        elif ilink == "softplus":
            self.nl = nn.Softplus(beta=beta)
        else:
            raise NotImplementedError("Unknown inverse link function")

        if loss == "mse":
            self.criterion = nn.MSELoss(reduction="mean")
        elif loss == "poisson":
            self.criterion = nn.PoissonNLLLoss(
                log_input=False, full=True, reduction="mean"
            )
        else:
            raise NotImplementedError(f"Unknown loss function {loss}")

        self.masking_strategy = masking_strategy
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.masker = mask.Masker()
        self.train_loss = torchmetrics.MeanMetric()
        self.train_dataset = SpikesDataset(self.data_source)
        self.val_r2 = torchmetrics.MeanMetric()
        self.val_dataset = SpikesDataset(self.data_source, DATASET_MODES.val)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def model_step(self, batch, masking: bool = True):
        X, rate, _, _ = batch
        if masking:
            the_mask = self.masker(X)
        else:
            the_mask = torch.zeros_like(X)
        the_mask = the_mask.to(device=self.device)

        assert X.shape == the_mask.shape
        assert the_mask.sum() < 0.5 * the_mask.numel()

        # Replace masked values with 0
        if self.masking_strategy == "zeros":
            X_masked = X * (1 - the_mask.to(torch.float32))
            assert X_masked.sum() <= X.sum()
        elif self.masking_strategy == "replace":
            # Forward and backward masking
            the_mask = torch.zeros_like(X).to(torch.bool)
            mask_copy = the_mask.clone()
            X_masked = X.clone() * (1 - mask_copy.to(torch.float32))
            while mask_copy.any():
                # Forward masking
                X_masked = X_masked * (1 - mask_copy.to(torch.float32)) + torch.roll(
                    X_masked, 1, dims=2
                ) * mask_copy.to(torch.float32)
                mask_copy = torch.roll(mask_copy, 1, dims=2) & mask_copy
                # Backward masking
                X_masked = X_masked * (1 - mask_copy.to(torch.float32)) + torch.roll(
                    X_masked, -1, dims=2
                ) * mask_copy.to(torch.float32)
                mask_copy = torch.roll(mask_copy, -1, dims=2) & mask_copy

            assert X_masked.sum() <= X.sum()
        else:
            raise NotImplementedError("Unknown masking strategy")

        X_smoothed = self.net((X_masked).to(dtype=torch.float32))
        X_smoothed = self.nl(X_smoothed)
        loss = self.criterion(
            X_smoothed[the_mask].to(dtype=torch.float32),
            X[the_mask].to(dtype=torch.float32),
        )
        return loss, X_smoothed, X, the_mask, rate

    def on_train_start(self):
        self.train_loss.reset()

    def forward(self, X: torch.Tensor):
        return self.net(X)

    def training_step(self, batch, batch_idx: int):
        self.last_step = self.model_step(batch)

        loss, preds, targets, mask, rate = self.last_step

        assert targets.min() >= 0, "Negative targets"

        preds_exp = torch.exp(preds)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        self.log("train/mean_preds", preds_exp.mean())
        self.log("train/std_preds", preds_exp.std())
        self.log("train/mean_targets", targets.to(torch.float32).mean())
        self.log("train/std_targets", targets.to(torch.float32).std())
        self.log("train/mean_mask", mask.to(torch.float32).mean())

        # return loss or backpropagation will fail
        return loss

    def on_validation_epoch_start(self) -> None:
        self.preds_unmasked = []
        self.rates_unmasked = []

    def validation_step(self, batch, batch_idx: int):
        loss, preds, _, _, rate = self.model_step(batch, masking=False)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if rate is not None:
            # Figure out whether we can predict the rate well
            _, preds, targets, mask, rate = self.model_step(batch, masking=False)
            self.preds_unmasked.append(preds.detach().cpu().numpy())
            self.rates_unmasked.append(rate.detach().cpu().numpy())

        return loss

    def on_validation_epoch_end(self):
        preds_unmasked = np.concatenate(self.preds_unmasked, axis=0)
        rates_unmasked = np.concatenate(self.rates_unmasked, axis=0)
        r2s = []
        for i in range(preds_unmasked.shape[1]):
            r2s.append(
                sklearn.metrics.r2_score(
                    rates_unmasked[:, i, :].ravel(), preds_unmasked[:, i, :].ravel()
                )
            )

        self.log("val/r2", np.mean(r2s).astype(np.float32))

    def on_train_epoch_end(self) -> None:
        # Show the last step
        loss, preds, targets, mask, rate = self.last_step
        tensorboard = self.get_tb()
        if tensorboard is not None:
            tensorboard.add_image(
                "debug/preds", torch.exp(preds[0]), self.current_epoch, dataformats="HW"
            )
            tensorboard.add_image(
                "debug/targets", targets[0], self.current_epoch, dataformats="HW"
            )
            tensorboard.add_image(
                "debug/mask", mask[0], self.current_epoch, dataformats="HW"
            )
            tensorboard.add_image(
                "debug/target_rates",
                torch.clamp(rate[0], 0, 1),
                self.current_epoch,
                dataformats="HW",
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.net.parameters(), self.hparams.lr, amsgrad=True  # type: ignore
        )
        return {
            "optimizer": optimizer,
        }

    def get_tb(self) -> SummaryWriter | None:
        for logger in self.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                return logger.experiment
