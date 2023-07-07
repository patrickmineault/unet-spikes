import torch
import torch.optim
import torch.optim.lr_scheduler
import torchmetrics
from einops import rearrange
from lightning import LightningModule
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

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
        self.criterion = nn.PoissonNLLLoss(
            log_input=True,
            full=False,
        )
        self.criterion_base = nn.PoissonNLLLoss(
            log_input=False,
            full=False,
        )
        self.masker = mask.Masker(mask.MaskParams(), self.device)
        self.train_loss = torchmetrics.MeanMetric()
        self.train_dataset = SpikesDataset("../data/lorenz.yaml")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=32, shuffle=False)

    def model_step(self, batch):
        # Use a masked langueage model and predict the missing values
        X, rate, _, _ = batch
        X = X[:, :-1, :]
        rate = rate[:, :-1, :]
        _, X_masked = self.masker.mask_batch(X)

        # TODO(pmin): swallow this in the Data Loader
        X = rearrange(X, "batch time neurons -> batch neurons time")
        X_masked = rearrange(X_masked, "batch time neurons -> batch neurons time")
        rate = rearrange(rate, "batch time neurons -> batch neurons time")

        assert X.shape[0] == 32  # Batch size.
        assert X.shape[1] == 29  # Number of neurons.
        # The rest should be time, which can be variable.

        removed = X_masked >= 0
        X_smoothed = self.net((X * (1 - 1 * removed)).to(torch.float32))
        loss = self.criterion(X_smoothed[removed], X[removed])
        loss_base = self.criterion_base(X[removed], X[removed])
        return loss - loss_base, X_smoothed, X, removed, rate

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
                torch.fmax(
                    torch.fmin(torch.exp(rate[0]), torch.Tensor([1.0])),
                    torch.Tensor([0.0]),
                ),
                self.current_epoch,
                dataformats="HW",
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.net.parameters(), self.hparams.lr, amsgrad=True
        )
        return {
            "optimizer": optimizer,
        }

    def get_tb(self) -> SummaryWriter | None:
        for logger in self.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                return logger.experiment
