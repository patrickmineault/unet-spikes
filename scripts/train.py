import lightning as pl
from lightning.pytorch import loggers as pl_loggers
import os
import torch

from src import unet, unet_module

if __name__ == "__main__":
    # Note: to train on Mac, you will need to export the environment variable PYTORCH_ENABLE_MPS_FALLBACK=1
    logger = pl_loggers.TensorBoardLogger("lightning_logs", name="unet")

    net = unet.UNet1D(3, 50, 20).to(dtype=torch.float32)
    net.set_baseline_rate(0.2)
    model = unet_module.UnetLitModule(net, lr=1e-2, data_source="../data/config/chaotic.yaml").to(dtype=torch.float32)
    trainer = pl.Trainer(log_every_n_steps=10, max_epochs=50, logger=logger)
    trainer.fit(model)
