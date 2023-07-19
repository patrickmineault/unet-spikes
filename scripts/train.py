import lightning as pl
import torch
from lightning.pytorch import loggers as pl_loggers

from src import train_wrapper, unet

if __name__ == "__main__":
    # Note: to train on Mac, you will need to export the environment variable PYTORCH_ENABLE_MPS_FALLBACK=1
    logger = pl_loggers.TensorBoardLogger("lightning_logs", name="unet")

    net = unet.UNet1D(3, 29, 20).to(dtype=torch.float32)
    net.set_baseline_rate(0.2)
    model = train_wrapper.TrainWrapper(
        net,
        lr=1e-2,
        data_source="../data/config/lorenz.yaml",
        batch_size=256,
    ).to(dtype=torch.float32)
    train_wrapper = pl.Trainer(log_every_n_steps=10, max_epochs=50, logger=logger)
    train_wrapper.fit(model)
