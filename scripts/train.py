import lightning as pl
from lightning.pytorch import loggers as pl_loggers

from src import unet, unet_module

if __name__ == "__main__":
    logger = pl_loggers.TensorBoardLogger("lightning_logs", name="unet")

    net = unet.UNet1D(1, 29, 2)
    net.set_baseline_rate(0.2)
    model = unet_module.UnetLitModule(net)
    trainer = pl.Trainer(log_every_n_steps=10, max_epochs=10, logger=[logger])
    trainer.fit(model)
