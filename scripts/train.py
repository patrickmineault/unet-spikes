import lightning as pl
from lightning.pytorch import loggers as pl_loggers

from src import unet, unet_module

if __name__ == "__main__":
    logger = pl_loggers.TensorBoardLogger("lightning_logs", name="unet")

    net = unet.UNet1D(4, 29, 5)
    net.set_baseline_rate(0.2)
    model = unet_module.UnetLitModule(net, lr=1e-2)
    trainer = pl.Trainer(log_every_n_steps=10, max_epochs=50, logger=[logger])
    trainer.fit(model)
