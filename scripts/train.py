import lightning as pl

from src import unet, unet_module

if __name__ == "__main__":
    net = unet.UNet1D(1, 50, 2)
    model = unet_module.UnetLitModule(net)
    trainer = pl.Trainer(log_every_n_steps=10, max_epochs=10)
    trainer.fit(model)
