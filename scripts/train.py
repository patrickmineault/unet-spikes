import os.path

import torch
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src import mask, unet
from src.dataset import DATASET_MODES, SpikesDataset


def model_step(net, criterion, criterion_base, masker, batch, device, masking=True):
    X, rate, _, _ = batch
    X = X.to(device)
    rate = rate.to(device)
    if masking:
        X_masked, labels = masker.mask_batch(X)
    else:
        X_masked = X.clone()
        labels = torch.ones_like(X) * mask.UNMASKED_LABEL

    masked = labels != mask.UNMASKED_LABEL
    X_smoothed = net((X_masked).to(torch.float32))
    loss = criterion(X_smoothed[masked], X[masked])
    loss_base = criterion_base(X[masked], X[masked])
    return loss - loss_base, X_smoothed, X, masked, rate


def log_metrics(preds, targets, mask, logger, prefix, epoch):
    assert targets.min() >= 0, "Negative targets"
    preds_exp = torch.exp(preds)

    logger.add_scalar(f"{prefix}/loss", loss, epoch)
    logger.add_scalar(f"{prefix}/mean_preds", preds_exp.mean(), epoch)
    logger.add_scalar(f"{prefix}/std_preds", preds_exp.std(), epoch)
    logger.add_scalar(f"{prefix}/mean_targets", targets.to(torch.float32).mean(), epoch)
    logger.add_scalar(f"{prefix}/std_targets", targets.to(torch.float32).std(), epoch)
    logger.add_scalar(f"{prefix}/mean_mask", mask.to(torch.float32).mean(), epoch)


if __name__ == "__main__":
    data_source = "../data/config/lorenz.yaml"
    num_epochs = 50  # or the number of epochs you want to train for
    learning_rate = 1e-3  # or the learning rate you want to use

    # Instantiate your model here
    net = unet.UNet1D(3, 29, 10).to(dtype=torch.float32)
    net.set_baseline_rate(0.2)

    logger = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # M1 Mac-specific
    if device == torch.device("cpu") and torch.backends.mps.is_available():
        device = torch.device("mps")

    net = net.to(device)
    criterion = nn.PoissonNLLLoss(log_input=True, full=False)
    criterion_base = nn.PoissonNLLLoss(log_input=False, full=False)
    masker = mask.Masker(mask.MaskParams(), device)
    train_dataset = SpikesDataset(data_source)
    val_dataset = SpikesDataset(data_source, DATASET_MODES.val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, amsgrad=True)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        net.train()

        # Train loop
        for batch_num, batch in tqdm(enumerate(train_loader), desc="Train Batch"):
            optimizer.zero_grad()
            loss, preds, targets, the_mask, rate = model_step(
                net, criterion, criterion_base, masker, batch, device
            )
            loss.backward()
            optimizer.step()

            total_epoch = epoch * len(train_loader) + batch_num
            log_metrics(preds, targets, the_mask, logger, "train", total_epoch)

        # Validate loop
        net.eval()
        with torch.no_grad():
            for batch_num, batch in tqdm(enumerate(val_loader), desc="Val Batch"):
                total_epoch = epoch * len(val_loader) + batch_num
                loss, preds, targets, the_mask, rate = model_step(
                    net, criterion, criterion_base, masker, batch, device, False
                )
                log_metrics(preds, targets, the_mask, logger, "val", total_epoch)

                if rate is not None:
                    _, preds, targets, the_mask, rate = model_step(
                        net, criterion, criterion_base, masker, batch, device, False
                    )
                    r2 = (
                        torch.corrcoef(torch.stack([preds.ravel(), rate.ravel()]))[0, 1]
                        ** 2
                    )
                    logger.add_scalar("val/r2", r2, total_epoch)

        # Save model
        torch.save(net.state_dict(), os.path.join(logger.get_logdir(), f"model.pt"))
