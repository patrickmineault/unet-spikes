import argparse
import os.path
import warnings

import torch
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src import cnn, mask
from src.dataset import DATASET_MODES, SpikesDataset


def model_step(net, criterion, masker, batch, device, masking=True):
    X, rate, _, _ = batch
    X = X.to(device)
    rate = rate.to(device)
    if masking:
        the_mask = masker(X)
    else:
        the_mask = torch.zeros_like(X)
    the_mask = the_mask.to(device)

    assert X.shape == the_mask.shape
    assert the_mask.sum() < 0.5 * the_mask.numel()

    # Replace masked values with 0
    X_masked = X * (1 - the_mask.to(torch.float32))
    assert X_masked.sum() <= X.sum()

    X_smoothed = net((X_masked).to(dtype=torch.float32, device=device))
    loss = criterion(
        X_smoothed[the_mask].to(dtype=torch.float32, device=device),
        X[the_mask].to(dtype=torch.float32, device=device),
    )
    return loss, X_smoothed, X, the_mask, rate


def get_best_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use M1 Mac if available
    if device == torch.device("cpu") and torch.backends.mps.is_available():
        device = torch.device("mps")
        warnings.warn(
            "Using M1 Mac GPU. You may need to `export PYTORCH_ENABLE_MPS_FALLBACK=1`"
        )

    return device


def log_metrics(preds, targets, mask, logger, prefix, epoch):
    assert targets.min() >= 0, "Negative targets"

    logger.add_scalar(f"{prefix}/loss", loss, epoch)
    logger.add_scalar(f"{prefix}/mean_preds", preds.mean(), epoch)
    logger.add_scalar(f"{prefix}/std_preds", preds.std(), epoch)
    logger.add_scalar(f"{prefix}/mean_targets", targets.to(torch.float32).mean(), epoch)
    logger.add_scalar(f"{prefix}/std_targets", targets.to(torch.float32).std(), epoch)
    logger.add_scalar(f"{prefix}/mean_mask", mask.to(torch.float32).mean(), epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--data", type=str, default="../data/config/lorenz.yaml")
    args = parser.parse_args()

    # Instantiate your model here
    net = cnn.CNN(29, 10)

    logger = SummaryWriter()
    device = get_best_device()

    net = net.to(device)
    criterion = nn.MSELoss(reduce=True)
    masker = mask.Masker()
    train_dataset = SpikesDataset(args.data)
    val_dataset = SpikesDataset(args.data, DATASET_MODES.val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        net.train()

        # Train loop
        for batch_num, batch in tqdm(enumerate(train_loader), desc="Train Batch"):
            optimizer.zero_grad()
            loss, preds, targets, the_mask, rate = model_step(
                net, criterion, masker, batch, device
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
                    net, criterion, masker, batch, device, False
                )
                log_metrics(preds, targets, the_mask, logger, "val", total_epoch)

                if rate is not None:
                    _, preds, targets, the_mask, rate = model_step(
                        net, criterion, masker, batch, device, False
                    )
                    r2 = (
                        torch.corrcoef(torch.stack([preds.ravel(), rate.ravel()]))[0, 1]
                        ** 2
                    )
                    logger.add_scalar("val/r2", r2, total_epoch)

        # Save model
        torch.save(net.state_dict(), os.path.join(logger.get_logdir(), f"model.pt"))
