import argparse
import os.path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.metrics
import torch
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src import cnn, mask, unet
from src.dataset import DATASET_MODES, SpikesDataset


def model_step(
    net: nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    nl: nn.Module,
    masker: nn.Module,
    batch: tuple,
    device: torch.device,
    masking: bool = True,
    masking_strategy: str = "zeros",
):
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
    if masking_strategy == "zeros":
        X_masked = X * (1 - the_mask.to(torch.float32))
        assert X_masked.sum() <= X.sum()
    elif masking_strategy == "replace":
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

    X_smoothed = net((X_masked).to(dtype=torch.float32, device=device))
    X_smoothed = nl(X_smoothed)
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

    assert preds.mean() < 10


def show_debugging(preds, targets, rate, mask, logger, prefix, epoch, device):
    logger.add_image(f"debug_{prefix}/preds", preds[0], epoch, dataformats="HW")
    logger.add_image(f"debug_{prefix}/targets", targets[0], epoch, dataformats="HW")
    logger.add_image(f"debug_{prefix}/mask", mask[0], epoch, dataformats="HW")
    logger.add_image(
        f"debug_{prefix}/target_rates",
        torch.fmax(
            torch.fmin(
                rate[0],
                torch.Tensor([1.0]).to(device=device),
            ),
            torch.Tensor([0.0]).to(device=device),
        ),
        epoch,
        dataformats="HW",
    )
    logger.add_image(
        f"debug_{prefix}/rel_residuals",
        torch.fmax(
            torch.fmin(
                preds[0] / rate[0] - 0.5,
                torch.Tensor([1.0]).to(device=device),
            ),
            torch.Tensor([0.0]).to(device=device),
        ),
        epoch,
        dataformats="HW",
    )

    for name, rhs in (("targets", targets[0]), ("rates", rate[0])):
        plt.figure(figsize=(6, 4))
        m = torch.max(preds[0].max(), rhs.max()).detach().cpu().numpy()

        plt.plot([0, m], [0, m], "k-")
        plt.plot(
            preds[0].detach().cpu().numpy().ravel(),
            rhs.detach().cpu().numpy().ravel(),
            ".",
        )
        plt.xlabel("preds")
        plt.ylabel(name)

        logger.add_figure(
            f"debug_{prefix}/preds_v_{name}",
            plt.gcf(),
            epoch,
        )


class Exponential(nn.Module):
    """For consistency with other inverse link functions, we implement this as a module rather than as a lambda."""

    def forward(self, x):
        return torch.exp(x)


canonical_links = {
    "mse": "identity",
    "poisson": "exp",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--data", type=str, default="../data/config/lorenz.yaml")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "unet"])
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "poisson"],
        help="loss function",
    )
    parser.add_argument(
        "--ilink",
        type=str,
        default="canonical",
        choices=["canonical", "identity", "exp", "softplus"],
        help="inverse link function",
    )
    parser.add_argument("--beta", type=float, default=1.0, help="softplus beta")
    parser.add_argument("--latent_dim", type=int, default=10, help="latent dimension")
    parser.add_argument(
        "--masking",
        type=str,
        choices=["zeros", "replace"],
        default="zeros",
        help="Masking strategy",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup epochs")
    args = parser.parse_args()

    # Instantiate your model here
    if args.model == "cnn":
        net = cnn.CNN(29, args.latent_dim)
    elif args.model == "unet":
        net = unet.UNet1D(4, 29, args.latent_dim)
    else:
        raise NotImplementedError(f"Unknown model {args.model}")

    logger = SummaryWriter()
    device = get_best_device()

    net = net.to(device)

    ilink = args.ilink
    if ilink == "canonical":
        ilink = canonical_links[args.loss]

    if ilink == "identity":
        nl = nn.Identity()
    elif ilink == "exp":
        nl = Exponential()
    elif ilink == "softplus":
        nl = nn.Softplus(beta=args.beta)
    else:
        raise NotImplementedError(f"Unknown inverse link function {ilink}")

    print(nl)

    if args.loss == "mse":
        criterion = nn.MSELoss(reduction="mean")
    elif args.loss == "poisson":
        criterion = nn.PoissonNLLLoss(log_input=False, full=True, reduction="mean")
    else:
        raise NotImplementedError(f"Unknown loss function {args.loss}")

    net.unembedding.bias.data = np.log(0.25) * torch.ones_like(
        net.unembedding.bias.data
    )

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
                net, criterion, nl, masker, batch, device
            )
            loss.backward()
            optimizer.step()

            total_epoch = epoch * len(train_loader) + batch_num
            log_metrics(preds, targets, the_mask, logger, "train", total_epoch)

            if batch_num == len(train_loader) - 1:
                show_debugging(
                    preds, targets, rate, the_mask, logger, "train", total_epoch, device
                )

        # Validate loop
        net.eval()
        with torch.no_grad():
            preds_unmasked = []
            rates_unmasked = []
            for batch_num, batch in tqdm(enumerate(val_loader), desc="Val Batch"):
                total_epoch = epoch * len(val_loader) + batch_num

                loss, preds, targets, the_mask, rate = model_step(
                    net, criterion, nl, masker, batch, device, False
                )
                log_metrics(preds, targets, the_mask, logger, "val", total_epoch)

                # Same, for unmasked rates (i.e. pure smoothing mode).
                preds_unmasked.append(preds.cpu().numpy())
                rates_unmasked.append(rate.cpu().numpy())

                if batch_num == len(val_loader) - 1:
                    show_debugging(
                        preds,
                        targets,
                        rate,
                        the_mask,
                        logger,
                        "val",
                        total_epoch,
                        device,
                    )

            # Now calculate R2 as in the NDT paper: do it over all data,
            # then ravel batches and time, calculate R2 per neuron, then average.
            preds_unmasked = np.concatenate(preds_unmasked, axis=0)
            rates_unmasked = np.concatenate(rates_unmasked, axis=0)
            r2s = []
            for i in range(preds_unmasked.shape[1]):
                r2s.append(
                    sklearn.metrics.r2_score(
                        rates_unmasked[:, i, :].ravel(), preds_unmasked[:, i, :].ravel()
                    )
                )

            logger.add_scalar("val/r2", np.mean(r2s), (epoch + 1) * len(val_loader))
            logger.add_histogram(
                "val/r2s", np.array(r2s), (epoch + 1) * len(val_loader)
            )

        # Save model
        torch.save(net.state_dict(), os.path.join(logger.get_logdir(), f"model.pt"))
