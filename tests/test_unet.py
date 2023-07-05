import torch
from src import unet


def test_downsample():
    net = unet.DownsampleLayer(2, 4)
    X = torch.randn(1, 2, 101)
    Y = net(X)
    assert Y.shape == (1, 4, 51)


def test_upsample():
    net = unet.UpsampleLayer(4, 2)
    X = torch.randn(1, 4, 51)
    Y = net(X)
    assert Y.shape == (1, 2, 101)


def test_forward():
    net = unet.UNet1D(2, 10, 2)
    X = torch.randn(1, 10, 101)
    Y = net(X)
    assert X.shape == Y.shape
