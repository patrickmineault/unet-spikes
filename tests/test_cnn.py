import torch

from src.cnn import CNN


def test_cnn():
    cnn = CNN(29, 3)
    X = torch.rand(2, 29, 1000)
    cnn.forward(X)
    assert X.shape == (2, 29, 1000)
