import torch

from src.mask import Masker, MaskParams


def test_mask():
    masker = Masker(MaskParams(), "cpu")
    dummy = torch.zeros(1, 29, 49)
    dummy[0, 0, 0] = 1
    dummy = dummy.to(torch.long)
    _, data_masked = masker.mask_batch(dummy)
    assert data_masked.shape == dummy.shape
    cases = torch.mean(data_masked.to(torch.float).squeeze(), axis=0)
    print(data_masked[0, :, :])
    nbinary = ((cases == -100) | (cases == 0)).sum()
    assert nbinary == dummy.shape[2]
