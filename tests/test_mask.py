import pytest
import torch

from src.mask import Masker, MaskParams, MaskMode, UNMASKED_LABEL

def test_rel_qty():
    """Check that the relative quantity of masked tokens and replaced
    tokens is as expected."""
    masker = Masker(MaskParams(MASK_RATIO=0.2), "cpu")
    dummy = torch.zeros(1, 29, 49)
    dummy[0, 0, :] = 1
    dummy = dummy.to(torch.long)
    batch, data_masked = masker.mask_batch(dummy)
    percentage_same = (1.0 * (dummy == batch)).mean()
    assert .95 < percentage_same < 1.0
    percentage_masked = (1.0 * (data_masked != UNMASKED_LABEL)).mean()
    assert 0.1 < percentage_masked < .35

def test_dimension():
    """Check that the masking is taken along the expected dimensions."""
    for dim_type in [MaskMode.timestep, MaskMode.neuron]:
        masker = Masker(MaskParams(MASK_MODE=dim_type), "cpu")
        dummy = torch.zeros(1, 29, 49)
        dummy[0, 0, :] = 1
        dummy = dummy.to(torch.long)
        _, data_masked = masker.mask_batch(dummy)
        assert data_masked.shape == dummy.shape
        data_masked = data_masked[:, 1:, :]

        if dim_type == MaskMode.timestep:
            # Check that masking is along the right dimension
            cases = torch.mean(data_masked.to(torch.float).squeeze(), axis=0)  # type: ignore
            nbinary = ((cases == -100) | (cases == 0)).sum()
            assert nbinary == dummy.shape[2]
        elif dim_type == MaskMode.neuron:
            cases = torch.mean(data_masked.to(torch.float).squeeze(), axis=1)  # type: ignore
            nbinary = ((cases == -100) | (cases == 0)).sum()
            assert nbinary == dummy.shape[1] - 1