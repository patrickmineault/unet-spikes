import pytest
import torch

from src.mask import Masker, MaskMode


def test_rel_qty():
    """Check that the relative quantity of masked tokens and replaced
    tokens is as expected."""
    masker = Masker(MaskMode.neuron, 0.2)
    dummy = torch.zeros(1, 29, 49)
    dummy[0, 0, :] = 1
    dummy = dummy.to(torch.long)
    mask = masker(dummy)
    percentage_masked = (1.0 * (mask == 1)).mean()
    assert 0.1 < percentage_masked < 0.35


def test_dimension():
    """Check that the masking is taken along the expected dimensions."""
    for dim_type in [MaskMode.timestep, MaskMode.neuron]:
        masker = Masker(dim_type, 0.2)
        dummy = torch.zeros(2, 29, 49)
        dummy[0, 0, :] = 1
        dummy = dummy.to(torch.long)
        the_mask = masker(dummy)
        assert the_mask.shape == dummy.shape
        the_mask = the_mask[0, 1:, :]

        if dim_type == MaskMode.timestep:
            # Check that masking is along the right dimension
            cases = torch.mean(the_mask.to(torch.float).squeeze(), axis=0)  # type: ignore
            nbinary = ((cases == 1) | (cases == 0)).sum()
            assert nbinary == dummy.shape[2]
        elif dim_type == MaskMode.neuron:
            cases = torch.mean(the_mask.to(torch.float).squeeze(), axis=1)  # type: ignore
            nbinary = ((cases == 1) | (cases == 0)).sum()
            assert nbinary == dummy.shape[1] - 1
