from pathlib import Path

from src.dataset import SpikesDataset


def test_shape():
    root_dir = Path(__file__).absolute().parent
    dataset = SpikesDataset(root_dir / ".." / "data" / "lorenz.yaml")
    (spikes, rates, heldout_spikes, forward_spikes) = dataset[0]
    assert spikes.ndim == 2
    assert spikes.shape[0] == 29
    assert spikes.shape[1] % 2 == 1