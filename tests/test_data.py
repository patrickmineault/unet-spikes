from pathlib import Path

from src.dataset import SpikesDataset, merge_config

def test_merge():
    base = {'a': 1,
            'b': {'a': 0, 'b': 1}
           }
    delta = {'a': 2,
             'b': {'b': 2, 'c': 2}
            }
    changed = merge_config(base, delta)
    assert changed['a'] == 2
    assert changed['b']['a'] == 0
    assert changed['b']['b'] == 2
    assert changed['b']['c'] == 2

def test_mc_maze():
    root_dir = Path(__file__).absolute().parent
    dataset = SpikesDataset(root_dir / ".." / "data" / "config" / "mc_maze.yaml")
    (spikes, rates, heldout_spikes, forward_spikes) = dataset[0]
    assert spikes.ndim == 2
    assert spikes.shape[0] == 29
    assert spikes.shape[1] == 50


def test_lorenz():
    root_dir = Path(__file__).absolute().parent
    dataset = SpikesDataset(root_dir / ".." / "data" / "config" / "lorenz.yaml")
    (spikes, rates, heldout_spikes, forward_spikes) = dataset[0]
    assert spikes.ndim == 2
    assert spikes.shape[0] == 29
    assert spikes.shape[1] == 50


def test_chaotic():
    root_dir = Path(__file__).absolute().parent
    dataset = SpikesDataset(root_dir / ".." / "data" / "config" / "chaotic.yaml")
    (spikes, rates, heldout_spikes, forward_spikes) = dataset[0]
    assert spikes.ndim == 2
    assert spikes.shape[0] == 50
    assert spikes.shape[1] == 100
