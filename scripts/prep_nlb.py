import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import (
    make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
)

def main(args):
    # Based on lfads_data_prep.py
    valid_ratio = 0.2
    bin_size_ms = 5
    suf = '' if bin_size_ms == 5 else '_20'

    # ---- Data locations ---- #
    data_root = Path(args.data_root)
    datapath_dict = {
        'area2_bump': data_root / '000127' / 'sub-Han',
        'mc_maze': data_root / '000128' / 'sub-Jenkins',
        'mc_rtt': data_root / '000129' / 'sub-Indy',
        'dmfc_rsg': data_root / '000130' / 'sub-Haydn',
        'mc_maze_large': data_root / '000138' / 'sub-Jenkins',
        'mc_maze_medium': data_root / '000139' / 'sub-Jenkins',
        'mc_maze_small': data_root / '000140' / 'sub-Jenkins',
    }
    prefix_dict = {
        'mc_maze': '*full',
        'mc_maze_large': '*large',
        'mc_maze_medium': '*medium',
        'mc_maze_small': '*small',
    }
    for dataset_name in datapath_dict.keys():
        datapath = datapath_dict[dataset_name]
        dataset = NWBDataset(datapath)

        # Prepare dataset
        phase = 'val'

        # Choose bin width and resample
        bin_width = 5
        dataset.resample(bin_width)

        # Create suffix for group naming later
        suffix = '' if (bin_width == 5) else f'_{int(bin_width)}'

        train_split = 'train' if (phase == 'val') else ['train', 'val']
        train_dict = make_train_input_tensors(
            dataset, dataset_name=dataset_name, trial_split=train_split, save_file=False,
            include_behavior=True,
            include_forward_pred = True,
        )

        # Show fields of returned dict
        print(train_dict.keys())

        # Unpack data
        train_spikes_heldin = train_dict['train_spikes_heldin']
        # train_spikes_heldout = train_dict['train_spikes_heldout']
        # Print 3d array shape: trials x time x channel
        print(train_spikes_heldin.shape)

        # Make data tensors - use all chunks including forward prediction for training NDT
        eval_dict = make_train_input_tensors(
            dataset, dataset_name=dataset_name, trial_split=['val'], save_file=False, include_forward_pred=True,
        )
        eval_dict = {
            f'eval{key[5:]}': val for key, val in eval_dict.items()
        }
        eval_spikes_heldin = eval_dict['eval_spikes_heldin']

        print(eval_spikes_heldin.shape)

        h5_dict = {
            **train_dict,
            **eval_dict
        }

        h5_target = Path(__file__).parent.parent / "data" / "h5" / f"{dataset_name}.h5"
        save_to_h5(h5_dict, h5_target, overwrite=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        required=True,
        help="Location of the data root",
    )

    args = parser.parse_args()
    main(args)