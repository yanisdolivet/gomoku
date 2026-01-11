#!/usr/bin/env python3

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Train.my_torch.NetworkTorch import NetworkTorch
from src.Train.data_set.verify_data import verify_dataset
from src.Train.PSQ_parser import PSQParser

ERROR_CODE = 84
NN_FILE = "gomoku_model.bin"

def parse_arguments():
    """Parse command-line arguments and return the mode and file paths.
    Returns:
        tuple: (is_train (bool), loadfile (str), gofolder (str), savefile (str or None))
    """
    parser = argparse.ArgumentParser(
        usage="./gomoku_trainer.py gofolder1 gofolder2 ...",
    )
    parser.add_argument("gofolders", nargs='+', type=str, help="One or more folders containing Go game data")
    args = parser.parse_args()
    return args.gofolders

def main():
    """Main function to run the training or prediction process based on command-line arguments."""
    try:
        gofolders = parse_arguments()
        parser = PSQParser(board_size=20)
        X_data, Y_policy, Y_value = parser.load_dataset(gofolders)

        verify_dataset(X_data)

        if len(X_data) == 0:
            print("Error: No valid Go data.", file=sys.stderr)
            sys.exit(ERROR_CODE)

        indices = np.arange(len(X_data))
        np.random.shuffle(indices)
        X_data, Y_policy, Y_value = X_data[indices], Y_policy[indices], Y_value[indices]
        split = int(0.8 * len(X_data))
        X_train, X_val = X_data[:split], X_data[split:]
        Y_policy_train, Y_policy_val = Y_policy[:split], Y_policy[split:]
        Y_value_train, Y_value_val = Y_value[:split], Y_value[split:]
        network = NetworkTorch()
        
        network.train(
            NN_FILE,
            X_val=X_val,
            X_train=X_train,
            Y_value_val=Y_value_val,
            Y_value_train=Y_value_train,
            Y_policy_val=Y_policy_val,
            Y_policy_train=Y_policy_train,
        )

    except SystemExit:
        sys.exit(ERROR_CODE)


if __name__ == "__main__":
    main()
