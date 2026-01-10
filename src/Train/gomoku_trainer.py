#!/usr/bin/env python3

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Train.my_torch.Network import Network
from src.Train.data_set.verify_data import verify_dataset
from src.Train.PSQ_parser import PSQParser

ERROR_CODE = 84
NN_FILE = "gomoku_model.nn"

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

        input_size = 1200
        shared_size1 = 512
        shared_size2 = 256
        policy_size = 400
        value_size = 1

        weights = []
        biases = []

        w_shared1 = np.random.randn(input_size, shared_size1) * np.sqrt(2.0/input_size)
        b_shared1 = np.zeros((1, shared_size1))
        weights.append(w_shared1)
        biases.append(b_shared1)

        w_shared2 = np.random.randn(shared_size1, shared_size2) * np.sqrt(2.0/shared_size1)
        b_shared2 = np.zeros((1, shared_size2))
        weights.append(w_shared2)
        biases.append(b_shared2)

        w_policy = np.random.randn(shared_size2, policy_size) * np.sqrt(2.0/shared_size2)
        b_policy = np.zeros((1, policy_size))
        weights.append(w_policy)
        biases.append(b_policy)

        w_value = np.random.randn(shared_size2, value_size) * np.sqrt(2.0/shared_size2)
        b_value = np.zeros((1, value_size))
        weights.append(w_value)
        biases.append(b_value)

        network = Network()
        network.createLayer(weights, biases)

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
