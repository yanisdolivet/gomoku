#!/usr/bin/env python3

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Train.my_torch.Network import Network
from src.Train.data_set.verify_data import verify_dataset

ERROR_CODE = 84
NN_FILE = "gomoku_model.nn"

def parse_arguments():
    """Parse command-line arguments and return the mode and file paths.
    Returns:
        tuple: (is_train (bool), loadfile (str), gofile (str), savefile (str or None))
    """
    parser = argparse.ArgumentParser(
        usage="./gomoku_trainer.py GOFILE\n"
    )
    parser.add_argument("GOFILE", type=str)
    args = parser.parse_args()
    return args.GOFILE

def parse_data(gofile):
    pass

def main():
    """Main function to run the training or prediction process based on command-line arguments."""
    try:
        gofile = parse_arguments()
        X_data, Y_policy, Y_value = parse_data(gofile)

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

        input_size = 400
        hidden_size = 128
        policy_size = 400
        value_size = 1

        weights = []
        biases = []

        w_shared = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        b_shared = np.zeros((1, hidden_size))
        weights.append(w_shared)
        biases.append(b_shared)

        w_policy = np.random.randn(hidden_size, policy_size) * np.sqrt(2.0/hidden_size)
        b_policy = np.zeros((1, policy_size))
        weights.append(w_policy)
        biases.append(b_policy)

        w_value = np.random.randn(hidden_size, value_size) * np.sqrt(2.0/hidden_size)
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
