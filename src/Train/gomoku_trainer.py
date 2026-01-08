#!/usr/bin/env python3

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Train.my_torch.Network import Network

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

def he_normal(layer_size, seed=0):
    weights = []
    np.random.seed(seed)
    L = len(layer_size)
    for l in range(1, L):
        stddev = np.sqrt(2.0 / layer_size[l - 1])
        w = np.random.normal(0, stddev, (layer_size[l], layer_size[l - 1]))
        weights.append(w)
    return weights

def create_weights(init_method, layer_size):
    match init_method:
        case "he_normal":
            weights = he_normal(layer_size)
        case "xavier":
            weights = []
            L = len(layer_size)
            for l in range(1, L):
                stddev = np.sqrt(1.0 / layer_size[l - 1])
                w = np.random.normal(0, stddev, (layer_size[l], layer_size[l - 1]))
                weights.append(w)
        case "he_mixed_xavier":
            weights = []
            L = len(layer_size)
            for l in range(1, L):
                if l == L - 1:
                    stddev = np.sqrt(1.0 / layer_size[l - 1])
                else:
                    stddev = np.sqrt(2.0 / layer_size[l - 1])
                w = np.random.normal(0, stddev, (layer_size[l], layer_size[l - 1]))
                weights.append(w)
        case "random":
            weights = [
                np.random.rand(layer_size[l], layer_size[l - 1]) * 0.01
                for l in range(1, len(layer_size))
            ]
        case _:
            print(f"Unknown initialization method: {init_method}")
    return weights

def create_biases(layer_size):
    biases = [np.zeros((layer_size[l], 1)) for l in range(1, len(layer_size))]
    return biases


def main():
    """Main function to run the training or prediction process based on command-line arguments."""
    try:
        gofile = parse_arguments()
        X_data, Y_targets = parser.parse_file(gofile)

        if len(X_data) == 0:
            print("Error: No valid Go data.", file=sys.stderr)
            sys.exit(ERROR_CODE)

        indices = np.arange(len(X_data))
        np.random.shuffle(indices)
        X_data, Y_targets = X_data[indices], Y_targets[indices]

        split = int(0.8 * len(X_data))
        X_train, X_val = X_data[:split], X_data[split:]
        Y_train, Y_val = Y_targets[:split], Y_targets[split:]

        weights = create_weights("he_mixed_xavier", [400, 128, 400])
        biases = create_biases([400, 128, 400])

        network = Network()
        network.createLayer(weights, biases)

        network.train(
            NN_FILE,
            X_val=X_val,
            Y_val=Y_val,
            X_train=X_train,
            Y_train=Y_train,
        )

    except SystemExit:
        sys.exit(ERROR_CODE)


if __name__ == "__main__":
    main()
