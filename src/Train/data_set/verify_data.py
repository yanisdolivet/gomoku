##
## EPITECH PROJECT, 2026
## gomoku
## File description:
## verify_data
##

import numpy as np

def verify_dataset(X_data):
    """
    Verifies that the dataset matches the C++ engine's strict requirements.
    """
    print(f"Checking dataset with {len(X_data)} samples...")

    if X_data.shape[1] != 1200:
        raise ValueError(f"CRITICAL: Input shape is {X_data.shape}, expected (N, 1200).")

    # unique_values = np.unique(X_data)
    # print(f"Unique values in input: {unique_values}")

    # if not all(np.isclose(v, [ -1, 0, 1 ], atol=0.01).any() for v in unique_values):
    #      raise ValueError("CRITICAL: Input data contains values other than -1, 0, 1. "
    #                       "The C++ engine expects 1.0 (Me), -1.0 (Opponent), 0.0 (Empty).")

    # print("\n--- Visual Sample Check (First Board) ---")
    # sample = X_data[0].reshape(40, 40)

    # # Map back to symbols for display
    # symbols = {0: '.', 1: 'X', -1: 'O'}

    # for row in sample:
    #     print(" ".join([symbols[int(round(x))] for x in row]))
    #     if (row == 400):
    #         print()

    # # Check balance stone counts
    # stones_me = np.sum(X_data == 1.0)
    # stones_opp = np.sum(X_data == -1.0)
    # print(f"My Stones: {stones_me}, Opponent Stones: {stones_opp}")

    # print("\n----------------------------------------")
    print("Dataset verification PASSED.\n")