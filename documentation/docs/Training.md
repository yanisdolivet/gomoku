---
id: Training
title: The Machine Learning Pipeline
sidebar_position: 2
---

# The Machine Learning Pipeline

This document explains the end-to-end process of creating the "Brain" of our AI. It covers the data lifecycle, the mathematics of learning, and the high-performance computing stack used.

---

## 1. The Learning Paradigm: Supervised Learning

We train our AI using a **Supervised Learning** approach, specifically **Imitation Learning** (Behavioral Cloning).

### The Concept
Imagine observing a Grandmaster playing chess.
*   The Grandmaster sees the board (Input).
*   The Grandmaster makes a move (Target Output).
*   If we observe enough games, we can learn a function `f(Board) -> Move` that imitates the Grandmaster.

In our case:
*   **Dataset**: Thousands of high-level Gomoku games (from Renju players or self-play).
*   **Inputs**: The board state tensor (Is it Black's turn? Where are the stones?).
*   **Labels (Targets)**:
    1.  **Policy Target**: The move actually played in that situation.
    2.  **Value Target**: The final result of that game (Did Black win eventually?).

---

## 2. The Toolstack: PyTorch & Hardware Acceleration

We use **PyTorch**, the leading framework for Deep Learning research.
However, training a deep ResNet involves **billions** of floating-point operations (FLOPs). Doing this on a CPU is infeasible (it would take weeks).

### Apple Silicon Acceleration (MPS)
We exploit the **Metal Performance Shaders (MPS)** backend available on macOS.
*   **The M-Series Chips (M1/M2/M3/M4)** have a unified memory architecture and powerful Neural Engines / GPUs.
*   By moving our Tensors to the `mps` device (`tensor.to('mps')`), PyTorch offloads the matrix multiplications to the GPU.
*   **Result**: We achieve training speeds comparable to dedicated Nvidia GPUs, allowing us to iterate on model architecture in minutes rather than hours.

---

## 3. The Training Loop (Step-by-Step)

The training script `NetworkTorch.py` performs the following loop thousands of times:

### Step 1: Forward Pass
We feed a batch of board positions (e.g., 64 boards at once) into the network.
The network outputs its *guess*:
*   `Pred_Policy`: A probability distribution (e.g., 10% center, 0% corner).
*   `Pred_Value`: A score (e.g., +0.2).

### Step 2: Loss Calculation (The Error)
We compare the guess with the truth using **Loss Functions**.
*   **Policy Loss (Cross Entropy)**: Measures the distance between the predicted probability distribution and the actual move (which is a distribution with 100% on one move).
    *   Formula: `Loss = - sum(True_Prob * log(Pred_Prob))`
*   **Value Loss (MSE - Mean Squared Error)**: Measures the distance between the predicted score and the actual game result.
    *   Formula: `Loss = (True_Result - Pred_Score)^2`

`Total_Loss = Policy_Loss + Value_Loss`

### Step 3: Backward Pass (Backpropagation)
This is the "learning" part. Using the **Chain Rule** of calculus, PyTorch calculates the **Gradient** of the Loss with respect to every weight in the network.
*   "If I increase Weight #4023 by 0.001, the Error decreases by 0.0005."
*   It effectively computes the direction in which we should nudge the weights to make the error smaller.

### Step 4: Optimizer Step (SGD)
We use **Stochastic Gradient Descent (SGD) with Momentum**.
*   We update all weights: `Weight = Weight - Learning_Rate * Gradient`.
*   **Momentum**: Helps the optimization accelerate in relevant directions and dampens oscillations, like a heavy ball rolling down a hill.

---

## 4. The Bridge: Binary Export Protocol

Once the Python model is trained, it lives in PyTorch's memory. We need to transfer this "intelligence" to our C++ engine.

### Why not ONNX or TorchScript?
Standard export formats (ONNX, TorchScript) require huge runtime libraries (libtorch is ~500MB). We want our C++ engine to be lightweight and zero-dependency.

### Custom Binary Format (`.nn`)
We designed a minimalistic binary protocol. The Python script acts as a serializer:
1.  **Header**: Writes a "Magic Number" (`0x5245534E` - "RESN") to verify file integrity.
2.  **Metadata**: Writes architecture constants (Number of layers, Input channels).
3.  **Weights**: Iterates through every layer (Conv2d, BatchNorm) and writes the raw 32-bit floats from the GPU memory directly to the file.

The C++ engine's `loadModel` function is the exact mirror of this. It `mmap`s or reads the file linearly, reconstructing the network structure in milliseconds.
