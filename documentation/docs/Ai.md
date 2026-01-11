---
id: Ai
title: The Brain: Oracle & Investigator
sidebar_position: 2
---

# The Brain: A Detective Duo

To understand how our AI works, imagine it as a **Duo of Detectives** trying to solve a case (win the game) within a strict time limit (5 seconds).

This duo consists of two very different entities working hand in hand:

1.  **The Oracle (Neural Network)**: Has brilliant intuition but is lazy. Looking at a crime scene (the board), it instantly says: *"I think it's Colonel Mustard!"* But it doesn't check the evidence.
2.  **The Investigator (MCTS)**: Is methodical and obsessive. It takes the leads given by the Oracle and physically verifies if they hold up by simulating the future.

---

## Part 1: The Oracle (The Neural Network)

**Role: Pure Intuition.**

Don't think of the neural network as a complete brain, but rather as a pair of ultra-sophisticated eyes connected to a reflex.

### 1. The Input (What it sees)
We give it a "photo" of the board. But instead of colored pixels, it sees our Bitboard ($20 \times 20$):
*   **Layer 1**: Where are my stones? (1 or 0)
*   **Layer 2**: Where are the opponent's stones?
*   **Layer 3**: History (Last move played).

### 2. The Processing (The Filters)
The image passes through layers of "filters" (Convolution).
Imagine placing small $3 \times 3$ stencils over the board to look for specific patterns:
*   *Filter 1* looks for "3 aligned stones".
*   *Filter 2* looks for "a blocked diagonal".
*   *Filter 3* looks for "a broken L-shape".

By stacking these layers, the network no longer sees stones; it sees tactical concepts (*"Danger top-right"*, *"Opportunity in center"*).

### 3. The Output (The Prophecy)
At the end, the Oracle spits out two crucial pieces of information:

1.  **The Policy ($P$ - Heatmap)**:
    *   It doesn't tell us "Play at 10,10".
    *   It tells us: *"I have a very good feeling about 10,10 (80%), a doubt about 9,9 (15%), and the rest is trash (5%)."*
    *   **Analogy**: *"I bet the culprit is in this room."*

2.  **The Value ($V$ - Prediction)**:
    *   A single number between -1 (Lost) and +1 (Won).
    *   It tells us: *"Given the current board, I am 90% confident we will win in the end."*

---

## Part 2: The Investigator (MCTS)

**Role: Verification by Simulation.**

If we followed the Oracle blindly, we would make stupid mistakes (blunders) because it sometimes misses obvious things. The Investigator (**MCTS** - Monte Carlo Tree Search) takes the Oracle's predictions and builds a **Tree of Possibilities**.

It runs in a loop during your 5 seconds. Here is what it does at every iteration (one iteration = a few microseconds):

### Step 1: Selection (Choosing the Path)
We start at the root (current board). We must descend into the tree until we reach a leaf (a future we haven't explored yet). How to choose the path? We use a magic formula (**PUCT**) that balances two conflicting desires:
*   **Greed (Exploitation)**: *"The Oracle said this move was great, and my previous simulations confirm we usually win down this path. Let's go there again."*
*   **Curiosity (Exploration)**: *"We've never looked at this move, and the Oracle said it wasn't bad. Let's check it out, just in case."*

### Step 2: Expansion (Opening a Door)
We reach the end of a known path. We play one more move. We create a new Node.
> *"Okay, if we play here, the board looks like this."*

### Step 3: Evaluation (Calling the Oracle)
**This is THE difference from classic MCTS.**
*   *Before (Classic)*: We played randomly until the end of the game (thousands of moves) to see who won. Too slow and inaccurate for Gomoku.
*   **Now (AlphaZero)**: We stop immediately. We show the new board to the Oracle.
    > **Oracle**: *"Hmm, in this position, I think you have a 60% chance of winning ($V=0.6$)."*

### Step 4: Backpropagation (The Report)
The Investigator climbs back up the entire path to the root. It updates every node it passed through:
> *"Hey guys! I explored this branch, and the Oracle says it smells good (0.6). Increase the average score of this path!"*

---

## Synthesis: The Perfect Symbiosis

Why is this method unbeatable?

1.  **The Oracle saves time**: Thanks to it, MCTS doesn't waste time exploring stupid moves (like playing in an empty corner at the start). It focuses only on moves the Oracle deems "interesting".
2.  **The MCTS corrects the Oracle**: Sometimes the Oracle is wrong (*"Looks winning..."*). But by descending the tree, MCTS realizes that 3 moves later, we lose. It effectively lowers the score of that move, even if the Oracle liked it initially.

### Concrete Example on the Board
1.  **Situation**: The opponent has 3 aligned stones.
2.  **Oracle ($P$)**: Sees the pattern. Screams: *"Block here! 99% Probability!"*
3.  **MCTS**:
    *   Looks at the Oracle's suggestion.
    *   Simulates the block.
    *   Asks the Oracle about the position *after* the block.
    *   **Oracle ($V$)**: *"Now it's safe. Value +0.1."*
    *   MCTS reports back: *"Blocking is a good move."*
4.  **Result**: After 1000 simulations, the move "Block" was visited 990 times. Your bot plays this move.

---

## Technical Challenge (Code Implementation)
Since ML libraries are forbidden in the C++ engine:

1.  **Matrix Mathematics**: The Oracle is just a massive matrix multiplication. We coded `Matrix * Matrix` in optimized C++ (AVX/SIMD).
2.  **Weight Loading**: The Oracle needs to "know" (the learned filters). We train them in Python, save them to a `.bin` file, and the C++ engine loads them into `float[]` arrays at startup.
3.  **The Tree**: A lightweight pointer structure (`Node*`) designed to not explode the RAM.
