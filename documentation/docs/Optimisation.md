---
id: Optimisation
title: Advanced Code Optimisation
sidebar_position: 3
---

# Advanced Code Optimisation

In high-performance computing like AI Game Engines, every nanosecond counts. We implement "Zero-Cost Abstractions" to ensure our C++ code runs as close to the metal as possible. Here is a granular breakdown of our optimization techniques.

---

## 1. Bitboard Architecture (The Foundation)

### The Problem
A standard implementation uses an array `int board[20][20]`.
*   **Memory**: 400 integers * 4 bytes = 1600 bytes.
*   **Speed**: Checking a line requires iterating `board[x][y]`, `board[x+1][y]`, etc. This involves multiple memory fetches (Cache Misses) and comparison instructions.

### The Solution: `std::bitset`
We map the 2D board to a 1D sequence of bits.
*   **Memory**: 400 bits ≈ 50 bytes. This fits entirely into a singe L1 CPU Cache line.
*   **Access**: Reading a bit is instantaneous.

## 2. SIMD-like Bitwise Parallelism

This is our critical optimization for the MCTS simulations. We need to check win conditions (5 in a row) millions of times.

Instead of writing:
```cpp
for (int x=0; x<20; x++) { ... check horizontal ... }
```
We use bitwise shifts. This effectively simulates **SIMD (Single Instruction, Multiple Data)** behavior using standard registers.

### The Algorithm
To check if Player P has 5 stones in a row horizontally:
1.  Take the bitboard `B` (where 1 = stone present).
2.  Compute `B1 = B & (B >> 1)` (This keeps 1s only if there is a stone to the left).
3.  Compute `B2 = B1 & (B1 >> 1)` (Keeps 1s only if there were 2 stones to the left, i.e., 3 aligned).
4.  If `B2 & (B2 >> 1)` is not zero, we have 5 aligned.

**Impact**: This checks the *entire board* for horizontal wins in ~4 CPU clock cycles. A loop would take hundreds.

## 3. Memory Layout & Buffering

### Why `new` and `malloc` are bad
Allocating memory (`new Tensor`, `std::vector`) asks the Operating System for a heap block. This is extremely slow (context switches, kernel locks).

### Static Buffers in Neural Network
In `Network.cpp`, we declare our working memory **once** at startup.
```cpp
Tensor _buffer1(1, 64, 20, 20);
Tensor _buffer2(1, 64, 20, 20);
```
During the 1500+ MCTS simulations per move:
*   We **never** allocate memory.
*   We reuse these buffers locally.
*   The Convolution output writes directly into `_buffer1`, which becomes the input for the next layer. This is "Zero-Allocation Inference".

## 4. Branchless Logic

Modern CPUs (like M4 or Intel i9) use "Branch Prediction". They guess which way an `if` will go to pre-calculate code. If they guess wrong, they flush the pipeline (huge performance penalty ~15 cycles).

We write "Branchless Code" to avoid `if`.

**Bad (Branching):**
```cpp
if (isMyTurn) value += 1;
else value -= 1;
```

**Good (Branchless):**
```cpp
// if isMyTurn is 1 or 0
value += (isMyTurn * 2) - 1; 
```
The compiler translates this into pure arithmetic instructions (`IMUL`, `SUB`) which flow linearly through the CPU pipeline without stalls.

## 5. Binary Serialization

Loading the neural network weights from a text file (JSON/CSV) requires parsing strings to floats (`atof`), which is slow.

We implemented a custom Memory Dump format.
1.  **Python**: Maps the GPU float array to bytes.
2.  **C++**: `file.read((char*)ptr, count * 4)`.

This performs a **Direct Memory Copy (DMA equivalent)** from disk to RAM. It is the theoretical maximum speed possible for file reading (limited only by SSD speed).

## 6. Loop Unrolling & Pointer Arithmetic

In our Convolution kernel (`conv2d`):
*   We avoid `std::vector::at()` or `tensor[x][y]` inside tight loops because they often include bounds checking.
*   We use **Raw Pointers**:
    ```cpp
    float* outPtr = output.values.data();
    const float* inPtr = input.values.data();
    for (int i=0; i<N; i++) *outPtr++ = *inPtr++ * weight;
    ```
*   The compiler can auto-vectorize this easily (using AVX/NEON instructions) because it sees a contiguous block of memory with no side effects.
