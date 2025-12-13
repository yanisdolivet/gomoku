---
id: Optimisation
title: Optimisation
sidebar_position: 3
---

# Optimisation

## Bitset

To optimise the project, we use bitset to store the board state and avoid using vectors or 2D arrays.
The reason is that vectors and 2D arrays are not optimised for the minimax algorithm and can lead to performance issues.
A bitset uses only 1 bit per position, whereas a vector or 2D array of int uses 32 bits (4 bytes) per position. This makes a bitset 32 times more space-efficient, which is why we use it in the Board class.

## Bitwise operations

To optimise the project, we use bitwise operations to check for a winner. We need to optimise this function because it is call by the Monte Carlo Tree Search algorithm. Monte Carlo Tree Search search for the best move by simulating multiple games. If checkWin take only 10ms to check the winner, the Monte Carlo Tree Search can simulate 1 000 000 games in 1 second. If checkWin take 1 millisecond to check the winner, the Monte Carlo Tree Search can simulate 1 000 games in 1 second.
Bitwise operations are faster than using loops to check for a winner because they are executed at the hardware level.
To explain it simply, bitwise operations are executed at the hardware level, while loops are executed at the software level. To check winner with loop, we need to check each bit of the board, while with bitwise operations, we can check multiple bits at once. At the final, bitwise check 64 bits (because processor is 64 bits) in 1 operation, while loop check 1 bit in 1 operation. The processor execute only 7 operations to check for a winner with bitwise.
Bitwise is the ultimate optimisation for this project for 3 reasons:
- No loop on runtime, all the loop are implemented in the hardware level
- Register parallelism, the processor can execute multiple operations at once
- Branchless, no if condition to check for a winner. This condition is not optimised by the processor. The instruction flux is linear, this is perfect for the processor.

To add more optimisation on the bitwise operations, we use inline function. That reduce the function call overhead and improve the performance. In a optimised code function call makes loose many performance. That would be shameful for a project like this.

## Loading binary file for loading model

Loading model files as binary data is crucial for performance. Instead of parsing text-based formats, which can be slow and resource-intensive, we directly read the raw binary data representing the model's weights and biases. This allows for much faster loading times, which is especially important for applications requiring quick initialization or frequent model updates.

The following C++ lambda function `loadTensor` is used to efficiently load a `Tensor` object from a binary file:

```cpp
auto loadTensor = [&](Tensor &tensor) {
    file.read(reinterpret_cast<char*>(tensor.values.data()),
              tensor.size() * sizeof(float));
};
```

**Explanation:**

*   **`auto loadTensor = [&](Tensor &tensor)`**: This defines a lambda function named `loadTensor`.
    *   `auto`: The compiler deduces the type of `loadTensor`.
    *   `[&]`: This captures all external variables by reference within the lambda's scope. In this case, it likely captures `file`, which is an `ifstream` or similar file stream object.
    *   `(Tensor &tensor)`: The lambda takes a reference to a `Tensor` object as an argument. This `Tensor` is where the loaded data will be stored.
*   **`file.read(...)`**: This is a method typically found in C++ `fstream` objects (like `ifstream`). It reads a block of raw binary data from the file.
    *   **`reinterpret_cast<char*>(tensor.values.data())`**:
        *   `tensor.values.data()`: Assumes `tensor.values` is some form of container (e.g., `std::vector<float>`) that stores the tensor's data. The `.data()` method returns a pointer to the underlying array where the `float` values are stored.
        *   `reinterpret_cast<char*>`: The `file.read()` method expects a `char*` pointer to the buffer where the data should be read. Since `tensor.values.data()` returns a pointer to `float` (or similar), it's reinterpreted as a `char*` to allow the raw byte-level read operation. This is safe as long as the memory pointed to is valid and the size matches.
    *   **`tensor.size() * sizeof(float)`**: This calculates the total number of bytes to read.
        *   `tensor.size()`: Returns the total number of elements (e.g., `float`s) in the tensor.
        *   `sizeof(float)`: Returns the size in bytes of a single `float` data type (typically 4 bytes).
        *   Multiplying these two gives the total byte size of the data to be read for the entire tensor.

In essence, this `loadTensor` lambda reads the exact byte representation of the `Tensor`'s floating-point values directly from the binary file into the `tensor.values` buffer, making the model loading process very efficient.

**Reasons**
This method is the most efficient way to load a tensor from a binary file because it allows us to read the raw bytes of the data stored in the file. This is much faster than parsing the data as text, which can be slow and resource-intensive. In addition, it consummes less memory than parsing the data as text, which can be a problem when loading large models.
Finally, it consumes less cpu resource because it doesn't need loop to parse the data.

## Loop Unrolling

To optimize the dense layer, we use loop unrolling. To explain it simply, loop unrolling is a technique that allows us to execute multiple iterations of a loop in a single instruction. This is much faster than executing a loop in a single instruction.
```cpp
for (int n = 0; n < inputSize - 4; n += 4) {
    sum += inputData[n] * weightData[n * outputSize + neuron];
    sum += inputData[n + 1] * weightData[(n + 1) * outputSize + neuron];
    sum += inputData[n + 2] * weightData[(n + 2) * outputSize + neuron];
    sum += inputData[n + 3] * weightData[(n + 3) * outputSize + neuron];
}
```
In this loop, we execute 4 iterations of the loop in a single instruction. This is much faster than executing a loop in a single instruction.

## Brut pointer

In the dense layer, we use brut pointer to access the data. This is much faster than using an iterator.
```cpp
const float *inputData = input.values.data();
const float *weightData = weight.values.data();
```

## Branchless code (without if condition)

Use branchless code to avoid if condition in predict method. This is much faster than using if condition.
```cpp
for (int n = 0; n < 400; ++n) {
inputData[n] = (float)myBoard.test(n) - (float)opponentBoard.test(n);
}
```

## Pre allocated buffer

In the dense layer, we use pre allocated buffer to store the data. This is much faster than using a vector.
