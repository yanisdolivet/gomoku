---
id: Optimisation
title: Optimisation
sidebar_position: 2
---

# Optimisation

## Bitset
To optimise the project, we use bitset to store the board state and avoid using vectors or 2D arrays.
The reason is that vectors and 2D arrays are not optimised for the minimax algorithm and can lead to performance issues.
Bitset of int is 32 size more compact than vectors and 2D arrays of int. This is the reason why we use bitset in Board class.

## Bitwise operations
To optimise the project, we use bitwise operations to check for a winner. We need to optimise this function because it is call by the Monte Carlo Tree Search algorithm. Monte Carlo Tree Search search for the best move by simulating multiple games. If checkWin take only 10ms to check the winner, the Monte Carlo Tree Search can simule 1 000 000 games in 1 second. If checkWin take 1 millisecond to check the winner, the Monte Carlo Tree Search can simule 1 000 games in 1 second.
Bitwise operations are faster than using loops to check for a winner because they are executed at the hardware level.
To explain it simply, bitwise operations are executed at the hardware level, while loops are executed at the software level. To check winner with loop, we need to check each bit of the board, while with bitwise operations, we can check multiple bits at once. At the final, bitwise check 64 bits (because processor is 64 bits) in 1 operation, while loop check 1 bit in 1 operation. The processor execute only 7 operations to check for a winner with bitwise.
Bitwise is the ultimate optimisation for this project for 3 reasons:
- No loop on runtime, all the loop are implemented in the hardware level
- Register parallelism, the processor can execute multiple operations at once
- Branchless, no if condition to check for a winner. This condition is not optimised by the processor. The instruction flux is linear, this is perfect for the processor.
To add more optimisation on the bitwise operations, we use inline function. That reduce the function call overhead and improve the performance. In a optimised code function call makes loose many performance. That would be shameful for a project like this.



