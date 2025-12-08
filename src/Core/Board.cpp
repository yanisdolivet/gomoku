/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Board
*/

#include "Board.hpp"

/**
 * @brief Construct a new Board:: Board object
 *
 */
Board::Board() {
  resetBoard();
  _LeftMask.set();
  _RightMask.set();
  _FullMask.set();

  for (int i = 0; i < SIZE; ++i) {
    _LeftMask.reset(i * SIZE);
    _RightMask.reset(i * SIZE + (SIZE - 1));
  }
}

/**
 * @brief Reset the board to its initial state
 *
 */
void Board::resetBoard() {
  _myBoard.reset();
  _opponentBoard.reset();
}

/**
 * @brief Check if a move is valid
 *
 * @param x X coordinate
 * @param y Y coordinate
 * @return true if valid
 * @return false if not valid
 */
bool Board::isMoveValid(int x, int y) const {
  if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) {
    return false;
  }
  int index = y * SIZE + x;

  return !(_myBoard.test(index) || _opponentBoard.test(index));
}

/**
 * @brief Make a move on the board
 *
 * @param x X coordinate
 * @param y Y coordinate
 * @param player Player number (1 or 2)
 */
bool Board::makeMove(int x, int y, int player) {
  if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) {
    return false;
  }
  int index = y * SIZE + x;

  if (!isMoveValid(x, y)) {
    return false;
  }
  if (player == 1) {
    _myBoard.set(index);
  } else if (player == 2) {
    _opponentBoard.set(index);
  }
  return true;
}

inline bool Board::fiveAligned(const std::bitset<AREA> &board, int shift,
                               const std::bitset<AREA> &mask) const {
  std::bitset<AREA> tmp;

  // Isolate pair of stones
  tmp = board & (board >> shift) & mask;

  // Isolate triplet of stones
  tmp = tmp & (tmp >> (shift)) & mask;

  // Isolate quadruplet of stones
  tmp = tmp & (tmp >> (shift)) & mask;

  // Isolate quintuplet of stones
  tmp = tmp & (tmp >> (shift)) & mask;

  return tmp.any();
}

/**
 * @brief Check for a winner on the board
 *
 * @return int Player number (1 or 2) if there's a winner, 0 otherwise
 */
int Board::checkWinner() const {
  // Check for both players
  const std::bitset<AREA> *players[2] = {&_myBoard, &_opponentBoard};

  for (int p = 0; p < 2; ++p) {
    const std::bitset<AREA> &board = *players[p];

    // Check horizontal (shift 1)
    if (fiveAligned(board, 1, _FullMask)) {
      return p + 1;
    }

    // Check vertical (shift 20)
    if (fiveAligned(board, SIZE, _FullMask)) {
      return p + 1;
    }

    // Check diagonal (shift 21)
    if (fiveAligned(board, SIZE + 1, _RightMask)) {
      return p + 1;
    }

    // Check diagonal (shift 19)
    if (fiveAligned(board, SIZE - 1, _LeftMask)) {
      return p + 1;
    }
  }
  return 0;
}

/**
 * @brief Get my board bitset
 *
 * @return const std::bitset<AREA> &
 */
const std::bitset<AREA> &Board::getMyBoard() const { return _myBoard; }

/**
 * @brief Get opponent's board bitset
 *
 * @return const std::bitset<AREA> &
 */
const std::bitset<AREA> &Board::getOpponentBoard() const {
  return _opponentBoard;
}
