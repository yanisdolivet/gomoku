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
  const std::bitset<AREA> *players[2] = {&_myBoard, &_opponentBoard};

  for (int p = 0; p < 2; ++p) {
    const std::bitset<AREA> &board = *players[p];

    if (fiveAligned(board, 1, _FullMask)) {
      return p + 1;
    }

    if (fiveAligned(board, SIZE, _FullMask)) {
      return p + 1;
    }
    if (fiveAligned(board, SIZE + 1, _RightMask)) {
      return p + 1;
    }
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

int Board::getLastMoveIndex() const {
  for (int i = AREA - 1; i >= 0; --i) {
    if (_myBoard.test(i) || _opponentBoard.test(i)) {
      return i;
    }
  }
  return -1; // No moves made yet
}

/**
 * @brief Get opponent's board bitset
 *
 * @return const std::bitset<AREA> &
 */
const std::bitset<AREA> &Board::getOpponentBoard() const {
  return _opponentBoard;
}

/**
 * @brief Get winning move candidates for a player
 * @param player Player number (1 or 2)
 * @return std::bitset<AREA> Bitset of winning move candidates
 * @note A winning move candidate is a position that, if played by the player,
 * would result in an immediate win.
 */
std::bitset<AREA> Board::getWinningCandidates(int player) const {
  const std::bitset<AREA> &p = (player == 1) ? _myBoard : _opponentBoard;
  const std::bitset<AREA> &o = (player == 1) ? _opponentBoard : _myBoard;
  std::bitset<AREA> empty = ~(p | o);
  std::bitset<AREA> candidates;

  auto shift = [&](std::bitset<AREA> b, int dist, int dir) {
    if (dir == 1) {
        // Shift the bitset in the horizontal direction
      if (dist > 0) {
        // Shift left
        for (int k = 0; k < dist; ++k)
          b = (b & _RightMask) << 1;
      } else {
        // Shift right
        for (int k = 0; k < -dist; ++k)
          b = (b & _LeftMask) >> 1;
      }
    } else if (dir == SIZE) {
      // Shift the bitset in the vertical direction
      if (dist > 0)
        // Shift up
        b <<= (dist * SIZE);
      else
        // Shift down
        b >>= (-dist * SIZE);
    } else if (dir == SIZE + 1) {
      // Shift the bitset in the diagonal (bottom-left to top-right) direction
      if (dist > 0)
        // Shift up-left
        for (int k = 0; k < dist; ++k)
          b = (b & _RightMask) << (SIZE + 1);
      else
        // Shift down-right
        for (int k = 0; k < -dist; ++k)
          b = (b & _LeftMask) >> (SIZE + 1);
    } else if (dir == SIZE - 1) {
        // Shift the bitset in the diagonal (top-left to bottom-right) direction
      if (dist > 0)
        // Shift up-right
        for (int k = 0; k < dist; ++k)
          b = (b & _LeftMask) << (SIZE - 1);
      else
        // Shift down-left
        for (int k = 0; k < -dist; ++k)
          b = (b & _RightMask) >> (SIZE - 1);
    }
    return b;
  };

  const int dirs[] = {1, SIZE, SIZE + 1, SIZE - 1};

  for (int d : dirs) {
    std::bitset<AREA> s1 = shift(p, -1, d);
    std::bitset<AREA> s2 = shift(p, -2, d);
    std::bitset<AREA> s3 = shift(p, -3, d);
    std::bitset<AREA> s4 = shift(p, -4, d);

    std::bitset<AREA> f1 = shift(p, 1, d);
    std::bitset<AREA> f2 = shift(p, 2, d);
    std::bitset<AREA> f3 = shift(p, 3, d);
    std::bitset<AREA> f4 = shift(p, 4, d);

    candidates |= (f1 & f2 & f3 & f4);
    candidates |= (s1 & f1 & f2 & f3);
    candidates |= (s1 & s2 & f1 & f2);
    candidates |= (s1 & s2 & s3 & f1);
    candidates |= (s1 & s2 & s3 & s4);
  }

  return candidates & empty;
}

/**
 * @brief Get threat move candidates for a player
 * @param player Player number (1 or 2)
 * @return std::bitset<AREA> Bitset of threat move candidates
 * @note A threat move candidate is a position that, if played by the player,
 * would create a four-in-a-row, setting up for a potential win on the next turn.
 */
std::bitset<AREA> Board::getThreatCandidates(int player) const {
  const std::bitset<AREA> &p = (player == 1) ? _myBoard : _opponentBoard;
  const std::bitset<AREA> &o = (player == 1) ? _opponentBoard : _myBoard;
  std::bitset<AREA> empty = ~(p | o);
  std::bitset<AREA> candidates;

  auto shift = [&](std::bitset<AREA> b, int dist, int dir) {
    if (dir == 1) {
      if (dist > 0)
        for (int k = 0; k < dist; ++k)
          b = (b & _RightMask) << 1;
      else
        for (int k = 0; k < -dist; ++k)
          b = (b & _LeftMask) >> 1;
    } else if (dir == SIZE) {
      if (dist > 0)
        b <<= (dist * SIZE);
      else
        b >>= (-dist * SIZE);
    } else if (dir == SIZE + 1) {
      if (dist > 0)
        for (int k = 0; k < dist; ++k)
          b = (b & _RightMask) << (SIZE + 1);
      else
        for (int k = 0; k < -dist; ++k)
          b = (b & _LeftMask) >> (SIZE + 1);
    } else if (dir == SIZE - 1) {
      if (dist > 0)
        for (int k = 0; k < dist; ++k)
          b = (b & _LeftMask) << (SIZE - 1);
      else
        for (int k = 0; k < -dist; ++k)
          b = (b & _RightMask) >> (SIZE - 1);
    }
    return b;
  };

  const int dirs[] = {1, SIZE, SIZE + 1, SIZE - 1};

  for (int d : dirs) {
    std::bitset<AREA> s1 = shift(p, -1, d);
    std::bitset<AREA> s2 = shift(p, -2, d);
    std::bitset<AREA> s3 = shift(p, -3, d);

    std::bitset<AREA> f1 = shift(p, 1, d);
    std::bitset<AREA> f2 = shift(p, 2, d);
    std::bitset<AREA> f3 = shift(p, 3, d);
    candidates |= (f1 & f2 & f3);
    candidates |= (s1 & f1 & f2);
    candidates |= (s1 & s2 & f1);
    candidates |= (s1 & s2 & s3);
  }
  return candidates & empty;
}
