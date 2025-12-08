/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Board
*/

#pragma once

#include <bitset>

/**
 * @class Board
 * @brief Represents the game board for Gomoku.
 * @note We use bitboards to represent the board. This method is
 *       more efficient than using a 2D array.
 */

const int SIZE = 20;
const int AREA = SIZE * SIZE;

class Board {
public:
  Board();

  ~Board() = default;

  void resetBoard();

  bool makeMove(int x, int y, int player);

  bool isMoveValid(int x, int y) const;

  int checkWinner() const;

  const std::bitset<AREA> &getMyBoard() const;

  const std::bitset<AREA> &getOpponentBoard() const;

private:
  std::bitset<AREA> _myBoard;       // Board of the player
  std::bitset<AREA> _opponentBoard; // Board of the opponent

  std::bitset<AREA> _LeftMask;  // Mask on column 0, 20, 40, ...
  std::bitset<AREA> _RightMask; // Mask on column 19, 39, 59, ...
  std::bitset<AREA> _FullMask;  // Full board mask for the vertical checks

  bool fiveAlligned(const std::bitset<AREA> &board, int shift,
                     const std::bitset<AREA> &mask) const;
};
