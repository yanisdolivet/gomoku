/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** VCF
*/

#include "VCF.hpp"
#include <iostream>

/**
 * @brief Get the First Set Bit object
 *
 * @param b Bitset to search
 * @return int Index of the first set bit
 */
static int getFirstSetBit(const std::bitset<AREA> &b) {
  for (int i = 0; i < AREA; ++i) {
    if (b.test(i))
      return i;
  }
  return -1;
}

/**
 * @brief Find the best move using VCF
 *
 * @param board Current game board
 * @param player Player number (1 or 2)
 * @param depth Search depth
 * @return std::pair<int, int> Best move coordinates (x, y)
 */
std::pair<int, int> VCF::solve(const Board &board, int player, int depth) {
  int winningMove = -1;
  if (vcf(board, player, depth, winningMove)) {
    return {winningMove % SIZE, winningMove / SIZE};
  }
  return {-1, -1};
}

/**
 * @brief Find the best move using VCF
 *
 * @param board Current game board
 * @param player Player number (1 or 2)
 * @param depth Search depth
 * @param winningMove Index of the winning move
 * @return true If a winning move was found
 * @return false If no winning move was found
 */
bool VCF::vcf(const Board &board, int player, int depth, int &winningMove) {
  if (depth == 0)
    return false;

  std::bitset<AREA> wins = board.getWinningCandidates(player);
  if (wins.any()) {
    winningMove = getFirstSetBit(wins);
    return true;
  }
  std::bitset<AREA> threats = board.getThreatCandidates(player);

  if (threats.none())
    return false;

  int opponent = (player == 1) ? 2 : 1;

  for (int i = 0; i < AREA; ++i) {
    if (!threats.test(i))
      continue;
    Board nextBoard = board;
    nextBoard.makeMove(i % SIZE, i / SIZE, player);
    std::bitset<AREA> forcedWins = nextBoard.getWinningCandidates(player);

    if (forcedWins.none()) {
      continue;
    }
    if (forcedWins.count() > 1) {
      winningMove = i;
      return true;
    }
    int reply = getFirstSetBit(forcedWins);
    if (!nextBoard.isMoveValid(reply % SIZE, reply / SIZE)) {
      continue;
    }
    nextBoard.makeMove(reply % SIZE, reply / SIZE, opponent);
    int subWin = -1;
    if (vcf(nextBoard, player, depth - 1, subWin)) {
      winningMove = i;
      return true;
    }
  }

  return false;
}
