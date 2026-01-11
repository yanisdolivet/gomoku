/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** MCTS
*/

#pragma once

#include "Logger.hpp"
#include "Network.hpp"
#include "Node.hpp"
#include "VCF.hpp"

#include <random>
#include <utility>

#define DEFAULT_SECURITY_MS 350

/**
 * @brief Monte Carlo Tree Search Engine for Gomoku
 * @note This class implements the MCTS algorithm for the Gomoku game.
 */

class MCTS {
public:
  explicit MCTS(Network &network);
  ~MCTS();

  std::pair<int, int> findBestMove(const Board &board, int timeMs);
  void reset();
  void updateRoot(int moveIndex);

private:
  Network &_network;
  Node *_root = nullptr;

  Node *select(Node *node, Board &board);
  void expand(Node *node, const Board &board);
  void backpropagate(Node *node, float value);
};