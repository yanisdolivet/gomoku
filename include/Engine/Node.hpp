/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Node
*/

#pragma once

#include <cmath>
#include <vector>

/**
 * @brief Node struct representing a state in the MCTS tree
 * @note This struct encapsulates the properties and methods for a node in the
 * MCTS algorithm It is use to represent the futur state possible in the search
 * tree.
 */

struct Node {
  int moveIndex;   // Index of the move leading to this node
  int visits;      // Number of times this node has been visited
  float score;     // Cumulative score of this node
  float priorProb; // Prior probability of selecting this node
  int player;      // Player to move at this node ( 1 or 2 )

  Node *parent;                 // Pointer to the parent node
  std::vector<Node *> children; // Vector of child nodes

  /**
   * @brief Construct a new Node object
   * @param moveIdx Index of the move leading to this node
   * @param prior Prior probability of selecting this node
   * @param plyr Player to move at this node ( 1 or 2 )
   * @param par Pointer to the parent node (default: nullptr)
   */
  Node(int moveIdx, float prior, int plyr, Node *par = nullptr)
      : moveIndex(moveIdx), visits(0), score(0.0f), priorProb(prior),
        player(plyr), parent(par) {
    children.reserve(20);
  }

  /**
   * @brief Destructor to clean up child nodes
   * @note Recursively deletes all child nodes to prevent memory leaks
   */
  ~Node() {
    for (Node *child : children) {
      delete child;
    }
  }

  /**
   * @brief Calculate the PUCT value for this node
   * @param curiosity_factor Exploration parameter (default: 1.0f)
   * @return float PUCT value
   * @note The PUCT formula balances exploration and exploitation in MCTS.
   */
  float getPUCT(float curiosity_factor = 1.0f) const {
    if (visits == 0) {
      return priorProb;
    }
    float exploitation = (score / visits + 1.0f) / 2.0f;
    float exploration = curiosity_factor * priorProb *
                        std::sqrt(static_cast<float>(parent->visits)) /
                        (1.0f + visits);

    return exploitation + exploration;
  }
};
