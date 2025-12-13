/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** MCTS
*/

#include "MCTS.hpp"

/**
 * @brief Construct a new MCTS:: MCTS object
 *
 * @param network Reference to the neural network for evaluations
 */
MCTS::MCTS(Network &network) : _network(network) {}

/**
 * @brief Find the best move using MCTS
 *
 * @param board Current game board
 * @param timeMs Time limit in milliseconds for the search
 * @return std::pair<int, int> Best move coordinates (x, y)
 *
 * @note This function is call by the Parser when it's the bot's turn to play.
 */
std::pair<int, int> MCTS::findBestMove(const Board &board, int timeMs) {
  auto start = std::chrono::high_resolution_clock::now();
  int iterations = 0;
  int timeout = timeMs - DEFAULT_SECURITY_MS;
  int rootPlayer =
      (board.getMyBoard().count() <= board.getOpponentBoard().count()) ? 1 : 2;
  Node *root = new Node(-1, 1.0f, rootPlayer, nullptr);

  expand(root, board);
  while (true) {
    if ((iterations & 63) == 0) {
      auto now = std::chrono::high_resolution_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
              .count();
      if (elapsed >= timeout)
        break;
    }
    iterations++;

    Board simBoard = board;
    Node *selectedNode = select(root, simBoard);
    int winner = simBoard.checkWinner();
    if (winner != 0) {
      backpropagate(selectedNode, 1.0f);
    } else {
      expand(selectedNode, simBoard);
    }
  }
  Logger::addLogGlobal("MCTS completed " + std::to_string(iterations) +
                       " iterations");

  const Node *bestChild = nullptr;
  int maxVisits = -1;

  for (Node *child : root->children) {
    if (child->visits > maxVisits) {
      maxVisits = child->visits;
      bestChild = child;
    }
  }
  std::pair<int, int> bestMove = {10, 10};
  if (bestChild) {
    int moveIndex = bestChild->moveIndex;
    bestMove = {moveIndex % SIZE, moveIndex / SIZE};
  }
  delete root;
  return bestMove;
}

/**
 * @brief Select a node to expand
 *
 * @param node Current node
 * @param board Current game board
 * @return Node* Selected node for expansion
 * @note This function traverses the MCTS tree using the PUCT formula to
 * select the most promising node.
 */

Node *MCTS::select(Node *node, Board &board) {
  while (!node->children.empty()) {
    Node *bestChild = nullptr;
    float bestPUCT = -1e9;

    for (Node *child : node->children) {
      float score = child->getPUCT();
      if (score > bestPUCT) {
        bestPUCT = score;
        bestChild = child;
      }
    }
    node = bestChild;

    int moveX = node->moveIndex % SIZE;
    int moveY = node->moveIndex / SIZE;
    int currentPlayer = (node->player == 1) ? 2 : 1;

    board.makeMove(moveX, moveY, currentPlayer);
  }
  return node;
}

/**
 * @brief Expand a node by adding child nodes
 *
 * @param node Node to expand
 * @param board Current game board
 * @note This function uses the neural network to evaluate the board state
 * and generate prior probabilities for possible moves.
 */
void MCTS::expand(Node *node, const Board &board) {
  Output NeuralOutput = _network.predict(board);
  float resultValue = NeuralOutput.value;
  int nextPlayer = (node->player == 1) ? 2 : 1;
  const auto &myBoard = board.getMyBoard();
  const auto &opponentBoard = board.getOpponentBoard();

  for (int i = 0; i < AREA; i++) {
    if (!myBoard.test(i) && !opponentBoard.test(i)) {
      float probability = NeuralOutput.policy[i];
      if (probability > 0.001f) {
        Node *childNode = new Node(i, probability, nextPlayer, node);
        node->children.push_back(childNode);
      }
    }
  }
  backpropagate(node, -resultValue);
}

/**
 * @brief Backpropagate the result value up the tree
 * @param node Node to start backpropagation from
 * @param value Result value to propagate
 * @note This function updates the visit counts and scores of nodes
 * along the path from the given node to the root.
 */
void MCTS::backpropagate(Node *node, float value) {
  while (node != nullptr) {
    node->visits += 1;
    node->score += value;
    value = -value;
    node = node->parent;
  }
}
