/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** MCTS
*/

#include "MCTS.hpp"
#include <algorithm>
#include <chrono>

/**
 * @brief Construct a new MCTS:: MCTS object
 *
 * @param network Reference to the neural network for evaluations
 */
MCTS::MCTS(Network &network) : _network(network) {}

/**
 * @brief Destroy the MCTS::MCTS object
 */
MCTS::~MCTS() {
  if (_root) {
    delete _root;
  }
}

/**
 * @brief Reset the MCTS tree
 */
void MCTS::reset() {
  if (_root) {
    delete _root;
    _root = nullptr;
  }
}

/**
 * @brief Update the root of the MCTS tree with a new move
 * @param moveIndex Index of the move to update the root with
 * @note This function is called by the Parser when it's the bot's turn to play.
 */
void MCTS::updateRoot(int moveIndex) {
  if (!_root)
    return;

  Node *newRoot = nullptr;
  auto it = std::find_if(
      _root->children.begin(), _root->children.end(),
      [moveIndex](const Node *child) { return child->moveIndex == moveIndex; });

  if (it != _root->children.end()) {
    newRoot = *it;
  }

  if (newRoot) {
    auto &children = _root->children;
    children.erase(std::remove(children.begin(), children.end(), newRoot),
                   children.end());

    newRoot->parent = nullptr;
    delete _root;
    _root = newRoot;
    Logger::addLogGlobal("MCTS: Tree improved! Reusing " +
                         std::to_string(_root->visits) + " visits.");
  } else {
    Logger::addLogGlobal("MCTS: Move not in tree. resetting.");
    reset();
  }
}

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
  int timeout = timeMs;
  if (timeout < 0)
    timeout = 0;
  int rootPlayer =
      (board.getMyBoard().count() <= board.getOpponentBoard().count()) ? 1 : 2;

  std::pair<int, int> vcfWin = VCF::solve(board, rootPlayer, 12);
  if (vcfWin.first != -1) {
    Logger::addLogGlobal("VCF Win found at: " + std::to_string(vcfWin.first) +
                         ", " + std::to_string(vcfWin.second));
    return vcfWin;
  }

  int opponent = (rootPlayer == 1) ? 2 : 1;
  std::pair<int, int> vcfLoss = VCF::solve(board, opponent, 12);
  if (vcfLoss.first != -1) {
    Logger::addLogGlobal(
        "VCF Defense needed at: " + std::to_string(vcfLoss.first) + ", " +
        std::to_string(vcfLoss.second));
    return vcfLoss;
  }

  if (!_root) {
    _root = new Node(-1, 1.0f, rootPlayer, nullptr);
    expand(_root, board);
  }

  std::gamma_distribution<float> distribution(0.3f, 1.0f);
  std::mt19937 gen(std::random_device{}());
  float epsilon = 0.25f;

  float sumNoise = 0.0f;
  std::vector<float> noises;
  for (size_t i = 0; i < _root->children.size(); ++i) {
    float n = distribution(gen);
    noises.push_back(n);
    sumNoise += n;
  }

  for (size_t i = 0; i < _root->children.size(); ++i) {
    _root->children[i]->priorProb =
        (1 - epsilon) * _root->children[i]->priorProb +
        epsilon * (noises[i] / sumNoise);
  }

  Node *root = _root;
  while (true) {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
            .count();
    if (elapsed >= timeout)
      break;
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
  Output NeuralOutput = _network.predict(board, node->player);
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
