/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Parser
*/

#include "Parser.hpp"

/**
 * @brief Construct a new Parser:: Parser object
 */
Parser::Parser() {
  _isRunning = true;

  _commandHandlers["START"] = &Parser::StartCommand;
  _commandHandlers["TURN"] = &Parser::TurnCommand;
  _commandHandlers["BEGIN"] = &Parser::BeginCommand;
  _commandHandlers["BOARD"] = &Parser::BoardCommand;
  _commandHandlers["INFO"] = &Parser::InfoCommand;
  _commandHandlers["END"] = &Parser::EndCommand;
  _commandHandlers["ABOUT"] = &Parser::AboutCommand;
  _commandHandlers["RESTART"] = &Parser::RestartCommand;

  if (!_network.loadModel("assets/gomoku_model.bin")) {
    Logger::addLogGlobal("Failed to load neural network model");
  } else {
    Logger::addLogGlobal("Neural network model loaded successfully");
  }
}

/**
 * @brief Destroy the Parser:: Parser object
 */
Parser::~Parser() {
  _commandHandlers.clear();
  _isRunning = false;
}

/**
 * @brief Send an error message to the standard output
 */
void Parser::sendError(const std::string &message) {
  std::cout << "ERROR - " << message << std::endl;
  Logger::addLogGlobal("Sent error: " + message);
}

/**
 * @brief Run the main parser loop to process incoming commands
 */
void Parser::runParser() {
  std::string line;

  while (_isRunning && std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.empty())
      continue;

    std::stringstream ss(line);
    std::string command;
    ss >> command;

    auto it = _commandHandlers.find(command);
    if (it != _commandHandlers.end()) {
      CommandHandler handler = it->second;
      (this->*handler)(ss);
    } else {
      sendError("Unknown command: " + command);
      continue;
    }
  }
}

/**
 * @brief Handle the START command to initialize the game
 */
void Parser::StartCommand(std::stringstream &args) {
  int size;
  args >> size;
  if (size != SIZE) {
    sendError("Unsupported board size: " + std::to_string(size));
    return;
  }
  _gameBoard.resetBoard();
  Logger::addLogGlobal("MCTS: Doing Warm-up during START phase...");
  Board dummyBoard;
  MCTS mcts(_network);
  mcts.findBestMove(dummyBoard, 100);
  Logger::addLogGlobal("MCTS: Warm-up completed.");
  Logger::addLogGlobal("Game started with board size: " + std::to_string(size));
  std::cout << "OK" << std::endl;
}

bool Parser::parseCoordinates(const std::string &token, int &x, int &y) {
  size_t commaPos = token.find(',');
  if (commaPos == std::string::npos) {
    return false;
  }
  try {
    x = std::stoi(token.substr(0, commaPos));
    y = std::stoi(token.substr(commaPos + 1));
    if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) {
      return false;
    }
  } catch (const std::invalid_argument &) {
    return false;
  } catch (const std::out_of_range &) {
    return false;
  }
  return true;
}

/**
 * @brief Handle the TURN command to process opponent's move
 */
void Parser::TurnCommand(std::stringstream &args) {
  std::string move;

  if (args >> move) {
    int x = -1, y = -1;
    if (parseCoordinates(move, x, y)) {
      _gameBoard.makeMove(x, y, 2);
    } else {
      sendError("Invalid move format: " + move);
      return;
    }
  }
  Logger::addLogGlobal("MCTS Start Thinking...");
  MCTS mcts(_network);
  int timeLimit = (_timeoutTurn > 0) ? _timeoutTurn : 5000;

  std::pair<int, int> bestMove = mcts.findBestMove(_gameBoard, timeLimit - 400);

  _gameBoard.makeMove(bestMove.first, bestMove.second, 1);
  std::cout << bestMove.first << "," << bestMove.second << std::endl;

  Logger::addLogGlobal("AI played: " + std::to_string(bestMove.first) + "," +
                       std::to_string(bestMove.second));
}

/**
 * @brief Handle the BEGIN command to start the game as the first player
 */
void Parser::BeginCommand([[maybe_unused]] std::stringstream &args) {
  MCTS mcts(_network);
  std::pair<int, int> bestMove = mcts.findBestMove(_gameBoard, 1000);
  _gameBoard.makeMove(bestMove.first, bestMove.second, 1);
  std::cout << bestMove.first << "," << bestMove.second << std::endl;
}

// cppcheck-suppress constParameterCallback
void Parser::BoardCommand(std::stringstream &args) {
  std::string line;
  (void)args;

  _gameBoard.resetBoard();
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();

    if (line == "DONE") {
      int myStoneCount = _gameBoard.getMyBoard().count();
      int opponentStoneCount = _gameBoard.getOpponentBoard().count();

      bool isAiTurn = false;
      if (opponentStoneCount > myStoneCount) {
        isAiTurn = true;
      } else if (opponentStoneCount == 0 && myStoneCount == 0) {
        isAiTurn = true;
      }

      if (isAiTurn) {
        Logger::addLogGlobal(
            "BOARD done. Opponent has more stones (or start). AI Turn.");
        MCTS mcts(_network);
        std::pair<int, int> bestMove =
            mcts.findBestMove(_gameBoard, _timeoutTurn - 50);

        _gameBoard.makeMove(bestMove.first, bestMove.second, 2);

        std::cout << bestMove.first << "," << bestMove.second << std::endl;
        Logger::addLogGlobal("AI played: " + std::to_string(bestMove.first) +
                             "," + std::to_string(bestMove.second));
      } else {
        Logger::addLogGlobal("BOARD done. Counts equal (Opp=" +
                             std::to_string(opponentStoneCount) +
                             "). Waiting for opponent.");
      }
      return;
    }
    std::stringstream ssline(line);
    int x = -1, y = -1, player = -1;
    char sep1, sep2;

    if (ssline >> x >> sep1 >> y >> sep2 >> player) {
      if (sep1 != ',' || sep2 != ',') {
        sendError("Invalid BOARD command format: " + line);
        continue;
      }
      if (!(_gameBoard.isMoveValid(x, y))) {
        sendError("Invalid move in BOARD command at (" + std::to_string(x) +
                  "," + std::to_string(y) + ")");
        continue;
      }
      if (player == 1) {
        _gameBoard.makeMove(x, y, 1);
        Logger::addLogGlobal("Updated board: (" + std::to_string(x) + "," +
                             std::to_string(y) + ") P1 (AI)");
      } else if (player == 2) {
        _gameBoard.makeMove(x, y, 2);
        Logger::addLogGlobal("Updated board: (" + std::to_string(x) + "," +
                             std::to_string(y) + ") P2 (Opponent)");
      } else if (player == 3) {
        _gameBoard.makeMove(x, y, 2);
        Logger::addLogGlobal("Updated board: (" + std::to_string(x) + "," +
                             std::to_string(y) + ") P2 (from 3)");
      } else {
        sendError("Invalid player number in BOARD command: " +
                  std::to_string(player));
      }

    } else {
      sendError("Invalid BOARD command format: " + line);
    }
  }
}

/**
 * @brief Handle the INFO command to receive game information
 */
void Parser::InfoCommand(std::stringstream &args) {
  std::string key, value;

  if (args >> key >> value) {
    try {
      if (key == "timeout_turn") {
        int timeoutTurn = std::stoi(value);
        _timeoutTurn = timeoutTurn;
        Logger::addLogGlobal("Received timeout_turn info: " +
                             std::to_string(timeoutTurn));
      } else if (key == "timeout_match") {
        int timeoutMatch = std::stoi(value);
        Logger::addLogGlobal("Received timeout_match info: " +
                             std::to_string(timeoutMatch));
      } else if (key == "max_memory") {
        int maxMemory = std::stoi(value);
        Logger::addLogGlobal("Received max_memory info: " +
                             std::to_string(maxMemory));
      } else if (key == "time_left") {
        int timeLeft = std::stoi(value);
        Logger::addLogGlobal("Received time_left info: " +
                             std::to_string(timeLeft));
      } else if (key == "game_type") {
        int gameType = std::stoi(value);
        Logger::addLogGlobal("Received game_type info: " +
                             std::to_string(gameType));
      } else if (key == "rule") {
        int rule = std::stoi(value);
        Logger::addLogGlobal("Received rule info: " + std::to_string(rule));
      } else if (key == "evaluate") {
        int evaluate = std::stoi(value);
        Logger::addLogGlobal("Received evaluate info: " +
                             std::to_string(evaluate));
      } else if (key == "folder") {
        Logger::addLogGlobal("Received folder info: " + value);
      }
    } catch (const std::invalid_argument &) {
      sendError("Invalid value for INFO command: " + value);
    } catch (const std::out_of_range &) {
      sendError("Value out of range for INFO command: " + value);
    }
  } else {
    sendError("Invalid INFO command format");
  }
}

/**
 * @brief Handle the END command to terminate the game
 */
void Parser::EndCommand([[maybe_unused]] std::stringstream &args) {
  _isRunning = false;
  Logger::addLogGlobal("Game ended by END command");
}

/**
 * @brief Handle the ABOUT command to provide bot information
 */
void Parser::AboutCommand([[maybe_unused]] std::stringstream &args) {
  std::cout << "name=\"Tensor_Flowless\", version=\"1.0\", author=\"Enzo "
               "Gaggiotti && Yanis Dolivet\", country=\"French(Lyon)\""
            << std::endl;
  Logger::addLogGlobal("Sent ABOUT information");
}

/**
 * @brief Handle the RESTART command to reset the game
 */
void Parser::RestartCommand([[maybe_unused]] std::stringstream &args) {
  _gameBoard.resetBoard();
  Logger::addLogGlobal("Game restarted by RESTART command");
  std::cout << "OK" << std::endl;
}
