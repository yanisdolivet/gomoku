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
Parser::Parser(Logger &logger) : _logger(&logger) {
  _isRunning = true;

  _commandHandlers["START"] = &Parser::StartCommand;
  _commandHandlers["TURN"] = &Parser::TurnCommand;
  _commandHandlers["BEGIN"] = &Parser::BeginCommand;
  _commandHandlers["BOARD"] = &Parser::BoardCommand;
  _commandHandlers["INFO"] = &Parser::InfoCommand;
  _commandHandlers["END"] = &Parser::EndCommand;
  _commandHandlers["ABOUT"] = &Parser::AboutCommand;
  _commandHandlers["RESTART"] = &Parser::RestartCommand;
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
  _logger->addLog("Sent error: " + message);
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
  _logger->addLog("Game started with board size: " + std::to_string(size));
  std::cout << "OK" << std::endl;
}

bool Parser::parseCoordinates(std::string &token, int &x, int &y) {
  size_t commaPos = token.find(',');
  if (commaPos == std::string::npos) {
    return false;
  }
  try {
    x = std::stoi(token.substr(0, commaPos));
    y = std::stoi(token.substr(commaPos + 1));
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
  int x = -1, y = -1;

  args >> move;

  if (!parseCoordinates(move, x, y)) {
    sendError("Invalid move format: " + move);
    return;
  }
  _gameBoard.makeMove(x, y, 2);
  _logger->addLog("Processed opponent's move to (" + std::to_string(x) + "," +
                  std::to_string(y) + ")");

  // future implementation of the ia here
  int myX = 10;
  int myY = 10;
  _gameBoard.makeMove(myX, myY, 1);
  std::cout << myX << "," << myY << std::endl;
  _logger->addLog("Made my move to (" + std::to_string(myX) + "," +
                  std::to_string(myY) + ")");
}

/**
 * @brief Handle the BEGIN command to start the game as the first player
 */
void Parser::BeginCommand([[maybe_unused]] std::stringstream &args) {
  _gameBoard.makeMove(10, 10, 1);
  std::cout << "10,10" << std::endl;
  _logger->addLog("Began game as first player, moved to (10,10)");
}

/**
 * @brief Handle the BOARD command to update the game board
 */
void Parser::BoardCommand(std::stringstream &args) {
  std::string line;

  _gameBoard.resetBoard();

  while (std::getline(args, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line == "DONE") {
      break;
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
      if (player == 1 || player == 2) {
        _gameBoard.makeMove(x, y, player);
        _logger->addLog("Updated board with move (" + std::to_string(x) + "," +
                        std::to_string(y) + ") by player " +
                        std::to_string(player));
      }
      if (player == 3) {
        _gameBoard.makeMove(x, y, 2);
        _logger->addLog("Updated board with move (" + std::to_string(x) + "," +
                        std::to_string(y) + ") by player 2");
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
        _logger->addLog("Received timeout_turn info: " +
                        std::to_string(timeoutTurn));
      } else if (key == "timeout_match") {
        int timeoutMatch = std::stoi(value);
        _logger->addLog("Received timeout_match info: " +
                        std::to_string(timeoutMatch));
      } else if (key == "max_memory") {
        int maxMemory = std::stoi(value);
        _logger->addLog("Received max_memory info: " +
                        std::to_string(maxMemory));
      } else if (key == "time_left") {
        int timeLeft = std::stoi(value);
        _logger->addLog("Received time_left info: " + std::to_string(timeLeft));
      } else if (key == "game_type") {
        int gameType = std::stoi(value);
        _logger->addLog("Received game_type info: " + std::to_string(gameType));
      } else if (key == "rule") {
        int rule = std::stoi(value);
        _logger->addLog("Received rule info: " + std::to_string(rule));
      } else if (key == "evaluate") {
        int evaluate = std::stoi(value);
        _logger->addLog("Received evaluate info: " + std::to_string(evaluate));
      } else if (key == "folder") {
        _logger->addLog("Received folder info: " + value);
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
  _logger->addLog("Game ended by END command");
}

/**
 * @brief Handle the ABOUT command to provide bot information
 */
void Parser::AboutCommand([[maybe_unused]] std::stringstream &args) {
  std::cout << "name=Tensor_Flowless, version=1.0, author=Enzo Gaggiotti && "
               "Yanis Dolivet, country=French(Lyon)"
            << std::endl;
  _logger->addLog("Sent ABOUT information");
}

/**
 * @brief Handle the RESTART command to reset the game
 */
void Parser::RestartCommand([[maybe_unused]] std::stringstream &args) {
  _gameBoard.resetBoard();
  _logger->addLog("Game restarted by RESTART command");
  std::cout << "OK" << std::endl;
}
