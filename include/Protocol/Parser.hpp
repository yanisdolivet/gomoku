/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Parser
*/

#pragma once

#include "../Core/Board.hpp"
#include "../Utils/Logger.hpp"
#include "../Neural/Network.hpp"
#include "../Engine/MCTS.hpp"

#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Protocol Parser for Gomoku
 * @note This reads the std::cin input and parse the commands for the Gomoku
 * game. It cuts the input into tokens and call the corresponding functions to
 * handle each command.
 */

class Parser {
public:
  Parser();
  ~Parser();

  void runParser();

  void StartCommand(std::stringstream &args);
  void TurnCommand(std::stringstream &args);
  void BeginCommand(std::stringstream &args);
  void BoardCommand(std::stringstream &args);
  void InfoCommand(std::stringstream &args);
  void EndCommand(std::stringstream &args);
  void AboutCommand(std::stringstream &args);
  void RestartCommand(std::stringstream &args);

private:
  Board _gameBoard;
  bool _isRunning;
  Network _network;
  int _timeoutTurn = 5000;

  void sendError(const std::string &message);

  bool parseCoordinates(const std::string &token, int &x, int &y);

  using CommandHandler = void (Parser::*)(std::stringstream &args);

  std::map<std::string, CommandHandler> _commandHandlers;
};