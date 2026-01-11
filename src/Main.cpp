/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Main
*/

#include "Logger.hpp"
#include "Parser.hpp"
#include <iostream>

/**
 * @brief Main function to start the Gomoku application
 *
 * @note Optimize I/O operations with std::ios_base::sync_with_stdio(false)
 * and std::cin.tie(NULL) for better performance during gameplay.
 */
int main() {
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(NULL);

  Logger::getInstance().initLogger();
  Logger::getInstance().addLog("Gomoku game started");

  try {
    Parser parser;
    parser.runParser();
  } catch (const std::exception &e) {
    Logger::getInstance().addLog("CRITICAL ERROR: " + std::string(e.what()));
    Logger::getInstance().closeLogFile();
    return 1;
  }
  Logger::getInstance().addLog("Gomoku game ended");
  Logger::getInstance().closeLogFile();
  return 0;
}