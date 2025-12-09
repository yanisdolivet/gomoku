/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Main
*/

#include "Logger.hpp"
#include "Parser.hpp"

/**
 * @brief Main function to start the Gomoku application
 *
 * @note Optimize I/O operations with std::ios_base::sync_with_stdio(false)
 * and std::cin.tie(NULL) for better performance during gameplay.
 */
int main() {
  Logger logger;
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(NULL);

  logger.initLogger();
  logger.addLog("Gomoku game started");
  Parser parser(logger);
  parser.runParser();
  logger.addLog("Gomoku game ended");
  logger.closeLogFile();
  return 0;
}