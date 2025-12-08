/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Main
*/

#include "Board.hpp"
#include "Logger.hpp"
#include <stdio.h>

int main() {
  Logger logger;
  logger.initLogger();
  logger.addLog("Gomoku game started");

  Board board;

  logger.addLog("Gomoku game ended");
  logger.closeLogFile();
  return 0;
}