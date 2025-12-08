/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Main
*/

#include "Logger.hpp"
#include <stdio.h>

int main(void) {
  Logger logger;
  logger.initLogger();
  logger.addLog("Application started");
  printf("Hello, Gomoku!\n");
  return 0;
}
