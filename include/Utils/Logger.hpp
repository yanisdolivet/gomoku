/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Logger
*/

#pragma once
#include <fstream>
#include <time.h>

/**
 * @brief Logger utility for Gomoku
 * @note This class provides logging functionalities for debugging and
 * monitoring the Gomoku application.
 * Provides methods to log messages in log files.
 */

class Logger {
public:

  Logger() = default;
  ~Logger() = default;

  void initLogger();

  void addLog(const std::string &message);

private:
  std::ofstream _logFile;
};