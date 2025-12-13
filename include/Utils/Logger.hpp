/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Logger
*/

#pragma once
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>

/**
 * @brief Logger utility for Gomoku
 * @note This class provides logging functionalities for debugging and
 * monitoring the Gomoku application.
 * Provides methods to log messages in log files.
 */

class Logger {
public:
  Logger() = default;
  ~Logger();

  void initLogger();

  void addLog(const std::string &message);
  // Access a global logger instance from anywhere: Logger::getInstance().addLog(...)
  static Logger &getInstance();
  // Convenience static wrapper so you can call Logger::addLogGlobal("msg");
  static void addLogGlobal(const std::string &message);

  bool isLogFileOpen() const;

  void closeLogFile();

private:
  std::ofstream _logFile;
};