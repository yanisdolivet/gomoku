/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Logger
*/

#include "Logger.hpp"

/**
 * @brief Access a global logger instance from anywhere
 * @return Logger& global logger instance
 */
Logger &Logger::getInstance()
{
  static Logger instance;
  return instance;
}

/**
 * @brief Convenience static wrapper to add a log globally
 * @param message log message
 */
void Logger::addLogGlobal(const std::string &message)
{
  Logger::getInstance().addLog(message);
}

/**
 * @brief Destroy the Logger:: Logger object
 *
 */
Logger::~Logger() {
  if (isLogFileOpen()) {
    closeLogFile();
  }
}

/**
 * @brief Initialize the logger
 */
void Logger::initLogger() {
  _logFile.open("debug.log", std::ios::out | std::ios::trunc);
}

/**
 * @brief Add a log message to the log file
 *
 * @param message log to print in log file
 */
void Logger::addLog(const std::string &message) {
  if (!_logFile.is_open()) {
    return;
  }
  std::time_t now = std::time(nullptr);
  const std::tm *tm_now = std::localtime(&now);
  char buffer[32];

  if (std::strftime(buffer, sizeof(buffer), "%a %b %d %H:%M:%S %Y", tm_now)) {
    _logFile << "[" << buffer << "] " << message << std::endl;
  }
}

/**
 * @brief Check if the log file is open
 *
 * @return true if open
 * @return false if not open
 */
bool Logger::isLogFileOpen() const { return _logFile.is_open(); }

/**
 * @brief Close the log file
 */
void Logger::closeLogFile() {
  if (_logFile.is_open()) {
    _logFile.close();
  }
}

