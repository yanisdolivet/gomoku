/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Logger
*/

#include "Logger.hpp"

Logger::~Logger() {
  if (isLogFileOpen()) {
    closeLogFile();
  }
}

void Logger::initLogger() {
  _logFile.open("debug.log", std::ios::out | std::ios::trunc);
}

void Logger::addLog(const std::string &message) {
  if (!_logFile.is_open()) {
    return;
  }
  std::time_t now = std::time(nullptr);
  std::tm *tm_now = std::localtime(&now);
  char buffer[32];

  if (std::strftime(buffer, sizeof(buffer), "%a %b %d %H:%M:%S %Y", tm_now)) {
    _logFile << "[" << buffer << "] " << message << std::endl;
  }
}

bool Logger::isLogFileOpen() const { return _logFile.is_open(); }

void Logger::closeLogFile() {
  if (_logFile.is_open()) {
    _logFile.close();
  }
}
