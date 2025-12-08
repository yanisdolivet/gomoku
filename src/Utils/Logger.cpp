/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Logger
*/

#include "Logger.hpp"

void Logger::initLogger() {
  _logFile.open("debug.log", std::ios::out | std::ios::trunc);
}

void Logger::addLog(const std::string &message) {
    if (!_logFile.is_open()) {
        return;
    }
    time_t now = time(0);
    char* dt = ctime(&now);
    if (dt) {
        dt[strlen(dt) - 1] = '\0';
        _logFile << "[" << dt << "] " << message << std::endl;
    }
}
