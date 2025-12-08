/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Logger
*/

#include "Logger.hpp"
#include <criterion/criterion.h>
#include <criterion/redirect.h>

/**
 * @brief Test suite for Logger class
 * This suite tests the functionality of the Logger class, including
 * initialization, logging messages, and file handling.
 */

Test(logger, init_logger) {
  Logger logger;
  logger.initLogger();
  logger.closeLogFile();
  cr_assert_eq(logger.isLogFileOpen(), false);
}

Test(logger, write_log) {
  Logger logger;
  logger.initLogger();
  std::string testMsg = "Test message";
  logger.addLog(testMsg);
  logger.closeLogFile();

  std::ifstream file("debug.log");
  cr_assert_eq(file.is_open(), true);

  std::string line;
  std::getline(file, line);
  cr_assert(line.find(testMsg) != std::string::npos);
  file.close();
}
