/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** TestParser
*/

#include "Parser.hpp"
#include <criterion/criterion.h>
#include <criterion/redirect.h>
#include <sstream>

void redirect_all_std(void) {
  cr_redirect_stdout();
  cr_redirect_stderr();
}

Test(Parser, run_parser_special_return, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::istringstream input("START 20\r\n");
  std::streambuf *cinbuf = std::cin.rdbuf();
  std::cin.rdbuf(input.rdbuf());

  parser.runParser();

  std::cin.rdbuf(cinbuf);

  cr_assert_stdout_eq_str("OK\n");
}

Test(Parser, run_parser_empty_input, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::istringstream input("\n");
  std::streambuf *cinbuf = std::cin.rdbuf();
  std::cin.rdbuf(input.rdbuf());

  parser.runParser();

  std::cin.rdbuf(cinbuf);

  cr_assert_stdout_eq_str("");
}

Test(Parser, run_parser_command, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::istringstream input("START 20\n");
  std::streambuf *cinbuf = std::cin.rdbuf();
  std::cin.rdbuf(input.rdbuf());

  parser.runParser();

  std::cin.rdbuf(cinbuf);

  cr_assert_stdout_eq_str("OK\n");
}

Test(Parser, run_parser_unknown_command, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::istringstream input("UNKNOWN_COMMAND\n");
  std::streambuf *cinbuf = std::cin.rdbuf();
  std::cin.rdbuf(input.rdbuf());

  parser.runParser();

  std::cin.rdbuf(cinbuf);

  cr_assert_stdout_eq_str("ERROR - Unknown command: UNKNOWN_COMMAND\n");
}

Test(Parser, start_command, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "20";
  parser.StartCommand(ss);

  cr_assert_stdout_eq_str("OK\n");
}

Test(Parser, start_command_invalid_size, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "10";
  parser.StartCommand(ss);

  cr_assert_stdout_eq_str("ERROR - Unsupported board size: 10\n");
}

Test(Parser, turn_command_valid, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream start_ss;
  start_ss << "20";
  parser.StartCommand(start_ss);
  fflush(stdout);

  std::stringstream ss;
  ss << "10,10";
  parser.TurnCommand(ss);
}

Test(Parser, begin_command, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  parser.BeginCommand(ss);
}

Test(Parser, invalid_board_format, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "invalid_format";
  parser.TurnCommand(ss);

  cr_assert_stdout_eq_str("ERROR - Invalid move format: invalid_format\n");
}

Test(Parser, invalid_board_command_format, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "invalid_board_command";
  parser.BoardCommand(ss);

  cr_assert_stdout_eq_str("ERROR - Invalid BOARD command format: invalid_board_command\n");
}

Test(Parser, board_command_valid_moves, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "10,10,1\n";
  ss << "11,11,2\n";
  ss << "DONE\n";
  parser.BoardCommand(ss);
}

Test(Parser, board_player_3_move, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "12,12,3\n";
  ss << "DONE\n";
  parser.BoardCommand(ss);
}

Test(Parser, board_invalid_command_format, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "15,15\n";
  ss << "DONE\n";
  parser.BoardCommand(ss);

  cr_assert_stdout_eq_str("ERROR - Invalid BOARD command format: 15,15\n");
}

Test(Parser, board_invalid_move, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "20,20,1\n";
  ss << "DONE\n";
  parser.BoardCommand(ss);

  cr_assert_stdout_eq_str("ERROR - Invalid move in BOARD command at (20,20)\n");
}

Test(Parser, board_invalid_separator, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "10;10;1\n";
  ss << "DONE\n";
  parser.BoardCommand(ss);

  cr_assert_stdout_eq_str("ERROR - Invalid BOARD command format: 10;10;1\n");
}

Test(Parser, info_command) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "timeout_turn 1000";
  parser.InfoCommand(ss);
}

Test(Parser, invaid_info_command_format, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "timeout_turn";
  parser.InfoCommand(ss);

  cr_assert_stdout_eq_str("ERROR - Invalid INFO command format\n");
}

Test(Parser, invalid_info_command_value, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "timeout_turn invalid_value";
  parser.InfoCommand(ss);

  cr_assert_stdout_eq_str("ERROR - Invalid value for INFO command: invalid_value\n");
}

Test(Parser, info_command_timeout_turn, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "timeout_turn 1500";
  parser.InfoCommand(ss);

  cr_assert_stdout_eq_str("");
}

Test(Parser, info_command_timeout_match, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "timeout_match 60000";
  parser.InfoCommand(ss);

  cr_assert_stdout_eq_str("");
}

Test(Parser, info_command_memory_limit, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "max_memory 256";
  parser.InfoCommand(ss);

  cr_assert_stdout_eq_str("");
}

Test(Parser, info_command_time_left, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "time_left 30000";
  parser.InfoCommand(ss);

  cr_assert_stdout_eq_str("");
}

Test(Parser, info_command_game_type, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "game_type 1";
  parser.InfoCommand(ss);

  cr_assert_stdout_eq_str("");
}

Test(Parser, info_command_rule, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "rule 0";
  parser.InfoCommand(ss);

  cr_assert_stdout_eq_str("");
}

Test(Parser, info_command_evaluate, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "evaluate 1";
  parser.InfoCommand(ss);

  cr_assert_stdout_eq_str("");
}

Test(Parser, info_command_folder, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "folder /path/to/folder";
  parser.InfoCommand(ss);

  cr_assert_stdout_eq_str("");
}

Test(Parser, info_command_out_of_range_key, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  ss << "unknown_key 123389869O869645685689I8569";
  parser.InfoCommand(ss);

  cr_assert_stdout_eq_str("");
}

Test(Parser, end_command) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  parser.EndCommand(ss);
}

Test(Parser, about_command, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  parser.AboutCommand(ss);

  cr_assert_stdout_eq_str("name=Tensor_Flowless, version=1.0, author=Enzo "
                          "Gaggiotti && Yanis Dolivet, country=French(Lyon)\n");
}

Test(Parser, restart_command, .init = redirect_all_std) {
  Logger logger;
  logger.initLogger();
  Parser parser(logger);

  std::stringstream ss;
  parser.RestartCommand(ss);

  cr_assert_stdout_eq_str("OK\n");
}