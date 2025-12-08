/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** TestBoard
*/

#include "Board.hpp"
#include <criterion/criterion.h>

Test(Board, initialization) {
  Board board;
  cr_assert_eq(board.getMyBoard().none(), true,
               "My board should be empty initially");
  cr_assert_eq(board.getOpponentBoard().none(), true,
               "Opponent board should be empty initially");
}

Test(Board, valid_move) {
  Board board;
  cr_assert_eq(board.makeMove(0, 0, 1), true,
               "Move (0,0) should be valid for player 1");
  cr_assert_eq(board.getMyBoard().test(0), true,
               "Bit at index 0 should be set for player 1");
  cr_assert_eq(board.getOpponentBoard().test(0), false,
               "Bit at index 0 should NOT be set for player 2");
}

Test(Board, invalid_move_out_of_bounds) {
  Board board;
  cr_assert_eq(board.makeMove(-1, 0, 1), false,
               "Negative coordinates should be invalid");
  cr_assert_eq(board.makeMove(20, 0, 1), false,
               "Coordinates >= SIZE should be invalid");
}

Test(Board, invalid_move_occupied) {
  Board board;
  board.makeMove(10, 10, 1);
  cr_assert_eq(board.makeMove(10, 10, 2), false,
               "Cannot place on occupied spot");
}

Test(Board, check_winner_horizontal) {
  Board board;
  // Player 1 places horizontal line
  for (int i = 0; i < 5; ++i) {
    board.makeMove(i, 0, 1);
  }
  cr_assert_eq(board.checkWinner(), 1, "Player 1 should win horizontally");
}

Test(Board, check_winner_vertical) {
  Board board;
  // Player 2 places vertical line
  for (int i = 0; i < 5; ++i) {
    board.makeMove(0, i, 2);
  }
  cr_assert_eq(board.checkWinner(), 2, "Player 2 should win vertically");
}

Test(Board, check_winner_diagonal) {
  Board board;
  // Player 1 diagonal
  for (int i = 0; i < 5; ++i) {
    board.makeMove(i, i, 1);
  }
  cr_assert_eq(board.checkWinner(), 1, "Player 1 should win diagonally");
}

Test(Board, check_winner_anti_diagonal) {
  Board board;
  // Player 2 diagonal / at (0, 4) to (4, 0)
  for (int i = 0; i < 5; ++i) {
    board.makeMove(i, 4 - i, 2);
  }
  cr_assert_eq(board.checkWinner(), 2,
               "Player 2 should win anti-diagonally (/)");
}

Test(Board, check_move_valid) {
  Board board;
  cr_assert_eq(board.isMoveValid(0, 0), true, "Move (0,0) should be valid");
}

Test(Board, check_move_invalid) {
  Board board;
  board.makeMove(0, 0, 1);
  cr_assert_eq(board.isMoveValid(0, 0), false, "Move (0,0) should be invalid");
}

Test(Board, check_move_invalid_out_of_bounds) {
  Board board;
  cr_assert_eq(board.isMoveValid(-1, 0), false,
               "Move (-1,0) should be invalid");
  cr_assert_eq(board.isMoveValid(20, 0), false,
               "Move (20,0) should be invalid");
}

Test(Board, check_no_winner) {
  Board board;
  cr_assert_eq(board.checkWinner(), 0, "No winner should be found");
}
