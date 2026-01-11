/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** VCF
*/

#pragma once

#include "Board.hpp"
#include <utility>

class VCF {
public:
  static std::pair<int, int> solve(const Board &board, int player, int depth);

private:
  static bool vcf(const Board &board, int player, int depth, int &winningMove);
};
