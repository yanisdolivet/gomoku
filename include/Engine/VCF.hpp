/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** VCF
*/

#pragma once

#include "../Core/Board.hpp"
#include <bitset>
#include <utility>
#include <vector>

class VCF {
public:
  static std::pair<int, int> solve(const Board &board, int player, int depth);

private:
  static bool vcf(Board board, int player, int depth, int &winningMove);
};
