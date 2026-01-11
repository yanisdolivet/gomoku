/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Tensor
*/

#pragma once

#include <vector>

/**
 * @brief Tensor struct for multi-dimensional arrays
 * @note This struct encapsulates the properties and methods for handling
 * multi-dimensional arrays (tensors) used in neural network computations.
 */

struct Tensor {
  std::vector<float> values;
  int rows;
  int cols;
  int channels;
  int num;

  Tensor() : rows(0), cols(0), channels(1), num(1) {}

  Tensor(int r, int c) : rows(r), cols(c), channels(1), num(1) {
    values.resize(r * c, 0.0f);
  }

  Tensor(int n, int ch, int r, int c) : rows(r), cols(c), channels(ch), num(n) {
    values.resize(n * ch * r * c, 0.0f);
  }

  inline float &operator[](int index) { return values[index]; }

  inline const float &operator[](int index) const { return values[index]; }

  inline float &at(int r, int c) { return values[r * cols + c]; }

  inline const float &at(int r, int c) const { return values[r * cols + c]; }

  inline float &at(int n, int ch, int r, int c) {
    return values[((n * channels + ch) * rows + r) * cols + c];
  }

  inline const float &at(int n, int ch, int r, int c) const {
    return values[((n * channels + ch) * rows + r) * cols + c];
  }

  inline int size() const { return values.size(); }
};