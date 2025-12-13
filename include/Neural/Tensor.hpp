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
    std::vector<float> values; // 400 value for 20x20 board
    int rows; // number of rows
    int cols; // number of columns

    /**
     * @brief Default constructor for Tensor
     */
    Tensor(): rows(0), cols(0) {}

    /**
     * @brief Parameterized constructor for Tensor
     * @param r Number of rows
     * @param c Number of columns
     */
    Tensor(int r, int c) : rows(r), cols(c) {
        values.resize(r * c, 0.0f);
    }

    /**
     * @brief Overloaded subscript operator for accessing tensor elements
     * @param index Index of the element
     * @return Reference to the element at the specified index
     */
    inline float &operator[](int index) {
        return values[index];
    }

    /**
     * @brief Overloaded subscript operator for accessing tensor elements (const version)
     * @param index Index of the element
     * @return Const reference to the element at the specified index
     */
    inline const float &operator[](int index) const {
        return values[index];
    }

    /**
     * @brief Method to access tensor elements using row and column indices
     * @param r Row index
     * @param c Column index
     * @return Reference to the element at the specified row and column
     */
    inline float &at(int r, int c) {
        return values[r * cols + c];
    }

    /**
     * @brief Method to know size of the tensor
     */
    inline int size() const {
        return values.size();
    }
};