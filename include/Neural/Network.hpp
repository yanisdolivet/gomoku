/*
** EPITECH PROJECT, 2025
** Delivery
** File description:
** Network
*/

#pragma once

#include "Tensor.hpp"
#include "../Utils/Logger.hpp"
#include "../Core/Board.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>


/**
 * @brief Output struct for Neural Network predictions
 * @note This struct holds the policy and value outputs from the neural network.
 */
struct Output {
    std::vector<float> policy; // Probability distribution over possible moves
    float value;
};

/**
 * @brief Neural Network class for Gomoku
 * @note This class encapsulates the neural network used for decision making in
 * the Gomoku game. It is a interference motor. It contains the structure of the
 * network. It contains container for mathematical data (1D, 2D, 3D Matrix).
 */

class Network {
    public:
        Network();
        ~Network();

        bool loadModel(const std::string &modelPath);
        Output predict(const Board &board);

    private:
        Tensor _inputBuffer; // Input tensor buffer
        Tensor _hiddenBuffer; // Hidden layer tensor buffer
        Tensor _policyLogitsBuffer; // Policy logits tensor buffer
        Tensor _valueOutBuffer; // Value output tensor buffer

        Tensor _weights1; // Weights for the first layer
        Tensor _biases1;  // Biases for the first layer
        Tensor _weights2; // Weights for the second layer
        Tensor _biases2;  // Biases for the second layer
        Tensor _weightsPolicy; // Weights for the policy output layer
        Tensor _biasesPolicy;  // Biases for the policy output layer

        void denseLayer(const Tensor &input, const Tensor &weights, const Tensor &biases, Tensor &output);
        void relu(Tensor &tensor);
        void softmax(std::vector<float> &values);

};