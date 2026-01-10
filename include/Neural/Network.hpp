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
        Tensor _inputBuffer;        // Input tensor buffer (1, 1200)
        Tensor _hidden1Buffer;      // First hidden layer output (1, 512)
        Tensor _hidden2Buffer;      // Second hidden layer output (1, 256)
        Tensor _policyLogitsBuffer; // Policy logits tensor buffer (1, 400)
        Tensor _valueOutBuffer;     // Value output tensor buffer (1, 1)

        Tensor _weightsShared1;     // Weights for first shared layer (1200, 512)
        Tensor _biasesShared1;      // Biases for first shared layer (1, 512)
        Tensor _weightsShared2;     // Weights for second shared layer (512, 256)
        Tensor _biasesShared2;      // Biases for second shared layer (1, 256)
        Tensor _weightsPolicy;      // Weights for policy head (256, 400)
        Tensor _biasesPolicy;       // Biases for policy head (1, 400)
        Tensor _weightsValue;       // Weights for value head (256, 1)
        Tensor _biasesValue;        // Biases for value head (1, 1)

        void denseLayer(const Tensor &input, const Tensor &weights, const Tensor &biases, Tensor &output);
        void relu(Tensor &tensor);
        void softmax(std::vector<float> &values);

};