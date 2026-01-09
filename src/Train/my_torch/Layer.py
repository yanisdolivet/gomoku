##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## Layer
##

import numpy as np


class Layer:

    def __init__(self, inputSize, outputSize, activation_type, dropout_rate=0.0):
        """Initialize a neural network layer.

        Args:
            inputSize (int): Number of inputs to this layer.
            outputSize (int): Number of outputs from this layer.
            activation_type (str): Type of activation function.
            dropout_rate (float): Dropout rate for regularization.
        """
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.dropout_rate = dropout_rate

        limit = np.sqrt(2.0 / self.inputSize)
        self.weights = np.random.randn(self.inputSize, self.outputSize) * limit
        self.biases = np.zeros((1, outputSize))

        self.v_weights = np.zeros((inputSize, outputSize))
        self.v_biases = np.zeros((1, outputSize))
        self.beta = 0.9  # Coefficient de momentum

        # Set activation function
        self.activation_type = activation_type
        if self.activation_type == "relu":
            self.act_func = lambda x: np.maximum(0, x)
        else:
            self.act_func = lambda x: x

        self.cache_input = None
        self.cache_z = None
        self.cache_a = None
        self.dropout_mask = None

    def forward(self, input_data, training=True):
        """Perform forward pass through the layer.

        Args:
            input (numpy.ndarray): Input data, can be 1D or 2D.

        Returns:
            numpy.ndarray: Activated output.
        """
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        self.cache_input = input_data

        z = np.dot(input_data, self.weights) + self.biases
        self.cache_z = z
        a = self.act_func(z)

        if training and self.dropout_rate > 0:
            # Binary mask for dropout
            self.dropout_mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(
                np.float32
            )
            # Multiply by mask and scale to maintain magnitude
            a = (a * self.dropout_mask) / (1.0 - self.dropout_rate)
        else:
            self.dropout_mask = None

        self.cache_a = a
        return a

    def activate_derivative(self, z):
        """Compute the derivative of the activation function at z.

        Args:
            z (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Derivative of the activation function.
        """
        if self.activation_type == "relu":
            return np.where(z > 0, 1, 0)
        elif self.activation_type == "sigmoid":
            sig = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
            return sig * (1 - sig)
        else:
            return np.ones_like(z)

    def backward(self, gradient, learning_rate, lambda_reg=0.0):
        """Perform backward pass (backpropagation).

        Args:
            gradient (numpy.ndarray): Gradient from the next layer.
            learning_rate (float): Learning rate for weight updates.
            lambda_reg (float): L2 regularization strength.

        Returns:
            numpy.ndarray: Gradient to pass to the previous layer.
        """
        if self.dropout_mask is not None:
            gradient = (gradient * self.dropout_mask) / (1.0 - self.dropout_rate)


        derivative = (
            np.where(self.cache_z > 0, 1, 0) if self.activation_type == "relu" else 1
        )
        dZ = gradient * derivative

        # gradients for parameters
        batch_size = self.cache_input.shape[0]
        dW = np.dot(self.cache_input.T, dZ) / batch_size
        dB = np.sum(dZ, axis=0, keepdims=True) / batch_size
        dX_prev = np.dot(dZ, self.weights.T)

        # Add L2 regularization gradient to weight gradient (not to bias)
        if lambda_reg > 0:
            dW += lambda_reg * self.weights

        # Momentum
        self.v_weights = self.beta * self.v_weights + (1 - self.beta) * dW
        self.v_biases = self.beta * self.v_biases + (1 - self.beta) * dB

        # Update weights and biases
        self.weights -= learning_rate * self.v_weights
        self.biases -= learning_rate * self.v_biases

        return dX_prev
