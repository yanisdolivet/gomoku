##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## Network (CORRECTED)
##

import struct
import sys
import copy
import numpy as np
from src.Train.my_torch.Layer import Layer
from src.Train.model_specification import ModelSpecifications

MAGIC_NUMBER = 0x48435254
ERROR_CODE = 84


class Network:
    """Neural Network class managing layers, training, and prediction."""

    def __init__(self, input_size=400, hidden_size=128, output_size=400):
        self.shared_layer = Layer(input_size, hidden_size, "relu", dropout_rate=0.2)
        self.policy_head = Layer(hidden_size, output_size, "linear", dropout_rate=0.2)
        self.value_head = Layer(hidden_size, 1, "tanh", dropout_rate=0.2)

        self.model_spec = ModelSpecifications
        self.model_spec.layer_sizes = [input_size, hidden_size, output_size]
        self.model_spec.learning_rate = 0.005
        self.model_spec.initialization = "he_normal"
        self.model_spec.batch_size = 64
        self.model_spec.epochs = 100
        self.model_spec.lreg = 0.01
        self.model_spec.dropout_rate = 0.2

        self.layerSize = self.model_spec.layer_sizes
        self.layerCount = len(self.layerSize)
        self.layers = []
        self.matrix_input = None
        self.matrix_output = None



    def createLayer(self, weights, biases):
        """Create layers of the network with given weights and biases or initialize them.
        Args:
            weights (list or None): List of weight matrices for each layer or None for initialization.
            biases (list or None): List of bias vectors for each layer or None for initialization.
        """
        self.shared_layer.weights = weights[0]
        self.shared_layer.biases = biases[0]
        self.policy_head.weights = weights[1]
        self.policy_head.biases = biases[1]
        self.value_head.weights = weights[2]
        self.value_head.biases = biases[2]
        self.layers.append(self.shared_layer)
        self.layers.append(self.policy_head)
        self.layers.append(self.value_head)

    def loss_policy(self, predicted_policy, expected_policy, current_batch_size):
        """Compute the cross-entropy loss between predicted and expected policy distributions.
        Args:
            predicted_policy (numpy.ndarray): Predicted policy probabilities.
            expected_policy (numpy.ndarray): Expected policy one-hot encoded vectors.
        Returns:
            float: Cross-entropy loss value.
        """

        # Loss (Cross-Entropy)
        epsilon = 1e-15
        predicted_clipped = np.clip(predicted_policy, epsilon, 1 - epsilon)
        loss = (
            -np.sum(expected_policy * np.log(predicted_clipped))
            / current_batch_size
        )
        return loss

    def loss_value(self, predicted_value, expected_value):
        """Compute the mean squared error loss between predicted and expected values.
        Args:
            predicted_value (numpy.ndarray): Predicted value outputs.
            expected_value (numpy.ndarray): Expected value outputs.
        Returns:
            float: Mean squared error loss value.
        """
        loss = np.mean((predicted_value - expected_value) ** 2)
        return loss

    def train(
        self,
        saveFile,
        X_val=None,
        Y_val=None,
        X_train=None,
        Y_train=None,
    ):
        """Train the neural network using mini-batch gradient descent with L2 regularization.
        Args:
            saveFile (str): File path to save the trained model.
            X_val (numpy.ndarray or None): Validation input data.
            Y_val (numpy.ndarray or None): Validation output data.
            X_train (numpy.ndarray or None): Training input data.
            Y_train (numpy.ndarray or None): Training output data.
        """
        if X_train is not None and Y_train is not None:
            self.matrix_input = X_train
            self.matrix_output = Y_train
        num_samples = len(self.matrix_input)
        print(f"Starting training on {num_samples} samples...")

        batch_size = self.model_spec.batch_size
        learningRate = self.model_spec.learning_rate

        best_val_acc = 0.0
        best_weights = None
        best_biases = None

        for epoch in range(self.model_spec.epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = self.matrix_input[indices]
            Y_shuffled = self.matrix_output[indices]

            total_loss = 0.0
            total_correct = 0

            for i in range(0, num_samples, batch_size):
                input_data = X_shuffled[i : i + batch_size]
                expected_output = Y_shuffled[i : i + batch_size]
                current_batch_size = len(input_data)

                # Forward
                predicted_policy, predicted_value = self.forward(input_data, training=True)

                # Loss Calculation
                loss_policy = self.loss_policy(predicted_policy, expected_output, current_batch_size)
                loss_value = self.loss_value(predicted_value, expected_output)

                # L2 regularization
                w = 0
                for i in range(len(self.layers)):
                    w += np.sum(np.square(self.layers[i].weights))
                scale_w = (self.model_spec.lreg / 2) * w
                total_loss += (loss_policy + loss_value + scale_w) * current_batch_size

                # Accuracy Train
                train_preds = np.argmax(predicted_policy, axis=1)
                train_labels = np.argmax(expected_output, axis=1)
                total_correct += np.sum(train_preds == train_labels)

                # Backward
                gradient = predicted_policy - expected_output
                self.backward(gradient, learningRate, self.model_spec.lreg)

            # Metrics
            avg_loss = total_loss / num_samples
            train_acc = total_correct / num_samples

            val_msg = ""
            if X_val is not None and Y_val is not None:
                val_output = self.forward(X_val, training=False)
                val_preds = np.argmax(val_output, axis=1)
                val_truth = np.argmax(Y_val, axis=1)
                val_acc = np.mean(val_preds == val_truth)
                val_loss = -np.sum(
                    Y_val * np.log(np.clip(val_output, 1e-15, 1 - 1e-15))
                ) / len(X_val)
                val_msg = f" - Val Acc: {val_acc:.2%}"

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = [copy.deepcopy(l.weights) for l in self.layers]
                    best_biases = [copy.deepcopy(l.biases) for l in self.layers]

            print(
                f"Epoch {epoch + 1}/{self.model_spec.epochs} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2%}{val_msg}"
            )

            # Learning rate decay
            if (epoch + 1) % 20 == 0:
                learningRate *= 0.9

        # Restore the best model
        if best_weights is not None:
            print(f"Restoring best model (Val Acc: {best_val_acc:.2%})")
            for i, layer in enumerate(self.layers):
                layer.weights = best_weights[i]
                layer.biases = best_biases[i]

        self.saveTrainedNetwork(saveFile)

    def forward(self, input, training=True) -> np.array:
        """Perform forward pass through the network.
        Args:
            input (numpy.ndarray): Input data.
            training (bool): Whether in training mode (affects dropout).
        """
        hidden = self.shared_layer.forward(input, training)

        policy_output = self.policy_head.forward(hidden, training)

        # Softmax
        shift = policy_output - np.max(policy_output, axis=1, keepdims=True)
        exps = np.exp(shift)
        policy_probs = exps / np.sum(exps, axis=1, keepdims=True)

        value_output = self.value_head.forward(hidden, training)
        value_pred = np.tanh(value_output)

        return policy_probs, value_pred

    def backward(self, gradient, learning_rate, lambda_reg):
        """Perform backward pass through all layers with L2 regularization.

        Args:
            gradient (numpy.ndarray): Initial gradient from loss function.
            learning_rate (float): Learning rate for updates.
            lambda_reg (float): L2 regularization strength.

        Returns:
            numpy.ndarray: Gradient propagated to input layer.
        """
        current_gradient = gradient
        for i in range(len(self.layers) - 1, -1, -1):
            current_gradient = self.layers[i].backward(
                current_gradient, learning_rate, lambda_reg
            )
        return current_gradient

    def _encode_string(self, s):
        """Encode string as length-prefixed bytes.
        Args:
            s (str): String to encode.
        Returns:
            bytes: Encoded string with length prefix.
        """
        encoded = s.encode("utf-8")
        return struct.pack("I", len(encoded)) + encoded

    def saveTrainedNetwork(self, filePath):
        """Save the trained network to a binary file with full configuration.
        Args:
            filePath (str): Path to save the network file.
        """
        try:
            VERSION = 2
            with open(filePath, "wb") as f:
                f.write(struct.pack("III", MAGIC_NUMBER, VERSION, len(self.layerSize)))
                f.write(struct.pack(f"{len(self.layerSize)}I", *self.layerSize))

                for i in range(len(self.model_spec.type)):
                    f.write(self._encode_string(self.model_spec.type[i]))
                    f.write(self._encode_string(self.model_spec.activation[i]))

                f.write(struct.pack("f", self.model_spec.learning_rate))
                f.write(self._encode_string(self.model_spec.initialization))

                f.write(struct.pack("I", self.model_spec.batch_size))
                f.write(struct.pack("I", self.model_spec.epochs))
                f.write(struct.pack("f", self.model_spec.lreg))
                f.write(struct.pack("f", self.model_spec.dropout_rate))
                f.write(self._encode_string(self.model_spec.loss_function))

                for layer in self.layers:
                    w_flat = layer.weights.astype(np.float32).flatten()
                    f.write(struct.pack(f"{len(w_flat)}f", *w_flat))
                for layer in self.layers:
                    b_flat = layer.biases.astype(np.float32).flatten()
                    f.write(struct.pack(f"{len(b_flat)}f", *b_flat))

            print(f"Saved trained network to {filePath}")
        except IOError as e:
            print(f"IOError: {e}", file=sys.stderr)
            sys.exit(ERROR_CODE)
