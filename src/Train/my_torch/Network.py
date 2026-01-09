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

    def __init__(self, input_size=800, shared_size1=512, shared_size2=256, output_size=400):
        self.shared_layer1 = Layer(input_size, shared_size1, "relu", dropout_rate=0.2)
        self.shared_layer2 = Layer(shared_size1, shared_size2, "relu", dropout_rate=0.2)
        self.policy_head = Layer(shared_size2, output_size, "linear", dropout_rate=0.2)
        self.value_head = Layer(shared_size2, 1, "tanh", dropout_rate=0.2)

        self.model_spec = ModelSpecifications
        self.model_spec.layer_sizes = [input_size, shared_size1, shared_size2, output_size]
        self.model_spec.learning_rate = 0.005
        self.model_spec.initialization = "he_normal"
        self.model_spec.batch_size = 64
        self.model_spec.epochs = 20
        self.model_spec.lreg = 0.01
        self.model_spec.dropout_rate = 0.2

        self.layerSize = self.model_spec.layer_sizes
        self.layerCount = len(self.layerSize)
        self.layers = []
        self.matrix_input = None
        self.matrix_policy_output = None
        self.matrix_value_output = None



    def createLayer(self, weights, biases):
        """Create layers of the network with given weights and biases or initialize them.
        Args:
            weights (list or None): List of weight matrices for each layer or None for initialization.
            biases (list or None): List of bias vectors for each layer or None for initialization.
        """
        self.shared_layer1.weights = weights[0]
        self.shared_layer1.biases = biases[0]
        self.shared_layer2.weights = weights[1]
        self.shared_layer2.biases = biases[1]
        self.policy_head.weights = weights[2]
        self.policy_head.biases = biases[2]
        self.value_head.weights = weights[3]
        self.value_head.biases = biases[3]
        self.layers.append(self.shared_layer1)
        self.layers.append(self.shared_layer2)
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
        X_train=None,
        Y_policy_val=None,
        Y_policy_train=None,
        Y_value_val=None,
        Y_value_train=None,
    ):
        """Train the neural network using mini-batch gradient descent with L2 regularization.
        Args:
            saveFile (str): File path to save the trained model.
            X_val (numpy.ndarray or None): Validation input data.
            Y_policy_val (numpy.ndarray or None): Validation policy output data.
            Y_value_val (numpy.ndarray or None): Validation value output data.
            X_train (numpy.ndarray or None): Training input data.
            Y_policy_train (numpy.ndarray or None): Training policy output data.
            Y_value_train (numpy.ndarray or None): Training value output data.
        """
        if X_train is not None and Y_policy_train is not None and Y_value_train is not None:
            self.matrix_input = X_train
            self.matrix_policy_output = Y_policy_train
            self.matrix_value_output = Y_value_train
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
            Y_policy_shuffled = self.matrix_policy_output[indices]
            Y_value_shuffled = self.matrix_value_output[indices]

            total_loss = 0.0
            total_correct = 0
            total_top5 = 0

            for i in range(0, num_samples, batch_size):
                input_data = X_shuffled[i : i + batch_size]
                policy_batch = Y_policy_shuffled[i : i + batch_size]
                value_batch = Y_value_shuffled[i : i + batch_size]
                current_batch_size = len(input_data)

                # Forward
                predicted_policy, predicted_value = self.forward(input_data, training=True)

                # Loss Calculation
                loss_policy = self.loss_policy(predicted_policy, policy_batch, current_batch_size)
                loss_value = self.loss_value(predicted_value, value_batch)

                # L2 regularization
                w = 0
                for i in range(len(self.layers)):
                    w += np.sum(np.square(self.layers[i].weights))
                scale_w = (self.model_spec.lreg / 2) * w
                total_loss += (loss_policy + loss_value + scale_w) * current_batch_size

                # Accuracy Train
                train_preds = np.argmax(predicted_policy, axis=1)
                train_labels = np.argmax(policy_batch, axis=1)
                total_correct += np.sum(train_preds == train_labels)

                # Top-5 Accuracy
                top5_preds = np.argsort(predicted_policy, axis=1)[:, -5:]
                true_indices = np.argmax(policy_batch, axis=1).reshape(-1, 1)
                in_top5 = np.any(top5_preds == true_indices, axis=1)
                total_top5 += np.sum(in_top5)

                # Backward
                self.backward(predicted_policy, predicted_value, policy_batch, value_batch, learningRate, self.model_spec.lreg)

            # Metrics
            avg_loss = total_loss / num_samples
            train_acc = total_correct / num_samples

            val_msg = ""
            if X_val is not None and Y_policy_val is not None and Y_value_val is not None:
                predicted_policy_val, predicted_value_val = self.forward(X_val, training=False)
                val_preds = np.argmax(predicted_policy_val, axis=1)
                val_truth = np.argmax(Y_policy_val, axis=1)
                val_acc = np.mean(val_preds == val_truth)
                val_loss_policy = -np.sum(
                    Y_policy_val * np.log(np.clip(predicted_policy_val, 1e-15, 1 - 1e-15))
                ) / len(X_val)
                val_loss_value = np.mean((predicted_value_val - Y_value_val) ** 2)
                val_msg = f" - Val Acc: {val_acc:.2%} - Val Loss Policy: {val_loss_policy:.4f} - Val Loss Value: {val_loss_value:.4f}"

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = [copy.deepcopy(l.weights) for l in self.layers]
                    best_biases = [copy.deepcopy(l.biases) for l in self.layers]

            print(
                f"Epoch {epoch + 1}/{self.model_spec.epochs} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2%}{val_msg}, Top-5 Acc: {total_top5 / num_samples:.2%}"
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

        self.save_for_cpp(saveFile)

    def forward(self, input, training=True) -> np.array:
        """Perform forward pass through the network.
        Args:
            input (numpy.ndarray): Input data.
            training (bool): Whether in training mode (affects dropout).
        """
        hidden1 = self.shared_layer1.forward(input, training)
        hidden2 = self.shared_layer2.forward(hidden1, training)

        policy_output = self.policy_head.forward(hidden2, training)

        # Softmax
        shift = policy_output - np.max(policy_output, axis=1, keepdims=True)
        exps = np.exp(shift)
        policy_probs = exps / np.sum(exps, axis=1, keepdims=True)

        value_output = self.value_head.forward(hidden2, training)
        value_pred = np.tanh(value_output)

        return policy_probs, value_pred

    def backward(self, predicted_policy, predicted_value, expected_policy, expected_val, learning_rate, lambda_reg):
        """Perform backward pass through all layers with L2 regularization.

        Args:
            gradient (numpy.ndarray): Initial gradient from loss function.
            learning_rate (float): Learning rate for updates.
            lambda_reg (float): L2 regularization strength.

        Returns:
            numpy.ndarray: Gradient propagated to input layer.
        """
        gradient_policy = predicted_policy - expected_policy
        batch_size = predicted_value.shape[0]
        gradient_val = 2 * (predicted_value - expected_val) / batch_size

        gradient_hiddend_policy = self.policy_head.backward(gradient_policy, learning_rate, lambda_reg)
        gradient_hiddend_val = self.value_head.backward(gradient_val, learning_rate, lambda_reg)

        total_hidden_gradient = gradient_hiddend_policy + gradient_hiddend_val

        grad_layer1 = self.shared_layer2.backward(total_hidden_gradient, learning_rate, lambda_reg)
        current_gradient = self.shared_layer1.backward(grad_layer1, learning_rate, lambda_reg)
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

    def save_for_cpp(self, filepath):
        """Exports weights in raw binary format matching C++ loadModel order:
           1. Shared 1 Weights (800x512)
           2. Shared 1 Biases (512)
           3. Shared 2 Weights (512x256)
           4. Shared 2 Biases (256)
           5. Policy Weights (256x400)
           6. Policy Biases (400)
           7. Value Weights (256x1)
           8. Value Biases (1)
        """
        try:
            with open(filepath, "wb") as f:
                # 1. Shared Layer
                f.write(self.shared_layer1.weights.astype(np.float32).tobytes())
                f.write(self.shared_layer1.biases.astype(np.float32).tobytes())

                f.write(self.shared_layer2.weights.astype(np.float32).tobytes())
                f.write(self.shared_layer2.biases.astype(np.float32).tobytes())

                # 2. Policy Head
                f.write(self.policy_head.weights.astype(np.float32).tobytes())
                f.write(self.policy_head.biases.astype(np.float32).tobytes())

                # 3. Value Head
                f.write(self.value_head.weights.astype(np.float32).tobytes())
                f.write(self.value_head.biases.astype(np.float32).tobytes())

            print(f"Successfully exported raw model to {filepath}")
        except IOError as e:
            print(f"Error saving model: {e}", file=sys.stderr)
