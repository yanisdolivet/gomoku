import struct
import sys
import copy
import numpy as np
import time

from src.Train.my_torch.Conv2d import Conv2d
from src.Train.my_torch.BatchNorm2d import BatchNorm2d
from src.Train.my_torch.ResBlock import ResBlock
from src.Train.model_specification import ModelSpecifications

MAGIC_NUMBER = 0x5245534E # RESN (ResNet)
ERROR_CODE = 84

class Network:
    """ResNet Neural Network class managing layers, training, and prediction."""

    def __init__(self, input_channels=5, board_size=20, num_res_blocks=5, num_filters=64):
        self.input_channels = input_channels
        self.board_size = board_size
        self.num_res_blocks = num_res_blocks
        
        # --- Architecture Definition ---
        
        # Initial Block
        self.conv_input = Conv2d(input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn_input = BatchNorm2d(num_filters)
        
        # Residual Tower
        self.backbone = []
        for _ in range(num_res_blocks):
            self.backbone.append(ResBlock(num_filters))
            
        # Policy Head (Fully Convolutional)
        # 1x1 Conv to reduce to 1 channel (representing logits for each cell)
        self.policy_conv = Conv2d(num_filters, 1, kernel_size=1, stride=1, padding=0)
        self.policy_bn = BatchNorm2d(1)
        
        # Value Head (Fully Convolutional)
        # 1x1 Conv to reduce to 1 channel
        self.value_conv = Conv2d(num_filters, 1, kernel_size=1, stride=1, padding=0)
        self.value_bn = BatchNorm2d(1)
        # Reduction Conv: 20x20 kernel on 20x20 input -> 1x1 output (Scalar value)
        self.value_reduce = Conv2d(1, 1, kernel_size=board_size, stride=1, padding=0)
        
        # -----------------------------
        
        self.model_spec = ModelSpecifications
        self.model_spec.learning_rate = 0.01
        self.model_spec.batch_size = 64
        self.model_spec.epochs = 20
        self.model_spec.lreg = 0.0001
        
        self.cache_input = None

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
        """Train the neural network using mini-batch gradient descent with L2 regularization."""
        if X_train is not None:
            self.matrix_input = X_train
            self.matrix_policy_output = Y_policy_train
            self.matrix_value_output = Y_value_train
        else:
            print("Error: No training data provided")
            return

        num_samples = len(self.matrix_input)
        print(f"Starting training on {num_samples} samples...")

        batch_size = self.model_spec.batch_size
        learningRate = self.model_spec.learning_rate

        best_val_acc = 0.0
        
        for epoch in range(self.model_spec.epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = self.matrix_input[indices]
            Y_policy_shuffled = self.matrix_policy_output[indices]
            Y_value_shuffled = self.matrix_value_output[indices]

            total_loss = 0.0
            total_correct = 0
            
            start_time = time.time()

            for i in range(0, num_samples, batch_size):
                input_data = X_shuffled[i : i + batch_size]
                policy_batch = Y_policy_shuffled[i : i + batch_size]
                value_batch = Y_value_shuffled[i : i + batch_size]
                current_batch_size = len(input_data)

                # Forward
                predicted_policy, predicted_value = self.forward(input_data, training=True)

                # Loss Calculation
                epsilon = 1e-15
                predicted_clipped = np.clip(predicted_policy, epsilon, 1 - epsilon)
                loss_policy = -np.sum(policy_batch * np.log(predicted_clipped)) / current_batch_size
                loss_value = np.mean((predicted_value - value_batch) ** 2)

                total_loss += (loss_policy + loss_value) * current_batch_size

                # Accuracy Train
                train_preds = np.argmax(predicted_policy, axis=1)
                train_labels = np.argmax(policy_batch, axis=1)
                total_correct += np.sum(train_preds == train_labels)

                # Backward
                self.backward(predicted_policy, predicted_value, policy_batch, value_batch, learningRate, self.model_spec.lreg)
            
            epoch_time = time.time() - start_time
            avg_loss = total_loss / num_samples
            train_acc = total_correct / num_samples
            
            val_msg = ""
            if X_val is not None:
                predicted_policy_val, predicted_value_val = self.forward(X_val, training=False)
                val_preds = np.argmax(predicted_policy_val, axis=1)
                val_truth = np.argmax(Y_policy_val, axis=1)
                val_acc = np.mean(val_preds == val_truth)
                val_loss_value = np.mean((predicted_value_val - Y_value_val) ** 2)
                val_msg = f" - Val Acc: {val_acc:.2%} - Val Loss Val: {val_loss_value:.4f}"
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    pass

            print(f"Epoch {epoch + 1}/{self.model_spec.epochs} ({epoch_time:.2f}s) - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2%}{val_msg}")

            # Learning rate decay
            if (epoch + 1) % 5 == 0:
                learningRate *= 0.5

        self.save_for_cpp(saveFile)


    def relu(self, x):
        return np.maximum(0, x)
        
    def d_relu(self, x):
        return (x > 0).astype(np.float32)

    def forward(self, x_flat, training=True):
        """
        x_flat: (Batch, Inputs) -> Reshape to (Batch, 5, 20, 20)
        """
        batch_size = x_flat.shape[0]
        x = x_flat.reshape(batch_size, self.input_channels, self.board_size, self.board_size)
        
        # Initial Block
        conv_out_pre = self.conv_input.forward(x, training)
        bn_out = self.bn_input.forward(conv_out_pre, training)
        out = self.relu(bn_out)
        
        if training:
            self.cache_initial_relu_input = bn_out # Store for d_relu in backward

        # Backbone
        for block in self.backbone:
            out = block.forward(out, training)
            
        # --- Policy Head ---
        p = self.policy_conv.forward(out, training)
        p = self.policy_bn.forward(p, training)
        p = self.relu(p)
        
        if training:
             # Store output for relu derivative (assuming relu(x) derivative depends on x or output).
             # d(relu(x))/dx = 1 if x>0.
             # We can use p (the output) to approximate: p>0 means x>0.
             # But if p=0, x could be <=0.
             # This is sufficient for Backprop.
             self.cache_policy_relu_output = p

        # Flatten directly (1x20x20 -> 400)
        policy_logits = p.reshape(batch_size, -1)
        
        # Solfmax
        shift = policy_logits - np.max(policy_logits, axis=1, keepdims=True)
        exps = np.exp(shift)
        policy_probs = exps / np.sum(exps, axis=1, keepdims=True)
        
        # --- Value Head ---
        v = self.value_conv.forward(out, training)
        v = self.value_bn.forward(v, training)
        v = self.relu(v)
        
        if training:
            self.cache_value_relu_output = v
            
        # Reduce (Conv 20x20 -> 1x1)
        v_reduced = self.value_reduce.forward(v, training)
        
        v_flat = v_reduced.reshape(batch_size, -1)
        value_pred = np.tanh(v_flat)
        
        if training:
            self.cache_forward = (out, p, v, v_flat, v_reduced) 
            
        return policy_probs, value_pred

    def backward(self, predicted_policy, predicted_value, expected_policy, expected_val, learning_rate, lambda_reg):
        """
        Correct gradient flow for Heads -> Backbone -> Input
        """
        backbone_out, p_vol, v_vol, v_flat, v_reduced = self.cache_forward
        batch_size = predicted_value.shape[0]

        # --- Policy Head Backward ---
        # dL/dLogits = P - Target
        d_p_logits = (predicted_policy - expected_policy) / batch_size
        
        # Reshape to Conv output shape (Batch, 1, 20, 20)
        d_p = d_p_logits.reshape(p_vol.shape)
        
        # ReLU derivative using stored output
        d_p = d_p * self.d_relu(self.cache_policy_relu_output)
        
        # BN
        d_p = self.policy_bn.backward(d_p, learning_rate, lambda_reg)
        
        # Conv
        d_p_head_in = self.policy_conv.backward(d_p, learning_rate, lambda_reg)


        # --- Value Head Backward ---
        # Tanh derivative: 1 - tanh^2
        d_tanh = 1 - predicted_value ** 2
        
        # MSE Gradient: 2 * (Pred - Target) / N
        d_v_loss = 2 * (predicted_value - expected_val) / batch_size 
        
        d_v_flat = d_v_loss * d_tanh
        
        # Reshape to (Batch, 1, 1, 1) if using Reduce Conv
        d_v_reduced = d_v_flat.reshape(v_reduced.shape)
        
        # Reduce Conv Backward
        d_v_reduce_in = self.value_reduce.backward(d_v_reduced, learning_rate, lambda_reg)
        
        # ReLU (using stored output)
        d_v = d_v_reduce_in * self.d_relu(self.cache_value_relu_output)
        
        # BN
        d_v = self.value_bn.backward(d_v, learning_rate, lambda_reg)
        
        # Conv
        d_v_head_in = self.value_conv.backward(d_v, learning_rate, lambda_reg)
        
        
        # --- Backbone Backward ---
        total_gradient = d_p_head_in + d_v_head_in
        
        for block in reversed(self.backbone):
            total_gradient = block.backward(total_gradient, learning_rate, lambda_reg)
            
        # --- Initial Block Backward ---
        # ReLU 
        d_out = total_gradient * self.d_relu(self.cache_initial_relu_input)
        
        d_bn = self.bn_input.backward(d_out, learning_rate, lambda_reg)
        d_in = self.conv_input.backward(d_bn, learning_rate, lambda_reg)
        
        return d_in

    def save_for_cpp(self, filepath):
        try:
            with open(filepath, "wb") as f:
                # Magic Number to identify ResNet vs MLP
                f.write(struct.pack("I", MAGIC_NUMBER)) 
                
                # Model Hyperparameters needed for loading
                f.write(struct.pack("I", self.input_channels))
                f.write(struct.pack("I", self.num_res_blocks))
                
                # --- Initial Block ---
                f.write(self.conv_input.weights.astype(np.float32).tobytes())
                f.write(self.conv_input.biases.astype(np.float32).tobytes())
                
                f.write(self.bn_input.gamma.astype(np.float32).tobytes())
                f.write(self.bn_input.beta.astype(np.float32).tobytes())
                f.write(self.bn_input.running_mean.astype(np.float32).tobytes())
                f.write(self.bn_input.running_var.astype(np.float32).tobytes())
                
                # --- Backbone ---
                for block in self.backbone:
                    # Conv1
                    f.write(block.conv1.weights.astype(np.float32).tobytes())
                    f.write(block.conv1.biases.astype(np.float32).tobytes())
                    f.write(block.bn1.gamma.astype(np.float32).tobytes())
                    f.write(block.bn1.beta.astype(np.float32).tobytes())
                    f.write(block.bn1.running_mean.astype(np.float32).tobytes())
                    f.write(block.bn1.running_var.astype(np.float32).tobytes())
                    
                    # Conv2
                    f.write(block.conv2.weights.astype(np.float32).tobytes())
                    f.write(block.conv2.biases.astype(np.float32).tobytes())
                    f.write(block.bn2.gamma.astype(np.float32).tobytes())
                    f.write(block.bn2.beta.astype(np.float32).tobytes())
                    f.write(block.bn2.running_mean.astype(np.float32).tobytes())
                    f.write(block.bn2.running_var.astype(np.float32).tobytes())
                    
                # --- Heads ---
                # Policy
                f.write(self.policy_conv.weights.astype(np.float32).tobytes())
                f.write(self.policy_conv.biases.astype(np.float32).tobytes())
                f.write(self.policy_bn.gamma.astype(np.float32).tobytes())
                f.write(self.policy_bn.beta.astype(np.float32).tobytes())
                f.write(self.policy_bn.running_mean.astype(np.float32).tobytes())
                f.write(self.policy_bn.running_var.astype(np.float32).tobytes())
                # NO FC LAYERS
                
                # Value
                f.write(self.value_conv.weights.astype(np.float32).tobytes())
                f.write(self.value_conv.biases.astype(np.float32).tobytes())
                f.write(self.value_bn.gamma.astype(np.float32).tobytes())
                f.write(self.value_bn.beta.astype(np.float32).tobytes())
                f.write(self.value_bn.running_mean.astype(np.float32).tobytes())
                f.write(self.value_bn.running_var.astype(np.float32).tobytes())
                
                # Reduction Conv (Replaces FC)
                f.write(self.value_reduce.weights.astype(np.float32).tobytes())
                f.write(self.value_reduce.biases.astype(np.float32).tobytes())

            print(f"Successfully exported ResNet model to {filepath}")
            
        except IOError as e:
            print(f"Error saving model: {e}", file=sys.stderr)
