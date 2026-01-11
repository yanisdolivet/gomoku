import torch
import torch.nn as nn
import torch.optim as optim
import struct
import sys
import numpy as np
import time

MAGIC_NUMBER = 0x5245534E # RESN (ResNet)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class GomokuResNet(nn.Module):
    def __init__(self, input_channels=5, board_size=20, num_res_blocks=5, num_filters=64):
        super(GomokuResNet, self).__init__()
        self.input_channels = input_channels
        self.num_res_blocks = num_res_blocks
        self.board_size = board_size
        
        # Initial Block
        self.conv_input = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=True)
        self.bn_input = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        
        # Backbone
        self.backbone = nn.ModuleList([ResBlock(num_filters) for _ in range(num_res_blocks)])
        
        # Policy Head
        # Conv 1x1 -> 1 Channel -> Flatten -> Softmax (handled by CrossEntropyLoss usually, but outputting logits here)
        self.policy_conv = nn.Conv2d(num_filters, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.policy_bn = nn.BatchNorm2d(1)
        
        # Value Head
        # Conv 1x1 -> 1 Channel -> Reduce 20x20 -> Scalar
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_reduce = nn.Conv2d(1, 1, kernel_size=board_size, stride=1, padding=0, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (Batch, Channels, H, W)
        out = self.conv_input(x)
        out = self.bn_input(out)
        out = self.relu(out)
        
        for block in self.backbone:
            out = block(out)
            
        # Policy
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = self.relu(p)
        p_logits = p.view(p.size(0), -1) # Flatten (Batch, 400)
        # Softmax is computed in loss function usually, but for inference we return logits/probs
        p_probs = torch.softmax(p_logits, dim=1)
        
        # Value
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = self.relu(v)
        v = self.value_reduce(v) # (Batch, 1, 1, 1)
        v = v.view(v.size(0), -1) # (Batch, 1)
        v_pred = self.tanh(v)
        
        return p_probs, v_pred
    
    def save_for_cpp(self, filepath):
        """Export weights to binary format compatible with C++ engine"""
        print(f"Exporting model to {filepath}...")
        try:
            with open(filepath, "wb") as f:
                # Magic
                f.write(struct.pack("I", MAGIC_NUMBER))
                f.write(struct.pack("I", self.input_channels))
                f.write(struct.pack("I", self.num_res_blocks))
                
                # Helper to write tensor
                def write_tensor(tensor):
                    # Move to cpu, convert to numpy float32, tobytes
                    f.write(tensor.detach().cpu().numpy().astype(np.float32).tobytes())

                # Initial Block
                write_tensor(self.conv_input.weight)
                write_tensor(self.conv_input.bias)
                write_tensor(self.bn_input.weight) # gamma
                write_tensor(self.bn_input.bias)   # beta
                write_tensor(self.bn_input.running_mean)
                write_tensor(self.bn_input.running_var)
                
                # Backbone
                for block in self.backbone:
                    # Conv1
                    write_tensor(block.conv1.weight)
                    write_tensor(block.conv1.bias)
                    write_tensor(block.bn1.weight)
                    write_tensor(block.bn1.bias)
                    write_tensor(block.bn1.running_mean)
                    write_tensor(block.bn1.running_var)
                    # Conv2
                    write_tensor(block.conv2.weight)
                    write_tensor(block.conv2.bias)
                    write_tensor(block.bn2.weight)
                    write_tensor(block.bn2.bias)
                    write_tensor(block.bn2.running_mean)
                    write_tensor(block.bn2.running_var)
                    
                # Policy Head
                write_tensor(self.policy_conv.weight)
                write_tensor(self.policy_conv.bias)
                write_tensor(self.policy_bn.weight)
                write_tensor(self.policy_bn.bias)
                write_tensor(self.policy_bn.running_mean)
                write_tensor(self.policy_bn.running_var)
                
                # Value Head
                write_tensor(self.value_conv.weight)
                write_tensor(self.value_conv.bias)
                write_tensor(self.value_bn.weight)
                write_tensor(self.value_bn.bias)
                write_tensor(self.value_bn.running_mean)
                write_tensor(self.value_bn.running_var)
                
                write_tensor(self.value_reduce.weight)
                write_tensor(self.value_reduce.bias)
                
            print(f"Successfully exported ResNet model to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}", file=sys.stderr)

class Network:
    """Wrapper to act like the Training interface"""
    def __init__(self, input_channels=5, board_size=20, num_res_blocks=5, num_filters=64):
        # Detect device
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
        
        self.model = GomokuResNet(input_channels, board_size, num_res_blocks, num_filters).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        
        # Loss functions
        # Policy: Cross Entropy (needs logits, but our model outputs probs, so use NLLLoss with log(probs))
        # Or better change model to output logits.
        # Current 'Network.py' returns probabilities.
        # Let's use MSE for value.
        self.value_criterion = nn.MSELoss()
        
    def train(self, saveFile, X_val, X_train, Y_policy_val, Y_policy_train, Y_value_val, Y_value_train):
        epochs = 10
        batch_size = 64
        
        # Convert data to Tensor
        # Reshape X from (N, 2000) to (N, 5, 20, 20)
        N = X_train.shape[0]
        X_train_t = torch.tensor(X_train.reshape(N, 5, 20, 20), dtype=torch.float32).to(self.device)
        Y_policy_train_t = torch.tensor(Y_policy_train, dtype=torch.float32).to(self.device) # Targets should be class indices for CrossEntropy or Probs for KLDiv
        # Assuming Y_policy_train is one-hot or prob distribution. 
        # For simplicity let's assume we want to match distribution -> KLDivLoss or just MSE for now since simplistic.
        # Correct way involves LogSoftmax + NLLLoss (if integer target) or CrossEntropy (if integer target).
        # Since Y_policy_train is likely one-hot encoded (400 floats), we can use CrossEntropy if we get indices.
        # Let's stick to a robust simple loss: MSE on probs (suboptimal but works) or KLDiv.
        # Let's compute target index for CrossEntropy
        train_labels = torch.argmax(torch.tensor(Y_policy_train), dim=1).to(self.device)
        
        Y_value_train_t = torch.tensor(Y_value_train, dtype=torch.float32).to(self.device)
        
        print("Training with PyTorch...")
        
        for epoch in range(epochs):
            self.model.train()
            permutation = torch.randperm(N)
            
            total_loss = 0.0
            
            for i in range(0, N, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x = X_train_t[indices]
                batch_labels = train_labels[indices]
                batch_val_target = Y_value_train_t[indices]
                
                self.optimizer.zero_grad()
                
                p_probs, v_pred = self.model(batch_x)
                
                # Policy Loss: Cross Entropy
                # We need LogProbs for NLLLoss
                # p_probs are Softmaxed. log(p_probs)
                loss_policy = nn.NLLLoss()(torch.log(p_probs + 1e-10), batch_labels)
                
                # Value Loss: MSE
                loss_value = self.value_criterion(v_pred, batch_val_target)
                
                loss = loss_policy + loss_value
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(indices)
                
            avg_loss = total_loss / N
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
        self.model.save_for_cpp(saveFile)
