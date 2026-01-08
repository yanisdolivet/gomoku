##
## EPITECH PROJECT, 2025
## chessboard-analyser
## File description:
## model_specification
##


class ModelSpecifications:
    def __init__(self):
        # Model architecture
        self.type = []
        self.layer_sizes = []
        self.activation = []
        self.num_layers = 0

        # Hyperparameters
        self.learning_rate = None
        self.initialization = None

        # Training parameters
        self.batch_size = None
        self.lreg = None
        self.dropout_rate = None
        self.epochs = None
        self.loss_function = None
