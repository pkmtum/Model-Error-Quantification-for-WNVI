# Import the necessary modules
import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def build_ffn(
    input_dim,
    output_dim,
    activation_func_name,
    hidden_layers,
    flagLayerNormalization=False,
):
    activation_func = getattr(nn, activation_func_name)()
    layers = []
    prev_dim = input_dim

    # Add hidden layers
    for num_neurons in hidden_layers:
        layers.append(nn.Linear(prev_dim, num_neurons))
        if flagLayerNormalization:
            layers.append(nn.LayerNorm(num_neurons))
        layers.append(activation_func)
        prev_dim = num_neurons

    # final, linear output layer
    layers.append(nn.Linear(prev_dim, output_dim))

    # Sequential model to hold the layers
    sequential_model = nn.Sequential(*layers)

    # Initialize weights
    sequential_model.apply(init_weights)
    return sequential_model


# # Define the input and output dimensions
# input_dim = 10
# output_dim = 5

# # Define the activation function name and hidden layers
# activation_func_name = "ReLU"
# hidden_layers = [20, 15, 10]

# # Build the feedforward neural network
# ffn = build_ffn(input_dim, output_dim, activation_func_name, hidden_layers)

# # Print the network architecture
# print(ffn)
