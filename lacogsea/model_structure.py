import torch
import torch.nn as nn

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        all_dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:
                layers.append(nn.BatchNorm1d(all_dims[i + 1]))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.apply(initialize_weights)

    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dims, input_dim):
        super().__init__()
        all_dims = [output_dim] + hidden_dims[::-1] + [input_dim]
        layers = []
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:
                layers.append(nn.BatchNorm1d(all_dims[i + 1]))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.apply(initialize_weights)

    def forward(self, x):
        return self.layers(x)
