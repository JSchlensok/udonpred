from typing import Union

import torch


class FNN(torch.nn.Module):
    def __init__(
        self, n_features: int, output_dim: int = 1, hidden_layer_sizes: list[int] = [], dropout_rate: Union[float, None] = None
    ):
        super(FNN, self).__init__()
        layer_sizes = [n_features] + hidden_layer_sizes + [output_dim]

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(torch.nn.LeakyReLU())
            if dropout_rate:
                layers.append(torch.nn.Dropout(dropout_rate))

        # Remove last dropout & replace last ReLU with sigmoid
        if dropout_rate:
            layers = layers[:-1]
        layers[-1] = torch.nn.Sigmoid()

        # Combine all layers into a sequential model
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze()
