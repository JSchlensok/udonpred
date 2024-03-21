import torch
from torch import nn
from torch.nn import Module


class CNN(Module):
    def __init__(self, n_features: int, hidden_layer: int, kernel_size: int = 5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                n_features,
                hidden_layer,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.Tanh(),
            nn.Conv2d(
                hidden_layer,
                1,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(2, 0, 1)
        return self.sigmoid(self.model(x))
        #return self.model(x)

    # self.model(x).squeeze(0)  # self.sigmoid(self.model(x)).squeeze(0)