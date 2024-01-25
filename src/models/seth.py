import torch
from torch import nn
from torch.nn import Module


class SETHClone(Module):
    def __init__(self, n_features: int, bottleneck_dim: int, kernel_size: int = 5):
        super().__init__()

        kernel_dim = (kernel_size, 1)
        padding = (kernel_size // 2, 0)

        self.model = nn.Sequential(
            nn.Conv2d(
                n_features, bottleneck_dim, kernel_size=kernel_dim, padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(bottleneck_dim, 1, kernel_size=kernel_dim, padding=padding),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(2, 0, 1)
        return self.sigmoid(self.model(x)).squeeze(0)

