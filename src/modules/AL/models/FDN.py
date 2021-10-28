import torch
import torch.nn as nn
from typing import Any


__all__ = ['FDN', 'fdn']


class FDN(nn.Module):

    def __init__(self, in_width, in_channels, num_classes: int = 1000) -> None:
        super(FDN, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_width*in_channels, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def fdn(**kwargs: Any) -> FDN:
    model = FDN(**kwargs)
    return model