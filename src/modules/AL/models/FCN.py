import torch
import torch.nn as nn
from typing import Any


__all__ = ['FCN', 'fcn']


class FCN(nn.Module):

    def __init__(self, in_width, in_channels, num_classes: int = 1000) -> None:
        super(FCN, self).__init__()

        in_width = ((in_width-1)//4)+1 # conv1

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 96, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(96),
            
            nn.Conv1d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            
            nn.Conv1d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(384),
            
            nn.Conv1d(384, 384, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(384),
            
            nn.Conv1d(384, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),

            nn.Conv1d(256, num_classes, kernel_size=1, stride=1, padding=0),            
            nn.MaxPool1d(kernel_size=in_width, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def fcn(**kwargs: Any) -> FCN:
    model = FCN(**kwargs)
    return model