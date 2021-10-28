import torch
import torch.nn as nn
from typing import Any


__all__ = ['AlexNet1d', 'alexnet1d']


class AlexNet1d(nn.Module):

    def __init__(self, in_width, in_channels, num_classes: int = 1000) -> None:
        super(AlexNet1d, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 96, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(96),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv1d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv1d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(384),
            
            nn.Conv1d(384, 384, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(384),
            
            nn.Conv1d(384, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        in_width = ((in_width-1)//4)+1 # conv1
        in_width = ((in_width-1)//2)+1 # maxpool1

        in_width = in_width # conv2
        in_width = ((in_width-1)//2)+1 # maxpool2

        in_width = in_width # conv3, conv4, conv5
        in_width = ((in_width-1)//2)+1 # maxpool3

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_width*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet1d(**kwargs: Any) -> AlexNet1d:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet1d(**kwargs)
    return model