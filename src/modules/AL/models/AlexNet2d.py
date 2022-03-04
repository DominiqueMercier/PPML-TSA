import torch
import torch.nn as nn
from typing import Any


__all__ = ['AlexNet2d', 'alexnet2d']


class AlexNet2d(nn.Module):

    def __init__(self, in_width, in_channels, num_classes: int = 1000) -> None:
        super(AlexNet2d, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            
            nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            
            nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
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


def alexnet2d(**kwargs: Any) -> AlexNet2d:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet2d(**kwargs)
    return model
