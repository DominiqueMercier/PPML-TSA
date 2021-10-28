import torch
import torch.nn as nn
from typing import Any


__all__ = ['LeNet', 'lenet']


class LeNet(nn.Module):

    def __init__(self, in_width, in_channels, num_classes: int = 1000) -> None:
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                                out_channels=6,
                                kernel_size=5, 
                                stride=1, 
                                padding=0)
        self.conv2 = nn.Conv1d(in_channels=6,
                                out_channels=16,
                                kernel_size=5, 
                                stride=1, 
                                padding=0)
        self.conv3 = nn.Conv1d(in_channels=16,
                                out_channels=120,
                                kernel_size=5,
                                stride=1, 
                                padding=0)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AvgPool1d(kernel_size = 2, stride = 2)

        # Adaptive calculation of linear layer size
        in_width = ((in_width-5)//1)+1 # conv1
        in_width = ((in_width-2)//2)+1 # avgpool1
        in_width = ((in_width-5)//1)+1 # conv2
        in_width = ((in_width-2)//2)+1 # avgpool2
        in_width = ((in_width-5)//1)+1 # conv3

        self.linear1 = nn.Linear(in_width*120, 84)
        self.linear2 = nn.Linear(84, num_classes)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.tanh(x)

        x = self.sigmoid(x)
        x = self.avgpool(x)
        
        x = self.conv2(x)
        x = self.tanh(x)
        
        x = self.sigmoid(x)
        x = self.avgpool(x)
        
        x = self.conv3(x)
        x = self.tanh(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.tanh(x)

        x = self.linear2(x)
        return x


def lenet(**kwargs: Any) -> LeNet:
    lenet = AlexNet1d(**kwargs)
    return model