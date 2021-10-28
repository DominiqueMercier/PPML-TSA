import torch
import torch.nn as nn
from typing import Any


__all__ = ['LSTM', 'lstm']


class LSTM(nn.Module):

    def __init__(self, in_width, in_channels, num_classes: int = 1000) -> None:
        super(LSTM, self).__init__()
        
        self.hidden_size = 512

        self.lstm_stack = nn.LSTM(input_size=in_channels, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(in_width*(self.hidden_size*2), num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lstm_stack(x)
        x = torch.flatten(x[0], 1)
        x = self.linear(x)
        return x


def lstm(**kwargs: Any) -> LSTM:
    model = LSTM(**kwargs)
    return model