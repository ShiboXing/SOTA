"""
Validate if the custom LSTM produces the same values as the torch.nn.LSTM
forward and backward
"""

from mylstm import LSTM
from torch import nn
import torch.multiprocessing as mp
import torch

model = LSTM(6, 10, 1).to("cuda")
X = torch.randn(4, 20, 6).to("cuda")
Y, (H, C) = model(X)