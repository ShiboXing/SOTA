"""
Validate if the custom LSTM produces the same values as the torch.nn.LSTM
forward and backward
"""

from mylstm import LSTM
from torch import nn
import torch

in_dim = 6
hidden_dim = 10
layer_num = 2
device = "cuda"

model = LSTM(in_dim, hidden_dim, layer_num).to(device)
X = torch.randn(4, 20, in_dim).to(device)
Y, (H, C) = model(X)
print(Y.shape, H.shape, C.shape)

model = nn.LSTM(in_dim, hidden_dim, layer_num, batch_first=True).to(device)
Y, (H, C) = model(X)
print(Y.shape, H.shape, C.shape)