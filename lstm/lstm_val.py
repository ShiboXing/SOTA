"""
Validate if the custom LSTM produces produces the same values as the torch.nn.LSTM
forward and backward
"""

from mylstm import LSTM
from torch import nn
import torch

torch.seed(

mylstm = LSTM(6, 10, 2).cuda()
lstm = nn.LSTM(6, 10, 2).cuda()

X = torch.randn(4, 20, 6).cuda()
yhat1, _ = mylstm(X)
yhat2, _ = lstm(X)

print(yhat1.shape)
print(yhat2.shape)