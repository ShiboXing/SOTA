"""
Validate if the custom LSTM produces produces the same values as the torch.nn.LSTM
forward and backward
"""

from mylstm import LSTM
from torch.nn import LSTM as torch_LSTM

mylstm = LSTM(6, 10, 2)
lstm = torch_LSTM(6, 10, 2)



