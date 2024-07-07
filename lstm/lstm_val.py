"""
Validate if the custom LSTM produces produces the same values as the torch.nn.LSTM
forward and backward
"""

from mylstm import LSTM
from torch import nn
import torch

torch.manual_seed(42)

lstm = nn.LSTM(6, 10, 2, batch_first=True).cuda()
mylstm = LSTM(6, 10, 2).cuda()

# X = torch.randn(4, 20, 6).cuda()
# yhat1, _ = mylstm(X)
# yhat2, _ = lstm(X)

for i, param in enumerate(lstm.parameters()):
    print(f"param{i}.shape: {param[10:13]}")
    # break
    
for i, param in enumerate(mylstm.parameters()):
    print(f"param{i}.shape: {param[10:13]}")
    # break