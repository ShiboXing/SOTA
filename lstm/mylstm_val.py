"""
Validate if the custom LSTM produces produces the same values as the torch.nn.LSTM
forward and backward
"""

from mylstm import LSTM
from torch import nn
import torch.multiprocessing as mp
import torch

torch.manual_seed(42)
# device = torch.device("cuda")
device = torch.device("cpu")

model = LSTM(6, 10, 1).to(device)

X = torch.randn(4, 20, 6).to(device)
Y, (H, C) = model(X)
(1+Y).sum().backward()

# Y[3, 0, 3] = 0.03134
for i, p in enumerate(model.parameters()):
    print(p.shape)

torch.save(X, "X2.pt")
torch.save(Y, "Y2.pt")
torch.save(H, "H2.pt")
torch.save(C, "C2.pt")
torch.save(next(model.parameters()).grad, "p2_grad.pt")
