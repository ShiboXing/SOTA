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

model = nn.LSTM(6, 10, 2, batch_first=True).to(device)

X = torch.randn(4, 20, 6).to(device)
Y, (H, C) = model(X)
(1+Y).sum().backward()

print("nn.LSTM H.shape", H.shape, C.shape)

for i, p in enumerate(model.parameters()):
    print(p.shape)
torch.save(X, "X1.pt")
torch.save(Y, "Y1.pt")
torch.save(H, "H1.pt")
torch.save(C, "C1.pt")
torch.save(next(model.parameters()).grad, "p1_grad.pt")
