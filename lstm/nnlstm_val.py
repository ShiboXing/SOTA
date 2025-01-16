"""
Validate if the custom LSTM produces the same values as the torch.nn.LSTM
forward and backward
"""

from mylstm import LSTM
from torch import nn
import torch.multiprocessing as mp
import torch

torch.manual_seed(42)
device = torch.device("cuda")
# device = torch.device("cpu")
in_dim = 100
hidden_dim = 200
layer_num = 4
BS = 16
SEQ_LEN = 200

model = LSTM(in_dim, hidden_dim, layer_num).to(device)
X = torch.randn(BS, SEQ_LEN, in_dim).to(device)
Y, (H, C) = model(X)

model.use_ext = False
Y_og, (H_og, C_og) = model(X)

print(Y.numel(), H.numel(), C.numel())
print((torch.abs(Y - Y_og) > 1e-4).sum())
print((torch.abs(H - H_og) > 1e-4).sum())
print((torch.abs(C - C_og) > 1e-4).sum())

# Y[3, 0, 3] = 0.03134
# for i, p in enumerate(model.parameters()):
#     print(p.shape)

torch.save(X, "X2.pt")
torch.save(Y, "Y2.pt")
torch.save(H, "H2.pt")
torch.save(C, "C2.pt")
torch.save(next(model.parameters()).grad, "p2_grad.pt")
