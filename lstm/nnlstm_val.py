"""
Validate if the custom LSTM produces the same values as the torch.nn.LSTM
forward and backward
"""

from mylstm import LSTM
from torch import nn
import torch.multiprocessing as mp
import torch
from time import time

torch.manual_seed(42)
device = torch.device("cuda")
# device = torch.device("cpu")
in_dim = 200
hidden_dim = 200
layer_num = 4
BS = 16
SEQ_LEN = 200

# model = nn.LSTM(in_dim, hidden_dim, layer_num).to(device)
# model = LSTM(in_dim, hidden_dim, layer_num, use_ext=False).to(device)
model = LSTM(in_dim, hidden_dim, layer_num, use_ext=True).to(device)
Y = torch.randn(BS, SEQ_LEN, in_dim).to(device)

start_t = time()
for _ in range(100):
    Y, (H, C) = model(Y)
end_t = time()

print(f"latency: {end_t - start_t}")

model.use_ext = False
Y_og, (H_og, C_og) = model(Y)

print(Y.numel(), H.numel(), C.numel())
print((torch.abs(Y - Y_og) > 1e-4).sum())
print((torch.abs(H - H_og) > 1e-4).sum())
print((torch.abs(C - C_og) > 1e-4).sum())

# Y[3, 0, 3] = 0.03134
# for i, p in enumerate(model.parameters()):
#     print(p.shape)

# torch.save(X, "X2.pt")
# torch.save(Y, "Y2.pt")
# torch.save(H, "H2.pt")
# torch.save(C, "C2.pt")
# torch.save(next(model.parameters()).grad, "p2_grad.pt")
