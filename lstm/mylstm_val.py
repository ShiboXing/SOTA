"""
Validate if the custom LSTM produces the same values as the torch.nn.LSTM
forward and backward
"""

from mylstm import LSTM as MyLSTM
from torch.nn import LSTM
import random
import torch.multiprocessing as mp
import torch
from time import sleep

def get_lstm_res(rank, queue):
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda:0")
    in_dim = 100
    hidden_dim = 200
    layer_num = 1
    BS = 16
    SEQ_LEN = 200
    if rank == 2:
        h_prev = torch.zeros(
            (layer_num, BS, hidden_dim), requires_grad=False, device=device
        )
        c_prev = torch.zeros(
            (layer_num, BS, hidden_dim), requires_grad=False, device=device
        )
    else:
        h_prev = torch.zeros(
            (layer_num, hidden_dim), requires_grad=False, device=device
        )
        c_prev = torch.zeros(
            (layer_num, hidden_dim), requires_grad=False, device=device
        )

    if rank == 0:
        model = MyLSTM(in_dim, hidden_dim, layer_num, use_ext=True).to(device)
    elif rank == 1:
        model = MyLSTM(in_dim, hidden_dim, layer_num, use_ext=False).to(device)
    else:
        model = LSTM(in_dim, hidden_dim, layer_num, batch_first=True).to(device)

    X = torch.randn(BS, SEQ_LEN, in_dim).to(device)
    Y, (H, C) = model(X, (h_prev, c_prev))
    queue.put((rank, Y.detach().cpu(), H.detach().cpu(), C.detach().cpu()))
    print(Y.numel(), H.numel(), C.numel())

    # model_params = enumerate(model.parameters())
    # print("param len: ", len(list(model.parameters())))
    # queue.put((rank, X.detach().clone().cpu(), \
            # next(model_params)[1].detach().clone().cpu(), \
            # next(model_params)[1].detach().clone().cpu()))


q = mp.Manager().Queue()
p0 = mp.Process(target=get_lstm_res, args=(0, q))
p1 = mp.Process(target=get_lstm_res, args=(1, q))
p2 = mp.Process(target=get_lstm_res, args=(2, q))
p0.start()
p1.start()
p2.start()

p0.join()
p1.join()
p2.join()
outputs = [None, None, None]
rank0, Y0, H0, C0 = q.get()
rank1, Y1, H1, C1 = q.get()
rank2, Y2, H2, C2 = q.get()
outputs[rank0] = (Y0, H0, C0)
outputs[rank1] = (Y1, H1, C1)
outputs[rank2] = (Y2, H2, C2)

print((torch.abs(outputs[0][0] - outputs[1][0]) > 1e-4).sum())
print((torch.abs(outputs[1][0] - outputs[2][0]) > 1e-4).sum())
print((torch.abs(outputs[0][1] - outputs[1][1]) > 1e-4).sum())
print((torch.abs(outputs[1][1] - outputs[2][1]) > 1e-4).sum())
print((torch.abs(outputs[0][2] - outputs[1][2]) > 1e-4).sum())
print((torch.abs(outputs[1][2] - outputs[2][2]) > 1e-4).sum())
# print((torch.abs(H-H_og) > 1e-4).sum())
# print((torch.abs(C-C_og) > 1e-4).sum())

# Y[3, 0, 3] = 0.03134
# for i, p in enumerate(model.parameters()):
#     print(p.shape)