"""
Validate if the custom LSTM produces produces the same values as the torch.nn.LSTM
forward and backward
"""

from mylstm import LSTM
from torch import nn
import torch.multiprocessing as mp
import torch


def run_module(is_pt: bool, q_param: mp.Queue, q_res: mp.Queue):
    torch.manual_seed(42)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    if is_pt:
        print("lstm created")
        model = nn.LSTM(6, 10, 1, batch_first=True).to(device)
    else:
        print("custom lstm created")
        model = LSTM(6, 10, 1).to(device)

    X = torch.randn(4, 20, 6).to(device)
    Y, (H, C) = model(X)

    q_param.put((is_pt, [p.detach().cpu() for p in model.parameters()]))
    q_res.put((is_pt, (Y.detach().cpu(), H.detach().cpu(), C.detach().cpu())))


man = mp.Manager()
q = man.Queue()
q_res = man.Queue()
pt_proc = mp.Process(target=run_module, args=(True, q, q_res))
custom_proc = mp.Process(target=run_module, args=(False, q, q_res))

pt_proc.start()
custom_proc.start()

pt_proc.join()
custom_proc.join()

out1 = q.get()
out2 = q.get()

assert out1[0] != out2[0]
assert len(out1[1]) == len(out2[1])

print(out1[1][2][3:6], out2[1][2][3:6])

for i in range(len(out1[1])):
    assert torch.equal(out1[1][i], out2[1][i])

is_pt1, (Y1, H1, C1) = q_res.get()
is_pt2, (Y2, H2, C2) = q_res.get()

assert is_pt1 != is_pt2
print(Y1.shape, Y2.shape, H1.shape, H2.shape, C1.shape, C2.shape)

diff = Y1 != Y2
print(
    torch.nonzero(diff, as_tuple=False),
    len(torch.nonzero(diff, as_tuple=False)),
    Y1.shape[0] * Y1.shape[1] * Y1.shape[2],
)
print(Y1.dtype, Y2.dtype)
torch.set_printoptions(precision=10)
print(Y1[0, 0, :10])
print(Y2[0, 0, :10])
assert torch.equal(Y1, Y2)
# assert torch.equal(H1, H2)
# assert torch.equal(C1, C2)
