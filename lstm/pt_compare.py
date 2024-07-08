from torch import nn
import torch


Y1 = torch.load("Y1.pt").cpu()
Y2 = torch.load("Y2.pt").cpu()
C1 = torch.load("C1.pt")
C2 = torch.load("C2.pt")
H1 = torch.load("H1.pt")
H2 = torch.load("H2.pt")

torch.set_printoptions(precision=13)
print(Y1.shape, Y2.shape, Y1[0, 0, :3], Y2[0, 0, :3])
# print(H1.shape, H2.shape, C1.shape, C2.shape)
# print(H1[0, -2, -2:], H2[0, -2, -2:])
# print(C1[0, -2, 4:6], C2[0, -2, 4:6])

diff = Y1 != Y2
# print(torch.nonzero(diff), torch.nonzero(diff).shape)
assert torch.equal(Y1, Y2)
assert torch.equal(C1, C2)
assert torch.equal(H1, H2)

