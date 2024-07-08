from torch import nn
import torch

X1 = torch.load("X1.pt")
X2 = torch.load("X2.pt")
Y1 = torch.load("Y1.pt")
Y2 = torch.load("Y2.pt")
C1 = torch.load("C1.pt")
C2 = torch.load("C2.pt")
H1 = torch.load("H1.pt")
H2 = torch.load("H2.pt")
PG1 = torch.load("p1_grad.pt")
PG2 = torch.load("p2_grad.pt")

torch.set_printoptions(precision=13)
print("Y2,Y2", Y1.shape, Y2.shape, Y1[:, 5, 3:5], Y2[:, 5, 3:5])
print(H1.shape, H2.shape, C1.shape, C2.shape)
print("H1,H2", H1[0, 0, :3], H2[0, :3])
# print(C1[0, -2, 4:6], C2[0, -2, 4:6])

diff = Y1 != Y2
# print(torch.nonzero(diff), torch.nonzero(diff).shape)
assert torch.equal(X1, X2)
# print(torch.abs(Y1-Y2))
torch.set_printoptions(threshold=10_000)

print("PG1,PG2", torch.nonzero(torch.abs(PG1-PG2) >= 1e-5), PG1[20:22, :2], PG2[20:22, :2])
PG1 = torch.nan_to_num(PG1)
PG2 = torch.nan_to_num(PG2)

assert torch.all(torch.abs(Y1-Y2) < 1e-5)
assert torch.all(torch.abs(C1-C2) < 1e-5)
assert torch.all(torch.abs(H1-H2) < 1e-5)
assert torch.all(torch.abs(PG1-PG2) < 1e-5)
