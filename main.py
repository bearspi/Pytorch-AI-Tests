import torch
import time

y = torch.rand(size=(4, 3, 2), device="mps", requires_grad=False)
x = torch.rand(size=(3, 2, 3), device="mps", requires_grad=False)

print(torch.matmul(x, y))
