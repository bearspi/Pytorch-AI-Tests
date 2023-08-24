import torch
import time

y = torch.rand(size=(1,1,2), device="mps", requires_grad=False)
print(y / 10)
