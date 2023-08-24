import torch

scalar = torch.tensor(7)
vector = torch.tensor([9, 0])
matrix = torch.tensor([[9, 5], [3, 2]])
tensor = torch.tensor([[[7, 0], [5, 4]],[[6, 9], [3, 1]]])
randtensor = torch.rand(4, 2, 3)
print(randtensor)