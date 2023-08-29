import torch

# Create a sparse tensor
indices = torch.tensor([[0, 1, 1],
                        [2, 0, 2]])
values = torch.tensor([3, 4, 5], dtype=torch.float32)
sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(3, 3))

# Convert the sparse tensor to dense
dense_tensor = sparse_tensor.to_dense()

# Perform an operation on the sparse tensor
sparse_result = sparse_tensor * 2

print("Sparse Tensor:")
print(sparse_tensor)
print("Dense Tensor:")
print(dense_tensor)
print("Sparse Result:")
print(sparse_result)
