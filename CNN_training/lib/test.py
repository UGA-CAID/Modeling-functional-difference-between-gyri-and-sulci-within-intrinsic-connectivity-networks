import torch


data = torch.randn(5, 2)
value, idxs = torch.max(data, dim=1)

print(data)
print(value)
print(idxs)

