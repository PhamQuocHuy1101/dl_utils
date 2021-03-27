import torch

import main as m

s = torch.tensor([1, 2])
k = torch.tensor([3, 4])
print(m.find_parameters(28, 1, 4, layer='conv'))

