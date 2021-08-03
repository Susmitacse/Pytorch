import torch
import numpy as np


# x=torch.tensor([2.4,3.34])
# x=torch.rand(5,5)
# y=torch.rand(2,2)
# y.add_(x)
# y.div_(x)
# print(x)
# print(x)
# print(x.view(25))
# print(x+y)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x+y
    z = z.to("cpu")
    print(z)

