import torch
x = torch.rand(6, requires_grad=True)
print (x)

y = x+2
print(y)

z=y*y*2
z=z.mean()
print(z)

z.backward()  #dy/dx
print(x.grad)