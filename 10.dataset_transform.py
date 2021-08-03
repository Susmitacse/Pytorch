import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# dataiter= iter(dataloader)
# data=dataiter.next()
# features, labels=data
# print(features,labels)
#training
num_epochs=2
total_samples=len(dataset)
n_iteration = math.ceil(total_samples/4)
print(total_samples, n_iteration)

for epoch in range(num_epochs):
    for i, (input, labels) in enumerate(dataloader):
        if (i+1)%5 ==0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iteration}, input {input.shape}')


