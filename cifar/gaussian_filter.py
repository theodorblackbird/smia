import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

def gaussian_blur_mat(n, std):
    m = torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            m[i][j] = torch.exp(torch.tensor(-(i*i + j*j)/(2*std*std)))/(torch.sqrt(torch.tensor(2*3.14156))*std)
    return m




class gaussianBlur(nn.Module):
    def __init__(self, K, std):
        super().__init__()
        self.gauss_mat = gaussian_blur_mat(K, std)
        self.gauss_mat
        P = int(np.floor((K-1)/2))
        self.conv = nn.Conv2d(3, 3, K, 1, P, bias=False)
        self.conv.weight = nn.Parameter(self.gauss_mat.unsqueeze(0).unsqueeze(0).repeat(3,3, 1,1))

    def forward(self,x):
        return self.conv(x)
"""
mnist_data = torchvision.datasets.MNIST('data/mnist', download=True, transform=transforms.Compose([transforms.ToTensor(),]))
data_loader = torch.utils.data.DataLoader(mnist_data,
                                          batch_size=1,
                                          shuffle=True,
                                         )
x, y = next(iter(data_loader))

g_b = gaussianBlur(28,7, 1000000)
x = g_b(x).detach()
plt.imshow(x.squeeze(), cmap='gray')
plt.show()
"""
