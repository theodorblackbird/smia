import torch
import attack
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Normalize, ToTensor
from models.resnet import resnet50
import matplotlib.pyplot as plt
from gaussian_filter import gaussianBlur
from utils import *
import torchgeometry
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
#chargement du modele resnet50

device = torch.device("cuda") 
net = resnet50().to(device)
net.load_state_dict(torch.load("resnet50-90-regular.pth"))


normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=normalize)
cifat100_test_loader = DataLoader(cifar100_test, shuffle=True, num_workers=4, batch_size=16)



n_samples = 0
n_well_classified = 0
n_well_classified_top5 = 0
net.eval()
with torch.no_grad():
    for i, batch in enumerate(cifat100_test_loader):
        image, target = batch
        image = image.to(device)
        target = target.to(device)
        n_samples += image.shape[0]
        _, prob = net(image).topk(5,1, largest=True, sorted=True)

        target = target.view(target.size(0), -1).expand_as(prob)
        correct = prob.eq(target).float()

        n_well_classified_top5 += correct[:,:5].sum()
        n_well_classified += correct[:,:1].sum()
        print(i)
print(n_well_classified/n_samples)
print(n_well_classified_top5/n_samples)



sample = next(iter(cifat100_test_loader))
image = sample[0][0].unsqueeze(0).to(device)
target = sample[1][0].unsqueeze(0).to(device)

per = attack.smia_attack(image, 0.05, 0, net, target, 10).squeeze()

image = denormalize_image(image.squeeze())
plt.imshow(image.cpu())
plt.show()
per = denormalize_image(per.squeeze())
per = per.cpu().detach()
plt.imshow(per)
plt.show()

"""
plt.figure(figsize=(10,1))
for i in range(10):
    x = sample[0][i].permute(1,2,0)
    x = (x*torch.tensor(std)) + torch.tensor(mean)
    plt.subplot(1, 10, i+1)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(x)
plt.show()
"""
