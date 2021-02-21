import torch
import attack
from utils import wellImageDataset
from utils import denormalize_image
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Normalize, ToTensor
from models.resnet import resnet50
import matplotlib.pyplot as plt
from gaussian_filter import gaussianBlur
from utils import *

BATCH_SIZE=124
def run():
    torch.multiprocessing.freeze_support()
    import torchgeometry
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    # chargement du modele resnet50

    device = torch.device("cuda")
    net = resnet50().to(device)
    net.load_state_dict(torch.load("resnet50-90-regular.pth"))

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=normalize)
    cifat100_test_loader = DataLoader(cifar100_test, shuffle=True, num_workers=4, batch_size=BATCH_SIZE)

    n_samples = 0
    n_well_classified = 0
    n_well_classified_top5 = 0
    net.eval()

    well_classified_images = torch.tensor([]).to(device)
    well_classified_targets = torch.tensor([]).to(device)

    with torch.no_grad():
        for i, batch in enumerate(cifat100_test_loader):
            if i > 100:
                break;
            image, target = batch
            image = image.to(device)
            target = target.to(device)
            n_samples += image.shape[0]

            prob = net(image).argmax(1)

            correct = (prob == target)
            n_well_classified += correct.sum()

            well_classified_images = torch.cat((well_classified_images, image[correct]), 0)
            well_classified_targets = torch.cat((well_classified_targets, target[correct]), 0)

    well_DS = wellImageDataset(well_classified_images.cpu(), well_classified_targets.cpu())
    well_DL = DataLoader(well_DS, shuffle=False, num_workers=1, batch_size=10)

    misclassified_images = torch.tensor([]).to(device)
    misclassified_images_perb = torch.tensor([]).to(device)
    misclassified_targets = torch.tensor([]).to(device)
    n_misc = 0
    for i, (batch_image, batch_target) in enumerate(well_DL):
        batch_image = batch_image.cuda()
        batch_target = batch_target.cuda()
        per = attack.smia_attack(batch_image, 0.005, 0, net, batch_target.long(), 10)
        prob = net(per).argmax(1)
        correct = prob == batch_target
        k_well_class = (prob == batch_target).sum()
        n_misc += k_well_class

        misclassified_images = torch.cat((misclassified_images, batch_image[~correct]), 0)
        misclassified_images_perb = torch.cat((misclassified_images_perb, per[~correct]), 0)
        misclassified_targets = torch.cat((misclassified_targets, batch_target[~correct]), 0)
        if i > 10:
            break;
        print(i)
        print(k_well_class)
    print(n_misc / len(well_classified_images))

    example = denormalize_image(misclassified_images[1].unsqueeze(0))
    example_perb = denormalize_image(misclassified_images_perb[1].unsqueeze(0))
    example = example.detach().cpu().squeeze()
    example_perb = example_perb.detach().cpu().squeeze()
    plt.imshow(example)
    plt.show()
    plt.imshow(example_perb)
    plt.show()
if __name__ == '__main__':
    run()



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
