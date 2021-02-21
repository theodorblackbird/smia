import torch

device = torch.device("cuda")

mean_t = torch.tensor((0.5070751592371323, 0.48654887331495095, 0.4409178433670343)).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
std_t = torch.tensor((0.2673342858792401, 0.2564384629170883, 0.27615047132568404)).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
def denormalize_image(image):
    image = image.permute(0,2,3,1)
    return image*mean_t + std_t

class wellImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
    def __len__(self):
        return len(self.data)
    def __getitem__(self, ix):
        return self.data[ix], self.target[ix]
