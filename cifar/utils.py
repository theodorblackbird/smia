import torch

device = torch.device("cuda")

mean_t = torch.tensor((0.5070751592371323, 0.48654887331495095, 0.4409178433670343)).to(device)
std_t = torch.tensor((0.2673342858792401, 0.2564384629170883, 0.27615047132568404)).to(device)
def denormalize_image(image):
    image = image.permute(1,2,0)
    return image*mean_t + std_t
