from torchgeometry import image
import torch
from torch.nn import functional as F
from gaussian_filter import gaussianBlur

def smia_attack(image, epsilon, alpha, model, target, n_step):
    eta = torch.zeros_like(image)
    W = gaussianBlur(10, 3 , 1).cuda()
    for i in range(n_step):
        image = (image + eta).requires_grad_()
        image.retain_grad()
        model.zero_grad()
        model(image)
        L_dev = F.nll_loss(model(image), target)
        L_sta = 0
        if i != 0 :
            L_sta = F.nll_loss(model(image), model(image + W(eta) - eta).argmax().unsqueeze(0))
        L_smia = L_dev - alpha * L_sta
        L_smia.backward()
        eta = epsilon * image.grad.data.sign()
    return (image + eta)

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

