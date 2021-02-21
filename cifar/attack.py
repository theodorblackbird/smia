from torchgeometry import image
import torch
from torch.nn import functional as F
from gaussian_filter import gaussianBlur


def smia_attack(image, epsilon, alpha, model, target, n_step):
    eta = torch.zeros_like(image)
    W = gaussianBlur(3 , 1).cuda()
    for i in range(n_step):
        image = (image + eta).requires_grad_()
        image.retain_grad()
        model.zero_grad()
        model(image)
        L_dev = F.nll_loss(model(image), target)
        L_sta = 0
        if i != 0 :
            L_sta = F.nll_loss(model(image), model(image + W(eta) - eta).argmax(1))
        L_smia = L_dev - alpha * L_sta
        L_smia.backward()
        eta = epsilon * image.grad.data.sign()
    return (image + eta)

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Projectional gradient descent

def projected_gradient_descent(model, x, y, num_steps, step_size, step_norm, eps, eps_norm,
                               clamp=(0, 1), y_target=None):
    """Performs the projected gradient descent attack on a batch of images."""
    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    targeted = y_target is not None
    num_channels = x.shape[1]

    for i in range(num_steps):
        _x_adv = x_adv.clone().detach().requires_grad_(True)

        prediction = model(_x_adv)
        loss = F.nll_loss(prediction, y_target if targeted else y)
        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            if step_norm == 'inf':
                gradients = _x_adv.grad.sign() * step_size
            else:
                # Note .view() assumes batched image data as 4D tensor
                gradients = _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1) \
                    .norm(step_norm, dim=-1) \
                    .view(-1, num_channels, 1, 1)

            if targeted:
                # Targeted: Gradient descent with on the loss of the (incorrect) target label
                # w.r.t. the image data
                x_adv -= gradients
            else:
                # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                # the model parameters
                x_adv += gradients

        # Project back into l_norm ball and correct range
        if eps_norm == 'inf':
            # Workaround as PyTorch doesn't have elementwise clip
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        else:
            delta = x_adv - x

            # Assume x and x_adv are batched tensors where the first dimension is
            # a batch dimension
            mask = delta.view(delta.shape[0], -1).norm(torch.norm, dim=1) <= eps

            scaling_factor = delta.view(delta.shape[0], -1).norm(torch.norm, dim=1)
            scaling_factor[mask] = eps

            # .view() assumes batched images as a 4D Tensor
            delta *= eps / scaling_factor.view(-1, 1, 1, 1)

            x_adv = x + delta

        #x_adv = x_adv.clamp(*clamp)

    return x_adv.detach()
