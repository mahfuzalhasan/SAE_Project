import torch
import os
import numpy as np
import torchvision
import torch.nn.functional as F

def save_model(model, opt, e, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file_path = os.path.join(save_dir, f'model_{e}.pth')
    states = {
        'epoch': e,
        'model': model.state_dict(),
        'optimizer': opt.state_dict()
    }
    torch.save(states, save_file_path)

def correntropy(truth, pred, sigma=0.60):
    pi = np.pi
    x = truth - pred
    kernel = (1 / (pi * sigma)) * torch.exp((-(x * x) / (2 * sigma * sigma)))
    return -torch.mean(kernel)

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Undo the normalization
    return tensor

def plot(img_tensor, run_id, epoch, batch, input=True):
    # mean = (0.1307,)
    # std = (0.3081,)
    path = f'./output/{run_id}/{epoch}/{batch}'
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(img_tensor.size(0)):
        # inv_img = inverse_normalize(img_tensor[i].clone(), mean, std)
        transform_to_pil = torchvision.transforms.ToPILImage()
        noisy_image = transform_to_pil(img_tensor[i])
        name = f'{i}_inp.jpg' if input else f'{i}_out.jpg'
        noisy_image.save(os.path.join(path, name))

def custom_loss(output, target, non_anchor, anchor, lamda=1):
    reconstruction_loss = lamda * F.mse_loss(output, target)
    latent_loss = torch.sum(non_anchor - anchor)
    return reconstruction_loss + latent_loss