import torch
import torchvision
from torchvision.datasets import KMNIST # The dataset
from torch.utils import data

import numpy as np


class DatasetKMNIST(data.Dataset):
    def __init__(self, root_path, train=True):
        self.train = train
        self.data = KMNIST(root=root_path, train=self.train, download=True)
        self.mean = (0.1307,)
        self.std = (0.3081,)
        self.train_transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomPerspective(), 
                    torchvision.transforms.RandomRotation(10, fill=(0,)), 
                    torchvision.transforms.ToTensor()
                    # torchvision.transforms.Normalize(self.mean, self.std)
                ])

        self.test_transform = torchvision.transforms.Compose([ 
                    torchvision.transforms.ToTensor()
                    # torchvision.transforms.Normalize(self.mean, self.std)
                ])

    def __len__(self):
        return len(self.data)

    def add_impulse_noise(self, image_tensor, noise_ratio=0.1, noise_color=128):
        mask = torch.rand_like(image_tensor[0]) < noise_ratio
        for c in range(image_tensor.shape[0]):  # Apply for each channel
            image_tensor[c][mask] = noise_color / 255.0
        return image_tensor

    def inverse_normalize(self, tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)  # Undo the normalization
        return tensor

    def plot(self, img_tensor):
        inv_img = self.inverse_normalize(img_tensor.clone(), self.mean, self.std)
        transform_to_pil = torchvision.transforms.ToPILImage()
        noisy_image = transform_to_pil(img)
        noisy_image.save('img.jpg')


    def __getitem__(self, index):
        img, label = self.data[index]
        # print(img.size, label)
        if self.train:
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)
        img = self.add_impulse_noise(img)
        label = np.array(label)
        label = torch.from_numpy(label)
        return img, label

        
if __name__=="__main__":
    dataset = DatasetKMNIST(root_path ='./data' , train=True)
    dataset.__getitem__(0)



    
        