import numpy as np
import torch
from utils import correntropy

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader # Pytorch data loading utility
from torch.optim import Adam # Optimizer

from dataset import DatasetKMNIST
# from train import training
# from test import evaluation
from utils import save_model, correntropy
from model import StackedAutoencoder, Classifier

from datetime import datetime
from utils import inverse_normalize, plot, custom_loss

def show_prediction(x, pred, run_id, epoch, idx):
    plot(x, run_id, epoch, idx)
    plot(pred, run_id, epoch, idx, input=False)


def evaluation(testDataLoader, model, criterion, device, run_id, epoch, plot = False):
    model.eval()
    losses = []
    valCorrect = 0
    total_sample = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(testDataLoader):
            (x, y) = (x.to(device), y.to(device))
            pred, non_anchor, anchor, latent = model(x)
            x = x.squeeze(dim=1)
            # print(pred.shape, x.shape)
            loss = custom_loss(pred, x, non_anchor, anchor)
            losses.append(loss.item())
            # valCorrect += (pred.argmax(1) == y).type(
            #     torch.float).sum().item()
            total_sample += x.size(0)
            # print(f'shape::: x:{x.size()} pred:{pred.size()}')
            if plot:
                show_prediction(x, pred, run_id, epoch, idx)
                # break

        avgValLoss = np.mean(np.asarray(losses))
        accVal = valCorrect / total_sample
    return model, avgValLoss, accVal

if __name__=="__main__":
    BATCH_SIZE = 256
    EPOCHS = 60
    INIT_LR = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = DatasetKMNIST(root_path ='./data' , train=False)
    testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    criterion = nn.MSELoss()
    bottleneck = 100
    model = StackedAutoencoder(bottleneck)
    model.to(device)
    weight_path = "/home/UFAD/mdmahfuzalhasan/Documents/Projects/class_projects/fall_2023/DL/project_2/SAE_Project/saved_models/11-28-23_0426/model_50.pth"
    state_dict = torch.load(weight_path)['model']
    model.load_state_dict(state_dict, strict=True)

    evaluation(testDataLoader, model, criterion, device, plot=True)