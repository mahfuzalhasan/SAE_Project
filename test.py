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


def evaluation(testDataLoader, model, criterion, device, writer, epoch, run_id, plot = False, lamda=0.001):
    model.eval()
    losses = []
    re_losses = []
    latent_losses = []
    valCorrect = 0
    total_sample = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(testDataLoader):
            (x, y) = (x.to(device), y.to(device))
            pred, non_anchor, anchor, latent = model(x)
            x = x.squeeze(dim=1)
            # print(pred.shape, x.shape)
            loss, re_loss, lat_loss = custom_loss(pred, x, non_anchor, anchor, lamda=lamda)
            losses.append(loss.item())
            re_losses.append(re_loss.item())
            latent_losses.append(lat_loss.item())
            # valCorrect += (pred.argmax(1) == y).type(
            #     torch.float).sum().item()
            total_sample += x.size(0)
            # print(f'shape::: x:{x.size()} pred:{pred.size()}')
            if plot:
                show_prediction(x, pred, run_id, epoch, idx)
                # break

        avgValLoss = np.mean(np.asarray(losses))
        accVal = valCorrect / total_sample

        avg_re_loss = np.mean(np.asarray(re_losses))
        avg_latent_loss = np.mean(np.asarray(latent_losses))

        print(f'loss_v:{avgValLoss} avg_re_loss_v:{avg_re_loss}  avg_latent_loss_v:{avg_latent_loss} ')

        writer.add_scalar('val_loss', avgValLoss, epoch)
        writer.add_scalar('val_re_loss', avg_re_loss, epoch)
        writer.add_scalar('val_latent_loss', avg_latent_loss, epoch)

    return avgValLoss, accVal

def evaluation_classifier(testDataLoader, model, criterion, device, writer, epoch):
    model.eval()
    losses = []
    valCorrect = 0
    total_sample = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(testDataLoader):
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)

            loss = criterion(pred, y)
            losses.append(loss.item())

            valCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()
            total_sample += x.size(0)
            

        avgValLoss = np.mean(np.asarray(losses))
        accVal = valCorrect / total_sample
        print(f'loss_v:{avgValLoss} acc_v:{accVal}')

        writer.add_scalar('val_loss_cls', avgValLoss, epoch)
        writer.add_scalar('val_acc', accVal, epoch)

    return avgValLoss, accVal

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