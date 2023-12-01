import numpy as np
import torch
from utils import correntropy, custom_loss

def training(trainDataLoader, model, criterion, opt, device, writer, epoch, lamda = 0.001):
    model.train()

    losses = []
    re_losses = []
    latent_losses = []
    trainCorrect = 0
    total_sample = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    for (x, y) in trainDataLoader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        pred, non_anchor, anchor, latent = model(x)
        
        x = x.squeeze(dim=1)
        loss, re_loss, lat_loss = custom_loss(pred, x, non_anchor, anchor, lamda=lamda)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        re_losses.append(re_loss.item())
        latent_losses.append(lat_loss.item())
        # trainCorrect += (pred.argmax(1) == y).type(
        #     torch.float).sum().item()
        total_sample += x.size(0)

    avgTrainLoss = np.mean(np.asarray(losses))
    accTrain = trainCorrect / total_sample

    avg_re_loss = np.mean(np.asarray(re_losses))
    avg_latent_loss = np.mean(np.asarray(latent_losses))
    print(f'loss_t:{avgTrainLoss} avg_re_loss_t:{avg_re_loss}  avg_latent_loss_t:{avg_latent_loss} ')
    writer.add_scalar('train_loss', avgTrainLoss, epoch)
    writer.add_scalar('train_re_loss', avg_re_loss, epoch)
    writer.add_scalar('train_latent_loss', avg_latent_loss, epoch)

    return model, avgTrainLoss, accTrain

def training_classifier(trainDataLoader, model, criterion, opt, device, writer, epoch):
    model.train()
    losses = []
    trainCorrect = 0
    total_sample = 0

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    for (x, y) in trainDataLoader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        pred = model(x)

        loss = criterion(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        trainCorrect += (pred.argmax(1) == y).type(
            torch.float).sum().item()
        total_sample += x.size(0)

    avgTrainLoss = np.mean(np.asarray(losses))
    accTrain = trainCorrect / total_sample

    print(f'loss_t:{avgTrainLoss} acc_t:{accTrain}')
    writer.add_scalar('train_loss_cls', avgTrainLoss, epoch)
    writer.add_scalar('train_acc', accTrain, epoch)

    return model, avgTrainLoss, accTrain
