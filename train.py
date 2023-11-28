import numpy as np
import torch

def training(trainDataLoader, model, criterion, opt, device):
    model.train()

    losses = []
    trainCorrect = 0
    total_sample = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

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

    return model, avgTrainLoss, accTrain
