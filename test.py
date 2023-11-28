import numpy as np
import torch

def evaluation(testDataLoader, model, criterion, device):
    model.eval()
    losses = []
    valCorrect = 0
    total_sample = 0
    with torch.no_grad():
        for (x, y) in testDataLoader:
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            loss = criterion(pred, y)
            losses.append(loss.item())
            valCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()
            total_sample += x.size(0)

        avgValLoss = np.mean(np.asarray(losses))
        accVal = valCorrect / total_sample
    return model, avgValLoss, accVal