import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader # Pytorch data loading utility
from dataset import DatasetKMNIST
from train import training
from test import evaluation
from utils import save_model



def Main():
    BATCH_SIZE = 64
    EPOCHS = 30
    INIT_LR = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = DatasetKMNIST(root_path ='./data' , train=True)
    test_dataset = DatasetKMNIST(root_path ='./data' , train=False)
    trainDataLoader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    bottleneck = 100
    num_classes = 10  # Define the number of classes
    model = Classifier(bottleneck, num_classes)  # 100 is the size of the bottleneck features
    model.to(device)
    lossFn = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=INIT_LR)

    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_acc = 0
    incident = 0

    for epoch in range(EPOCHS):
        print(f'####### epoch:{epoch}#########\n')
        model, loss_train, acc_train = training(trainDataLoader, model, lossFn, optimizer, device)
        score, loss_val, acc_val = evaluation(testDataLoader, model, device)
        print(f'loss_T:{loss_train} acc_T:{acc_train} loss_v:{loss_val} acc_V:{acc_val}')
        # update our training history
        H["train_loss"].append(loss_train)
        H["train_acc"].append(acc_train)
        H["val_loss"].append(loss_val)
        H["val_acc"].append(acc_val)

        if best_acc>= acc_val:
            incident += 1
            print(f"No improvement for {incident} epoch(s)")
            if incident == 10:
                print('model is not improving. Training is stopped')
                last_epoch = epoch
                break
        else:
            best_accuracy = acc_val
            print("Improvement of Val accuracy: {:.4f}\n".format(valCorrect))
            save_model(model, opt, epoch, save_dir='./saved_models')
            incident = 0





if __name__=="__main__":
    Main()
