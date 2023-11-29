import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader # Pytorch data loading utility
from torch.optim import Adam # Optimizer

from dataset import DatasetKMNIST
from train import training
from test import evaluation
from utils import save_model, correntropy
from model import StackedAutoencoder, Classifier

from datetime import datetime
from tensorboardX import SummaryWriter
import os



def Main(run_id):
    BATCH_SIZE = 256
    EPOCHS = 60
    INIT_LR = 1e-3
    alpha = 0.5
    k = 11

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = './logs'
    save_log = os.path.join(log_dir, str(run_id))
    if not os.path.exists(save_log):
        os.makedirs(save_log)
    writer = SummaryWriter(save_log)
    

    train_dataset = DatasetKMNIST(root_path ='./data' , train=True)
    test_dataset = DatasetKMNIST(root_path ='./data' , train=False)
    trainDataLoader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print(f'train ds: {len(train_dataset)} train step: {len(trainDataLoader)}')
    print(f'test ds: {len(test_dataset)} test step: {len(testDataLoader)}')

    bottleneck = 100
    num_classes = 10  # Define the number of classes
    # model = Classifier(bottleneck, num_classes)  # 100 is the size of the bottleneck features
    model = StackedAutoencoder(bottleneck, BATCH_SIZE, alpha, k)
    model.to(device)
    lossFn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=INIT_LR)

    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_acc = 0
    incident = 0
    plot = False
    for epoch in range(EPOCHS):
        print(f'####### epoch:{epoch}#########\n')
        model, loss_train, acc_train = training(trainDataLoader, model, lossFn, optimizer, device)
        if epoch%10==0:
            plot=True
        score, loss_val, acc_val = evaluation(testDataLoader, model, lossFn, device, run_id, epoch, plot)
        # print(f'loss_T:{loss_train} acc_T:{acc_train} loss_v:{loss_val} acc_V:{acc_val}')
        print(f'loss_T:{loss_train}  loss_v:{loss_val} ')
        save_model(model, optimizer, epoch, save_dir=f'./saved_models/{run_id}')
        plot = False

        writer.add_scalar('train_loss', loss_train, epoch)
        writer.add_scalar('val_loss', loss_val, epoch)


        # if best_acc>= acc_val:
        #     incident += 1
        #     print(f"No improvement for {incident} epoch(s)")
        #     if incident == 10:
        #         print('model is not improving. Training is stopped')
        #         last_epoch = epoch
        #         break
        # else:
        #     best_accuracy = acc_val
        #     print("Improvement of Val accuracy: {:.4f}\n".format(valCorrect))
        #     save_model(model, opt, epoch, save_dir='./saved_models')
        #     incident = 0





if __name__=="__main__":
    run_id = datetime.today().strftime('%m-%d-%y_%H%M')
    print(f'$$$$$$$$$$$$$ run_id:{run_id} $$$$$$$$$$$$$')

    Main(run_id)
