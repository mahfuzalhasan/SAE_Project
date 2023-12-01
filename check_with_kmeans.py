import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader # Pytorch data loading utility
from torch.optim import Adam # Optimizer

from dataset import DatasetKMNIST
from train import training
from test import evaluation
from utils import save_model, correntropy, add_noise
from model import StackedAutoencoder, Classifier
from tensorboardX import SummaryWriter

from datetime import datetime
import numpy as np
from metrices import *

def get_data(dataset):
    imgs = []
    labels = []
    for i in range(len(dataset)):
        x, y = dataset.noisy_data[i]
        x = np.array(x)
        # print(y, type(y))
        imgs.append(x)
        labels.append(y)
    return np.asarray(imgs), np.asarray(labels)



train_dataset = DatasetKMNIST(root_path ='./data' , train=True)
test_dataset = DatasetKMNIST(root_path ='./data' , train=False)

# train_data, train_label = get_data(train_dataset)
# test_data, test_label = get_data(test_dataset)
# train_data = add_noise(train_data, 128)
# test_data = add_noise(test_data, 128)

# train_data = train_data.astype(float) / 255.
# test_data = test_data.astype(float) / 255.
# X = np.concatenate((train_data, test_data))
# Y = np.concatenate((train_label, test_label))

# print(train_data.shape, test_data.shape)

# kmeans = KMeans(n_clusters=10)
kmeans = KMeans(n_clusters = 10)
# kmeans.fit(np.reshape(train_data, [-1, 28*28]))
# # kmeans = KMeans(n_clusters=10, random_state=0).fit(np.reshape(train_data, [-1, 28*28]))
# y_pred_test = kmeans.predict(np.reshape(test_data, [-1, 28*28]))
# print("Baseline accuracy on Test Using KMeans {:.2%}.".format(acc(test_label, y_pred_test)))

# y_pred_kmeans = kmeans.fit_predict(np.reshape(train_data, [-1, 28*28]))
# print("Baseline accuracy on Train Using KMeans {:.2%}.".format(acc(train_label, y_pred_kmeans)))
# plot_score(np.reshape(train_data, [-1, 28*28]), train_label, y_pred_kmeans, name="train_noisy")





BATCH_SIZE = 512
EPOCHS = 100
INIT_LR = 1e-3
alpha = 0.1
k = 10
lamda=1
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda:0'
# test_dataset = DatasetKMNIST(root_path ='./data' , train=False)
trainDataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
criterion = nn.MSELoss()
bottleneck = 50
model = StackedAutoencoder(bottleneck, alpha, k)
model.to(device)
ids = [105, 109, 136]
run_id = "11-30-23_0446"
for id in ids:
    print(f"######### weight from epoch {id} ###########")
    weight_path = f"./saved_models/{run_id}/model_{id}.pth"
    state_dict = torch.load(weight_path)['model']
    model.load_state_dict(state_dict, strict=True)
    latent_vec = []
    train_label = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(trainDataLoader):
            (x, y) = (x.to(device), y.to(device))
            pred, _, _, latent = model(x)
            latent_vec.append(latent)
            train_label.append(y)
            # x = x.squeeze(dim=1)
            # # print(pred.shape, x.shape)
            # loss = custom_loss(pred, x, non_anchor, anchor)
            # losses.append(loss.item())
            # # valCorrect += (pred.argmax(1) == y).type(
            # #     torch.float).sum().item()
            # total_sample += x.size(0)
        latent_vec = torch.cat(latent_vec, dim=0)
        latent_vec = latent_vec.detach().cpu().numpy()
        
        train_label = torch.cat(train_label, dim=0)
        train_label = train_label.detach().cpu().numpy()

        print(f'latent_vec for model_{id}:{latent_vec.shape} label:{train_label.shape} ')
        

        y_pretrained = kmeans.fit_predict(latent_vec)
        print(f'predicted label kmeans: {y_pretrained.shape}')
        print("Baseline accuracy on Latent Using KMeans {:.2%}.".format(acc(train_label, y_pretrained)))
        plot_score(latent_vec, train_label, y_pretrained, name=f'pre_{run_id}_{id}')