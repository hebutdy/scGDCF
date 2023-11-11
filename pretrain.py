import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evalution import eva
import torch.nn.functional as F
import pandas as pd


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar, z

class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def pretrain_ae(model, dataset, y):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    # Adam
    optimizer = Adam(model.parameters(), lr=0.001)
    for epoch in range(50):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()#x(1024,1529)
            x_bar, _ = model(x)
            x_bar = x_bar.cpu()
            x = x.cpu()
            loss = F.mse_loss(x_bar, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=Cluster_para[0], n_init=Cluster_para[1]).fit(z.data.cpu().numpy())
            eva(y, kmeans.labels_, epoch) #y=1529 lable=26178

        # Generate a pre-trained model
        torch.save(model.state_dict(), File[0])


File = ['E:/data/mouse.pkl']
# Para = [batch_size, lr, epoch]
# model_para = [n_enc_1(n_dec_3), n_enc_2(n_dec_2), n_enc_3(n_dec_1)]
model_para = [500, 500, 2000]
# Cluster_para = [n_cluster, n_init, n_input, n_z]
Cluster_para = [16, 20, 3187, 10]
model = AE(
            n_enc_1=model_para[0], n_enc_2=model_para[1], n_enc_3=model_para[2],
            n_dec_1=model_para[2], n_dec_2=model_para[1], n_dec_3=model_para[0], n_input=Cluster_para[2], n_z=Cluster_para[3],).cuda()
number = pd.read_csv("E:/scRNA-seq data/worm.csv")
x = number.loc[:, number.columns[1:]].to_numpy()
print(x.shape)
print(x)
label = pd.read_csv("E:/scRNA-seq data/worm_neuron_cell_label.csv")
y = label.loc[:, "true_label"].to_numpy()
dataset = LoadDataset(x)
pretrain_ae(model, dataset, y)