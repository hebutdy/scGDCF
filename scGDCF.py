from __future__ import print_function, division
import argparse
#import matplotlib.pyplot as plt
import random
from torch.autograd import Variable
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, structure_graph, feature_graph
from GNN import GNNLayer
from sklearn.manifold import TSNE
from evalution import cluster_acc, eva
from GCN_test1 import GCN, SFGCN
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import r2_score
import pandas as pd
import mpl_toolkits.axisartist
import matplotlib.pyplot as plt
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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
        enc_h1 = F.dropout(enc_h1, p=0.3)
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class Discriminator(nn.Module):
    def __init__(self, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(n_z, n_dec_1)
        self.lin2 = nn.Linear(n_dec_1, n_dec_3)
        self.lin3 = nn.Linear(n_dec_2, n_dec_3)
        self.lin4 = nn.Linear(n_dec_3, n_input)

    def forward(self, x):
        x1 = F.relu(self.lin1(x))
        x2 = F.relu(self.lin2(x1))
        x3 = F.relu(self.lin3(x2))
        x4 = F.relu(self.lin4(x3))
        return x4


class SDCN1(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, dropout, v=1):
        super(SDCN1, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,  
            n_enc_2=n_enc_2,  
            n_enc_3=n_enc_3,  
            n_dec_1=n_dec_1,  
            n_dec_2=n_dec_2,  
            n_dec_3=n_dec_3,  
            n_input=n_input,  
            n_z=n_z  # 10
        )
        self.discriministrator = Discriminator(
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z
        )
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'), False)
        self.gnn_1 = SFGCN(n_input, n_enc_1, dropout)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        # degree
        self.v = v

    def forward(self, x, adj1, adj2):
        # DNN Module
        x.to(device)
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        dis = self.discriministrator(z)
        # GCN Module
        h1, emb1, emb2, com1, com2 = self.gnn_1(x, adj1, adj2)   
        h5 = self.gnn_5(h1, adj1, active=False)   
        predict = F.softmax(h5, dim=1)
        #predict.shape(2100,8)
        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, emb1, emb2, com1, com2, dis

        # return x_bar, q, predict, z


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2) ** 2)
    return cost


def KL_devergence(p, q):
    """
    Calculate the KL-divergence of (p,q)
    :param p:
    :param q:
    :return:
    """
    q = torch.nn.functional.softmax(q, dim=0)
    q = torch.sum(q, dim=0) / 256              
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


def train_sdcn(dataset):
    setup_seed(23)
    feature_adj = feature_graph(args.name, k=3).to(device)
    structure_adj = structure_graph(args.name, k=3).to(device)
    feature_adj = feature_adj.to_dense()
    structure_adj = structure_adj.to_dense()
    model = SDCN1(500, 500, 2000, 2000, 500, 500, n_input=args.n_input, n_z=args.n_z, n_clusters=args.n_clusters,
                  dropout=args.dropout, v=1.0).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)  
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)
    print(z)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    eva(y, y_pred, 'pae')
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    acc_max = 0
    f1_max = 0
    for epoch in range(500):
        if epoch % 1 == 0:
            # update_interval
            _, tmp_q, pred, _, _, _, _, _, _ = model(data, feature_adj, structure_adj)
            # _, tmp_q, pred, _, _, _, _, _ = model(data, feature_adj, adj)
            tmp_q = tmp_q.data   
            p = target_distribution(tmp_q)
            res1 = tmp_q.cpu().numpy().argmax(1)  
            res2 = pred.data.cpu().numpy().argmax(1)  
            res3 = p.data.cpu().numpy().argmax(1)  
            eva(y, res3, str(epoch) + 'P')
            eva(y, res1, str(epoch) + 'Q')
            eva(y, res2, str(epoch) + 'Z')
            acc, f1 = cluster_acc(y, res2)
            nmi = nmi_score(y, res2, average_method='arithmetic')
            ari = ari_score(y, res2)
        x_bar, q, pred, _, emb1, com1, com2, emb2, _ = model(data, feature_adj, structure_adj)
        _, _, _, _, z1 = model.ae(data)
        z1.to(device)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)
        re_loss = r2_score(x_bar.detach().cpu().numpy(), data.detach().cpu().numpy())
        loss_com = common_loss(com1, com2)
        z_real_gauss = Variable(torch.Tensor(np.random.normal(-0.1, 0.1, (2746, 10))))
        z_fake_gauss = z1
        D_real_gauss = model.discriministrator(z_real_gauss)
        D_fake_gauss = model.discriministrator(z_fake_gauss)
        D_loss = torch.mean(D_real_gauss) + torch.mean(D_fake_gauss)
        G_loss = torch.mean(D_fake_gauss)
        loss1 = 0.01 * D_loss + 0.001 * G_loss
        loss = 0.1 * kl_loss + 0.001 * re_loss + 0.01 * ce_loss + 0.001 * loss1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('{} loss: {}'.format(epoch, loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='mouse')
    parser.add_argument('--k', type=int, default=3)    #knn
    parser.add_argument('--lr', type=float, default=1e-3)   #learning rate
    parser.add_argument('--n_clusters', default=16, type=int)   
    parser.add_argument('--n_z', default=10, type=int)         
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=20, help='Alpha for the leaky_relu.')
    parser.add_argument('--update_interval', default=5, type=int)  # [1,3,5]
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cpu" if args.cuda else "cpu")
    args.pretrain_path = 'E:/data/{}.pkl'.format(args.name)
    #C:/Users/Ziyu/Desktop/新建文件夹/实验代码/预训练/{}.pkl
    dataset = load_data(args.name)
    if args.name == 'mouse':
        args.n_clusters = 16
        args.n_input = 4187
    print(args)
    train_sdcn(dataset)



