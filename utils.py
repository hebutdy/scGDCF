import numpy as np
import scipy.sparse as sp
import h5py
import torch
from torch.utils.data import Dataset


def load_graph(dataset, k):
    if k:
        path = 'C:/Users/KNN/graph/{}{}_graph.txt'.format(dataset, k)
    else:
        path = 'C:/Users/KNN/graph/{}_graph.txt'.format(dataset)
    data = np.loadtxt('C:/Users/data/{}.txt'.format(dataset)).T
    n, _ = data.shape
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class load_data(Dataset):
    def __init__(self, dataset):
        self.x = np.loadtxt('C:/Users/data/{}.txt'.format(dataset), dtype=float)
        self.y = np.loadtxt('C:/Users/data/{}_label.txt'.format(dataset), dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))


def structure_graph(dataset, k):
    if k:
        path = 'C:/Users/KNN/graph/{}{}_graph.txt'.format(dataset, k)
    else:
        path = 'C:/Users/KNN/graph/{}_graph.txt'.format(dataset)
    data = np.loadtxt('C:/Users/data/{}.txt'.format(dataset)).T
    n, _ = data.shape
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    struct_edges = np.genfromtxt(path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(n, n), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    return nsadj


def feature_graph(dataset, k):
    if k:
        path = 'C:/Users/KNN/graph/{}{}_graph.txt'.format(dataset, k)
    else:
        path = 'C:/Users/KNN/graph/{}_graph.txt'.format(dataset)
    data = np.loadtxt('C:/Users/data/{}.txt'.format(dataset)).T
    n, _ = data.shape
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    feature_edges = np.genfromtxt(path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(n, n), dtype=np.float32)  #coo_matrix构建指定维度的矩阵
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    return nfadj
