import math
import torch
import torch.nn
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
from numpy import *



class GraphConvolution(Module):


    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) #可学习的权重
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features)) #可学习的偏差
        else:
            self.register_parameter('bias', None) #第一个参数必须按照字符串形式输入
        self.reset_parameters() #参数初始化

    def reset_parameters(self):#参数随机初始化函数
        #size包括in_features,out_features,size(1)指out_features,stdv=1/根号(out_features)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv) #weight在区间(-srdv,stdv)之间均匀分布随机初始化
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)  #bias均匀分布随机初始化

    def forward(self, input, adj): #前向传播
        support=torch.mm(input,self.weight)
        output = torch.spmm(adj, support) #稀疏矩阵的乘法
        if self.bias is not None:
            return output + self.bias #返回系数*输入+偏置
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class GCN(Module):
    def __init__(self, nfeat, nhid1,nhid2,dropout):
        #nfeat底层节点的参数，feature的个数，nhid隐层节点的个数，nclass最终的分类数
        #GCN是两个GraphConvolution层的结果，输出为输出层做log_softmax变换的结果
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.attention = Attention(nhid1)
        self.tanh = nn.Tanh()

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj)) #执行GraphConvolution中的forward，再经过relu函数
        x2 = F.dropout(x1, self.dropout,training=self.training)
        x3 = self.gc2(x2, adj)
        # x3 = F.dropout(x2,p=0.5)
        # _, att = self.attention(x1)
        #
        # return F.log_softmax(0.01*x3+0.99*att)

        return F.log_softmax(x3)


class SFGCN(nn.Module):
    def __init__(self, nfeat, nhid2, dropout): 
        super(SFGCN, self).__init__()

        self.SGCN1 = GraphConvolution(nfeat,  nhid2)
       
        self.SGCN2 = GraphConvolution(nfeat, nhid2)

        self.CGCN = GraphConvolution(nfeat, nhid2)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()



    def forward(self, x, sadj, fadj): #x:1529,26178
        emb1 = self.SGCN1(x, sadj) # Special_GCN out1 -- sadj structure graph
        com1 = self.CGCN(x, sadj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x, fadj)  # Common_GCN out2 -- fadj feature graph
        emb2 = self.SGCN2(x, fadj) # Special_GCN out2 -- fadj feature graph
        Xcom=torch.mul(com1,com2)
        Xcom = F.dropout(Xcom, p=0.3)
        ##attention
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        output, att = self.attention(emb)
        # output = self.MLP(emb)
        # return output, att, emb1, com1, com2, emb2, emb


        return output, emb1, com1, com2, emb2

class GraphConvolution(Module):


    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) #可学习的权重
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features)) #可学习的偏差
        else:
            self.register_parameter('bias', None) #第一个参数必须按照字符串形式输入
        self.reset_parameters() #参数初始化

    def reset_parameters(self):#参数随机初始化函数
        #size包括in_features,out_features,size(1)指out_features,stdv=1/根号(out_features)
        stdv = 1. / math.sqrt(self.weight.size(1))

        self.weight.data.uniform_(-stdv, stdv) #weight在区间(-srdv,stdv)之间均匀分布随机初始化
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)  #bias均匀分布随机初始化

    # def reset_parameters(self):
    #     glorot(self.weight)
    #     if self.weight is not None:
    #         glorot(self.weight)
        # zeros(self.bias)

    def forward(self, input, adj): #前向传播
        weight=torch.mul(self.weight,self.weight)
        support = torch.mm(input, weight) #input和self.weight矩阵相乘 
        output = torch.spmm(adj, support) #稀疏矩阵的乘法
        if self.bias is not None:
            return output + self.bias #返回系数*输入+偏置
        else:
            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


