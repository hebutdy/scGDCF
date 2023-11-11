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


        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) #��ѧϰ��Ȩ��
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features)) #��ѧϰ��ƫ��
        else:
            self.register_parameter('bias', None) #��һ���������밴���ַ�����ʽ����
        self.reset_parameters() #������ʼ��

    def reset_parameters(self):#���������ʼ������
        #size����in_features,out_features,size(1)ָout_features,stdv=1/����(out_features)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv) #weight������(-srdv,stdv)֮����ȷֲ������ʼ��
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)  #bias���ȷֲ������ʼ��

    def forward(self, input, adj): #ǰ�򴫲�
        support=torch.mm(input,self.weight)
        output = torch.spmm(adj, support) #ϡ�����ĳ˷�
        if self.bias is not None:
            return output + self.bias #����ϵ��*����+ƫ��
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
        #nfeat�ײ�ڵ�Ĳ�����feature�ĸ�����nhid����ڵ�ĸ�����nclass���յķ�����
        #GCN������GraphConvolution��Ľ�������Ϊ�������log_softmax�任�Ľ��
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.attention = Attention(nhid1)
        self.tanh = nn.Tanh()

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj)) #ִ��GraphConvolution�е�forward���پ���relu����
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


        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) #��ѧϰ��Ȩ��
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features)) #��ѧϰ��ƫ��
        else:
            self.register_parameter('bias', None) #��һ���������밴���ַ�����ʽ����
        self.reset_parameters() #������ʼ��

    def reset_parameters(self):#���������ʼ������
        #size����in_features,out_features,size(1)ָout_features,stdv=1/����(out_features)
        stdv = 1. / math.sqrt(self.weight.size(1))

        self.weight.data.uniform_(-stdv, stdv) #weight������(-srdv,stdv)֮����ȷֲ������ʼ��
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)  #bias���ȷֲ������ʼ��

    # def reset_parameters(self):
    #     glorot(self.weight)
    #     if self.weight is not None:
    #         glorot(self.weight)
        # zeros(self.bias)

    def forward(self, input, adj): #ǰ�򴫲�
        weight=torch.mul(self.weight,self.weight)
        support = torch.mm(input, weight) #input��self.weight������� 
        output = torch.spmm(adj, support) #ϡ�����ĳ˷�
        if self.bias is not None:
            return output + self.bias #����ϵ��*����+ƫ��
        else:
            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


