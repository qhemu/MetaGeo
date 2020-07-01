import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math


class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    逻辑回归的一个简单PyTorch实现。假设特征已经过k步图传播预处理。
    """

    def __init__(self, nfeat, nclass, n_way):
        super(SGC, self).__init__()
        self.W = nn.Linear(nfeat, nclass, bias=True)  # 9467,256
        self.W2 = nn.Linear(nclass, n_way, bias=True)  # 256,129
        # self.W2 = nn.Linear(600, nclass, bias=True)#256,129
        # self.acv = nn.ReLU()#9467,129
        # self.acv = nn.Tanh()#9467,129

    def forward(self, x, vars=None, mode=True):  # torch.Size([5, 9467])
        # print(vars)
        if vars is not None:
            vars = list(vars)
            idx = 0
            w, b = vars[idx], vars[idx + 1]
            y = F.linear(x, w, b)
            w2, b2 = vars[idx + 2], vars[idx + 3]
            y = F.linear(y, w2, b2)
            # w3, b3 = vars[idx + 4], vars[idx + 5]
            # y = F.linear(y, w3, b3)
        else:
            y = self.W(x)  # torch.Size([5, 129])
            y = self.W2(y)  # torch.Size([5, 129])
            # y= self.W3(y)#torch.Size([5, 129])
        # if mode==True:
        #     y=F.dropout(y,0.1,True)
        # y = self.acv(y)
        return y


class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input)
        output = torch.spmm(adj, support)
        return output


class GCN(nn.Module):
    """
    A Two-layer GCN.
    """

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


def get_model(model_opt, n_way, nfeat, nclass,
              nhid=0, dropout=0, usecuda=False):
    '''
    :param model_opt: SGC
    :param nfeat: 9467
    :param nclass: 129
    :param nhid: 0
    :param dropout: 0
    :param usecuda:
    :return:
    '''
    if model_opt == "GCN":
        model = GCN(nfeat=nfeat,
                    nhid=nhid,  # 0
                    nclass=nclass,
                    dropout=dropout)  # 0
    elif model_opt == "SGC":
        model = SGC(nfeat=nfeat,  # 9467
                    nclass=nclass,
                    n_way=n_way
                    )  # 129
    else:
        raise NotImplementedError(
            'model:{} is not implemented!'.format(model_opt))

    if usecuda:
        model.cuda()
    return model
