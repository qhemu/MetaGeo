# -*- coding: UTF-8 -*-
import torch
import argparse
import numpy as np
import os
import gzip
import pickle
import hickle
from sklearn.metrics import f1_score
import torch.nn.functional as F

# from torch_sparse import spspmm


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.
    将scipy稀疏矩阵转换为torch稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  # 9475,9475
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # 2,163785
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)  # 163785
    return torch.sparse.FloatTensor(indices, values, shape)


def load_obj(filename, serializer=pickle):  # './data/cmu/dump64.pkl'
    if serializer == hickle:
        obj = serializer.load(filename)
    else:
        with gzip.open(filename, 'rb') as fin:
            obj = serializer.load(fin, encoding='iso-8859-1')
            # obj = pickle.load(fin)
    return obj


def dump_obj(obj, filename, protocol=-1, serializer=pickle):
    if serializer == hickle:
        serializer.dump(obj, filename, mode='w', compression='gzip')
    else:
        with gzip.open(filename, 'wb') as fout:
            serializer.dump(obj, fout, protocol)


def sgc_precompute(data_args, features, adj, degree):
    for i in range(degree):  # degree4
        # torch.Size([9475, 9475]) torch.Size([9475, 9467])
        features = torch.spmm(adj, features)
        # features = F.relu(features)
    # data_dir = data_args.dir  # './data/cmu/'
    # feature_dump_file = os.path.join(data_dir, 'feature_dump.pkl')
    # print("save features")
    # dump_obj(features, feature_dump_file)
    # print('successfully dump data in {}'.format(str(feature_dump_file)))  #
    # 存？？
    return features  # 9475,9467


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro, macro


def shufflelists(lists):        # 多个序列以相同顺序打乱
    ri = np.random.permutation(len(lists[1]))
    out = []
    for l in lists:
        out.append(l[ri])
    return out


def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """
    parser = argparse.ArgumentParser()

    # data_args: control the data loading
    # -dir ./data/cmu/ -bucket 50 -celebrity 5
    # -dir ./data/na/ -bucket 2400 -celebrity 15
    # -dir ./data/world/ -bucket 2400 -celebrity 5
    parser.add_argument('-dir', metavar='str', help='the detail directory of dataset',
                        type=str, default='./data/cmu/dump-512')
    parser.add_argument(
        '-bucket',
        metavar='int',
        help='discretisation bucket size',
        type=int,
        default=50)
    parser.add_argument(
        '-mindf',
        metavar='int',
        help='minimum document frequency in BoW',
        type=int,
        default=10)
    parser.add_argument(
        '-encoding',
        metavar='str',
        help='Data Encoding (e.g.latin1, utf-8)',
        type=str,
        default='latin1')
    parser.add_argument(
        '-celebrity',
        metavar='int',
        help='celebrity threshold',
        type=int,
        default=5)
    parser.add_argument(
        '-vis',
        metavar='str',
        help='visualise representations',
        type=str,
        default=None)
    parser.add_argument(
        '-builddata',
        action='store_true',
        help='if true do not recreated dumped data',
        default=False)
    # parser.add_argument('-builddata', action='store_true', help='if true do not recreated dumped data', default=True)

    # process_args: control the data preprocess
    parser.add_argument(
        '-degree',
        type=int,
        help='degree of the approximation.',
        default=4)
    parser.add_argument('-normalization', type=str, help='Normalization method for the adjacency matrix.',
                        choices=['NormLap', 'Lap', 'RWalkLap', 'FirstOrderGCN', 'AugNormAdj', 'NormAdj',
                                 'RWalk', 'AugRWalk', 'NoNorm'], default='AugNormAdj')

    # model_args: the hyper-parameter of SGC model
    parser.add_argument(
        '-model',
        type=str,
        help='model to use.',
        default="SGC")
    parser.add_argument(
        '-usecuda',
        action='store_true',
        help='Use CUDA for training.',
        default=True)
    parser.add_argument(
        '-mamlusecuda',
        action='store_true',
        help='Use CUDA for training.',
        default=False)
    parser.add_argument(
        '-tune',
        action='store_true',
        help='if true use tuned hyper_params',
        default=False)
    parser.add_argument(
        '-seed',
        metavar='int',
        help='random seed',
        type=int,
        default=77)
    # parser.add_argument('-lr', type=float, help='Initial learning rate.', default=0.00001)
    parser.add_argument(
        '-save',
        action='store_true',
        help='if true, save the model after training',
        default=False)
    parser.add_argument(
        '-load',
        action='store_true',
        help='if true, load pretrained model from file',
        default=False)

    parser.add_argument('--n_way', type=int, help='n way', default=10)
    parser.add_argument(
        '--k_spt',
        type=int,
        help='k shot for support set',
        default=5)
    parser.add_argument(
        '--k_qry',
        type=int,
        help='k shot for query set',
        default=30)
    parser.add_argument(
        '--task_num',
        type=int,
        help='meta batch size, namely task num',
        default=32)
    parser.add_argument(
        '--number_of_training_steps_per_iter',
        type=int,
        default=5)
    parser.add_argument('--multi_step_loss_num_epochs', type=int, default=10)
    parser.add_argument('--splt_len', type=int, default=30)  # 75%96-350-220
    # parser.add_argument('--splt_len', type=int,default=190)#190-500-400
    parser.add_argument('-dge', action='store_true', default=False)
    parser.add_argument('-uselw', action='store_true', default=False)
    parser.add_argument('-usenormaliza', action='store_true', default=True)
    parser.add_argument('-splt', action='store_true', default=False)
    parser.add_argument('-dataset', type=str, help='.',
                        choices=['GeoTEXT', 'Twitter-US', 'Twitter-WORLD'], default='GeoTEXT')
    parser.add_argument(
        '-epochs',
        type=int,
        help='Number of epochs to train.',
        default=500)  # 150
    parser.add_argument(
        '-lr',
        type=float,
        help='Initial learning rate.',
        default=0.00001)
    parser.add_argument(
        '-weight_decay',
        type=float,
        help='Weight decay (L2 loss on parameters).',
        default=5e-8)
    parser.add_argument(
        '-patience',
        help='max iter for early stopping',
        type=int,
        default=35)
    parser.add_argument(
        '--update_lr',
        type=float,
        help='task-level inner update learning rate',
        default=0.05)
    parser.add_argument(
        '--update_step',
        type=int,
        help='task-level inner update steps',
        default=5)
    parser.add_argument(
        '--update_step_test',
        type=int,
        help='update steps for finetunning',
        default=10)
    args = parser.parse_args(argv)
    return args
