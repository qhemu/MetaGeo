# -*- coding: UTF-8 -*-
import sys
import os
import re
import csv
import gzip
import pickle
import hickle
import kdtree
import torch
import numpy as np
import pandas as pd
import networkx as nx
import _pickle as pickle
from haversine import haversine
from scipy._lib.six import xrange
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict, OrderedDict
from sklearn.neighbors import NearestNeighbors
from normalization import fetch_normalization
from utils import sgc_precompute, parse_args
from utils import sparse_mx_to_torch_sparse_tensor
import math
import gensim
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from gensim.models import doc2vec
import random
import re
from sklearn.externals import joblib


def dump_obj(obj, filename, protocol=-1, serializer=pickle):
    if serializer == hickle:
        serializer.dump(obj, filename, mode='w', compression='gzip')
    elif serializer == pickle:
        with gzip.open(filename, 'wb') as fout:
            serializer.dump(obj, fout, protocol)
    else:
        print("dd")
        joblib.dump(obj, filename)


def load_obj(filename, serializer=pickle):
    if serializer == hickle:
        obj = serializer.load(filename)
    elif serializer == hickle:
        with gzip.open(filename, 'rb') as fin:
            obj = serializer.load(fin)
    else:
        obj = joblib.load(filename)
    return obj


# 加边，m与相邻节点，m的相邻节点间加
# def efficient_collaboration_weighted_projected_graph(B, nodes, User, args, train_len):
#     nodes = set(nodes)
#     G = nx.Graph()
#     G.add_nodes_from(nodes)
#     all_nodes = set(B.nodes())
#     for m in all_nodes:
# 		nbrs = B[m]
# 		target_nbrs = [t for t in nbrs if t in nodes]
# 		# add edge between known nodesA(m) and known nodesB(n)
# 		if m in nodes:
# 			if m<train_len:
# 				locm = User[m].split(',')
# 				latm, lonm = float(locm[0]), float(locm[1])
# 				for n in target_nbrs:
# 					if n>train_len:
# 						if m < n:
# 							if not G.has_edge(m, n):
# 								G.add_edge(m, n)
# 					else:
# 						#如果m,n 都在train集中,mention对象距离大于del_dis不加入
# 						location = User[n].split(',')
# 						lat, lon = float(location[0]), float(location[1])
# 						distance = haversine((lat, lon), (latm, lonm))
# 						if distance< args.del_dis:
# 							if m < n:
# 								if not G.has_edge(m, n):
# 									G.add_edge(m, n)
# 			else:
# 				for n in target_nbrs:
# 					if m < n:
# 						if not G.has_edge(m, n):
# 							G.add_edge(m, n)
#
# 		# add edge between known n1 and known n2,
# 		# just because n1 and n2 have relation to m
# 		for n1 in target_nbrs:
# 			if n1<train_len:
# 				locm = User[n1].split(',')
# 				latm, lonm = float(locm[0]), float(locm[1])
# 				for n2 in target_nbrs:
# 					if n2<train_len:
# 						location = User[n2].split(',')
# 						lat, lon = float(location[0]), float(location[1])
# 						distance = haversine((lat, lon), (latm, lonm))
# 						if distance< args.del_dis:
# 							if n1 < n2:
# 								if not G.has_edge(n1, n2):
# 									G.add_edge(n1, n2)
# 					else:
# 						if n1 < n2:
# 							if not G.has_edge(n1, n2):
# 								G.add_edge(n1, n2)
# 			else:
# 				for n2 in target_nbrs:
# 					if n1 < n2:
# 						if not G.has_edge(n1, n2):
# 							G.add_edge(n1, n2)
#
# 	return G

def efficient_collaboration_weighted_projected_graph2(
        B, nodes, User, args, train_len):
    # B:        the whole graph including known nodes and mentioned nodes   --large graph
    # nodes:    the node_id of known nodes
    # --small graph node
    nodes = set(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    all_nodes = set(B.nodes())
    for m in all_nodes:
        nbrs = B[m]
        target_nbrs = [t for t in nbrs if t in nodes]
        if m in nodes:  # add edge between known nodesA(m) and known nodesB(n)
            for n in target_nbrs:
                if m < n:
                    if not G.has_edge(m, n):
                        G.add_edge(
                            m, n)  # Morton added for exclude the long edges
        # add edge between known n1 and known n2,
        # just because n1 and n2 have relation to m, why ? ? ?
        for n1 in target_nbrs:
            for n2 in target_nbrs:
                if n1 < n2:
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2)
    return G

# normalization according to row, each row represent a feature
# 按行归一化，每一行代表一个特征
# 均值和标准差z-score 标准化


def feature_normalization1(dt):
    mean_num = np.mean(dt, axis=0)  # 返回平均值
    sigma = np.std(dt, axis=0)  # 返回标准偏差
    return (dt - mean_num) / sigma


# normalization according to row, each row represent a feature
# (-1,1)标准化：
# 通过遍历feature
# vector里的每一个数据，将Max和Min的记录下来，并通过Max-Min作为基数（即Min=0，Max=1）进行数据的归一化处理：


def feature_normalization2(dt):
    mean_num = np.mean(dt, axis=0)
    max_num = np.max(dt, axis=0)
    min_num = np.min(dt, axis=0)
    # 将数据映射到[-1,1] # return (dt-max_num)/(max_num-min_num)#将数据映射到[0,1]
    return (dt - mean_num) / (max_num - min_num)


def feature_normalization3(data):
    mins = data.min(0)
    maxs = data.max(0)
    means = data.mean(0)
    ranges = maxs - mins
    norData = np.zeros(np.shape(data))
    row = data.shape[0]
    norData = data - np.tile(means, (row, 1))
    norData = norData / np.tile(ranges, (row, 1))
    return norData  # 将数据映射到[-1,1]
    # return (dt-max_num)/(max_num-min_num)#将数据映射到[0,1]


# def process_data(data,taskgraph,args, normalization="AugNormAdj",
# usecuda=True):
def process_data(data, args, normalization="AugNormAdj", usecuda=True):
    # '''
    #   图：
    #    adj,tuple-(9475,9475)
    #   X tweet文本？？Y是该用户所属的类
    #    X_train,tuple-(5685,9467)
    #    Y_train,tuple-(5685,)
    #    X_dev, tuple-(1895,9467)
    #    Y_dev, tuple-(1895,)
    #    X_test, tuple-(1895,9467)
    #    Y_test, tuple-(1895,)
    #  用户：
    #    U_train, list-5685
    #    U_dev, list-1895
    #    U_test, list-1895
    #  129个类的经纬度：
    #    classLatMedian, dict-129
    #    classLonMedian, dict-129
    #  用户经纬度：
    #    userLocation,dict-9475
    # '''
    adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, \
        classLatMedian, classLonMedian, userLocation, tf_idf_num = data  # , tf_idf_num

    '''porting to pyTorch and concat the matrix'''
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)  # 9475,9475
    adj = sparse_mx_to_torch_sparse_tensor(
        adj).float()  # torch.Size([9475, 9475])

    '''
	context_features
	features = np.vstack((X_train, X_dev, X_test))
	if args.usenormaliza == True:
		print("using feature_normalization1...")
		features = feature_normalization1(features)

	sum(tfidf)*context_features
	features1 = np.vstack((X_train, X_dev, X_test))
	features2 = tf_idf_num
	if args.usenormaliza == True:
		print("using feature_normalization2...")
		features1 = feature_normalization2(features1)
		# features1 = feature_normalization3(features1)#203.5  352.6
	features1 = torch.FloatTensor(features1)
	features2 = torch.FloatTensor(features2)
	features= features1.view(features1.size(1), -1)*features2
	features=features.view(features.size(1), -1)
	'''

    # tf-idf
    if args.usenormaliza == True:
        # X_train = X_train.todense()
        # X_test = X_test.todense()
        # X_dev = X_dev.todense()
        print("using feature_normalization1...")
        features = np.vstack((X_train, X_dev, X_test))
        features = feature_normalization1(features)
        features = torch.FloatTensor(features)

    features = torch.FloatTensor(features)
    print("features:", features.shape)
    '''get labels'''
    labels = torch.LongTensor(np.hstack((Y_train, Y_dev, Y_test)))  # 一维9475

    '''get index of train val and test'''
    len_train = int(X_train.shape[0])  # 5685
    len_val = int(X_dev.shape[0])  # 1895
    len_test = int(X_test.shape[0])  # 1895
    # 构建label tensor
    idx_train = torch.LongTensor(range(len_train))  # torch.Size([5685])
    idx_val = torch.LongTensor(
        range(
            len_train,
            len_train +
            len_val))  # [5685……7579]
    idx_test = torch.LongTensor(
        range(
            len_train +
            len_val,
            len_train +
            len_val +
            len_test))  # [7580……9474]

    '''convert to cuda'''
    if usecuda:
        print("converting data to CUDA format...")
        adj = adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    data = (adj, features, labels, idx_train, idx_val, idx_test, U_train, U_dev,
            U_test, classLatMedian, classLonMedian, userLocation)
    return data


def preprocess_data(data_args):
    """ obtain the parameters """
    data_dir = data_args.dir  # './data/cmu/'
    bucket_size = data_args.bucket  # 50
    encoding = data_args.encoding  # 'latin1'
    celebrity_threshold = data_args.celebrity  # 5
    mindf = data_args.mindf  # 10
    builddata = data_args.builddata  # False
    vocab_file = os.path.join(data_dir, 'vocab.pkl')  # './data/cmu/vocab.pkl'
    dump_file = os.path.join(data_dir, 'dump.pkl')  # './data/cmu/dump.pkl'
    # taskgraph_dump_file = os.path.join(data_dir, 'taskgraph_dump.pkl')
    # if os.path.exists(taskgraph_dump_file):
    # 	print("prepare task graph")
    # 	taskgraph = load_obj(taskgraph_dump_file)
    # 	print("over load task graph")
    # 	exit()
    # else:
    # 	print("prepare task graph")
    # 	dl = DataLoader(data_home=data_dir, bucket_size=bucket_size, encoding=encoding,
    # 					celebrity_threshold=celebrity_threshold, mindf=mindf,
    # 					token_pattern=r'(?u)(?<![@])#?\b\w\w+\b')
    # 	dl.load_data()
    # 	taskgraph = dl.get_taskgraph(data_args)
    # 	dump_obj(taskgraph, taskgraph_dump_file)
    # 	print("over load task graph")
    if os.path.exists(dump_file):
        if not builddata:
            print('loading data from dumped file...')
            data = load_obj(dump_file)
            return data

    dl = DataLoader(data_home=data_dir, bucket_size=bucket_size, encoding=encoding,
                    celebrity_threshold=celebrity_threshold, mindf=mindf, token_pattern=r'(?u)(?<![@])#?\b\w\w+\b')
    dl.load_data()          # 'user'        df_train          df_dev          df_test
    dl.assignClasses()      # 'lat', 'lon'  train_classes     dev_classes     test_class
    dl.tfidf()            # 'text'        X_train           X_dev           X_test
    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()

    dl.get_graph(data_args)
    dl.train_context_vectors(dl)    # doc2vec 文档向量
    Y_test = dl.test_classes
    Y_train = dl.train_classes
    Y_dev = dl.dev_classes
    X_train = dl.X_train
    X_dev = dl.X_dev
    X_test = dl.X_test
    tf_idf_num = dl.tf_idf_num
    classLatMedian = {
        str(c): dl.cluster_median[c][0] for c in dl.cluster_median}
    classLonMedian = {
        str(c): dl.cluster_median[c][1] for c in dl.cluster_median}

    P_test = [str(a[0]) + ',' + str(a[1])
              for a in dl.df_test[['lat', 'lon']].values.tolist()]
    P_train = [str(a[0]) + ',' + str(a[1])
               for a in dl.df_train[['lat', 'lon']].values.tolist()]
    P_dev = [str(a[0]) + ',' + str(a[1])
             for a in dl.df_dev[['lat', 'lon']].values.tolist()]
    userLocation = {}
    for i, u in enumerate(U_train):
        userLocation[u] = P_train[i]
    for i, u in enumerate(U_test):
        userLocation[u] = P_test[i]
    for i, u in enumerate(U_dev):
        userLocation[u] = P_dev[i]
    adj = nx.adjacency_matrix(dl.graph)
    print('adjacency matrix created.')

    # data = (adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test,classLatMedian, classLonMedian, userLocation,taskgraph)
    data = (adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test,
            classLatMedian, classLonMedian, userLocation, tf_idf_num)
    dump_obj(data, dump_file)
    print('successfully dump data in {}'.format(str(dump_file)))  # 存？？？
    return data


class DataLoader:
    def __init__(self, data_home, bucket_size=50, encoding='utf-8', celebrity_threshold=10, one_hot_labels=False,
                 mindf=10, maxdf=0.2, norm='l2', idf=True, btf=True, tokenizer=None, subtf=False, stops=None,
                 token_pattern=r'(?u)(?<![#@])\b\w\w+\b', vocab=None):
        self.data_home = data_home
        self.bucket_size = bucket_size
        self.encoding = encoding
        self.celebrity_threshold = celebrity_threshold
        self.one_hot_labels = one_hot_labels
        self.mindf = mindf
        self.maxdf = maxdf
        self.norm = norm
        self.idf = idf
        self.btf = btf
        self.tokenizer = tokenizer
        self.subtf = subtf
        self.stops = stops if stops else 'english'
        token_pattern = r'(?u)(?<![#@|,.-_+^……$%&*();:`，。？、：；;《》{}“”~#￥])\b\w\w+\b'
        self.token_pattern = token_pattern
        self.vocab = vocab
        # self.biggraph = None
    # 加载训练数据

    def load_data(self):
        print('loading the dataset from: {}'.format(self.data_home))
        train_file = os.path.join(self.data_home, 'user_info.train.gz')
        dev_file = os.path.join(self.data_home, 'user_info.dev.gz')
        test_file = os.path.join(self.data_home, 'user_info.test.gz')

        df_train = pd.read_csv(train_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                               quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_dev = pd.read_csv(dev_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                             quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_test = pd.read_csv(test_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                              quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_train.dropna(inplace=True)
        df_dev.dropna(inplace=True)
        df_test.dropna(inplace=True)

        df_train['user'] = df_train['user'].apply(lambda x: str(x).lower())
        df_train.drop_duplicates(['user'], inplace=True, keep='last')
        df_train.set_index(['user'], drop=True, append=False, inplace=True)
        df_train.sort_index(inplace=True)  # 排序？

        df_dev['user'] = df_dev['user'].apply(lambda x: str(x).lower())
        df_dev.drop_duplicates(['user'], inplace=True, keep='last')
        df_dev.set_index(['user'], drop=True, append=False, inplace=True)
        df_dev.sort_index(inplace=True)

        df_test['user'] = df_test['user'].apply(lambda x: str(x).lower())
        df_test.drop_duplicates(['user'], inplace=True, keep='last')
        df_test.set_index(['user'], drop=True, append=False, inplace=True)
        df_test.sort_index(inplace=True)

        self.df_train = df_train
        self.df_dev = df_dev
        self.df_test = df_test

    def sum_user(self):
        P_test = [str(a[0]) + ',' + str(a[1])
                  for a in self.df_test[['lat', 'lon']].values.tolist()]
        P_train = [str(a[0]) + ',' + str(a[1])
                   for a in self.df_train[['lat', 'lon']].values.tolist()]
        P_dev = [str(a[0]) + ',' + str(a[1])
                 for a in self.df_dev[['lat', 'lon']].values.tolist()]
        P_data = P_train + P_dev + P_test
        return P_data

    def get_graph(self, args):
        g = nx.Graph()
        # nodes大 'user' set-len:9475 构建新对象new empty set object
        nodes = set(
            self.df_train.index.tolist() +
            self.df_dev.index.tolist() +
            self.df_test.index.tolist())
        assert len(nodes) == len(self.df_train) + len(self.df_dev) + \
            len(self.df_test), 'duplicate target node'
        # 9475
        nodes_list = self.df_train.index.tolist() + self.df_dev.index.tolist() + \
            self.df_test.index.tolist()
        node_id = {node: id for id, node in enumerate(nodes_list)}
        g.add_nodes_from(node_id.values())
        for node in nodes:
            g.add_edge(node_id[node], node_id[node])
        pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
        pattern = re.compile(pattern)
        print('start adding the train graph')
        externalNum = 0
        for i in range(len(self.df_train)):
            user = self.df_train.index[i]
            user_id = node_id[user]
            mentions = [
                m.lower() for m in pattern.findall(
                    self.df_train.text[i])]  # imentions的用户
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
                    externalNum += 1
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(user_id, id)
        print('start adding the dev graph')
        externalNum = 0
        for i in range(len(self.df_dev)):
            user = self.df_dev.index[i]
            user_id = node_id[user]
            mentions = [
                m.lower() for m in pattern.findall(
                    self.df_dev.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
                    externalNum += 1
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)
        print('start adding the test graph')
        externalNum = 0
        for i in range(len(self.df_test)):
            user = self.df_test.index[i]
            user_id = node_id[user]
            mentions = [
                m.lower() for m in pattern.findall(
                    self.df_test.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
                    externalNum += 1
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)
        print('#nodes: %d, #edges: %d' %
              (nx.number_of_nodes(g), nx.number_of_edges(g)))

        celebrities = []
        for i in xrange(len(nodes_list), len(node_id)):
            deg = len(g[i])
            if deg == 1 or deg > self.celebrity_threshold:
                celebrities.append(i)  # 91477
        print('removing %d celebrity nodes with degree higher than %d' %
              (len(celebrities), self.celebrity_threshold))
        g.remove_nodes_from(celebrities)
        print('projecting the graph')
        User = self.sum_user()

        projected_g = efficient_collaboration_weighted_projected_graph2(
            g, range(len(nodes_list)), User, args, len(self.df_train) + len(self.df_dev))
        # nodes: 9475, #edges: 77155
        print(
            '#nodes: %d, #edges: %d' %
            (nx.number_of_nodes(projected_g),
             nx.number_of_edges(projected_g)))
        self.graph = projected_g

    def get_taskgraph(self, args):
        g = nx.Graph()
        # nodes大 'user' set-len:9475 构建新对象new empty set object
        nodes = set(self.df_train.index.tolist())
        assert len(nodes) == len(self.df_train), 'duplicate target node'
        # 9475
        nodes_list = self.df_train.index.tolist()
        node_id = {node: id for id, node in enumerate(nodes_list)}
        g.add_nodes_from(node_id.values())
        for node in nodes:
            g.add_edge(node_id[node], node_id[node])
        pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
        pattern = re.compile(pattern)
        externalNum = 0
        for i in range(len(self.df_train)):
            user = self.df_train.index[i]
            user_id = node_id[user]
            mentions = [
                m.lower() for m in pattern.findall(
                    self.df_train.text[i])]  # imentions的用户
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
                    externalNum += 1
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(user_id, id)
        celebrities = []
        for i in xrange(len(nodes_list), len(node_id)):
            deg = len(g[i])
            if deg == 1 or deg > self.celebrity_threshold:
                celebrities.append(i)  # 91477
        g.remove_nodes_from(celebrities)
        # projected_g = efficient_collaboration_weighted_projected_graph(g, range(len(nodes_list)))
        User = self.sum_user()

        projected_g = efficient_collaboration_weighted_projected_graph2(g, range(len(nodes_list)), User, args,
                                                                        len(self.df_train))

        all_nodes = set(projected_g.nodes())
        nodes = set(range(len(nodes_list)))
        g = {}
        gf = {}
        count = 0
        for m in all_nodes:
            nbrs = projected_g[m]
            target_nbrs = [t for t in nbrs if t in nodes]
            # print("m:",m,"one_hop_nbrs_len:",len(target_nbrs),"one_hop_nbrs:",target_nbrs)
            if len(target_nbrs) == 0:
                count = count + 1
            g.setdefault(m, []).append(target_nbrs)
        print(count)  # 60073
        for k, v in g.items():
            farnbrs = []
            for key, value in enumerate(v[0]):
                tnbrs = g[value]
                farnbrs = farnbrs + tnbrs[0]
            new_li = list(set(farnbrs))
            # print("m:",k,"two_hop_nbrs_len:",len(new_li),"two_hop_nbrs:",new_li)
            gf.setdefault(k, []).append(new_li)

        return gf
        # return g

    def longest_path(self, g):
        nodes = g.nodes()
        pathlen_counter = Counter()
        for n1 in nodes:
            for n2 in nodes:
                if n1 < n2:
                    for path in nx.all_simple_paths(g, source=n1, target=n2):
                        pathlen = len(path)
                        pathlen_counter[pathlen] += 1
        return pathlen_counter

    def tfidf(self):
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, token_pattern=self.token_pattern, use_idf=self.idf,
                                          norm=self.norm, binary=self.btf, sublinear_tf=self.subtf,
                                          min_df=self.mindf, max_df=self.maxdf, ngram_range=(
                                              1, 1),
                                          stop_words=self.stops,
                                          vocabulary=self.vocab, encoding=self.encoding, dtype=np.float32)
        X_train = self.vectorizer.fit_transform(self.df_train.text.values)
        X_dev = self.vectorizer.transform(self.df_dev.text.values)
        X_test = self.vectorizer.transform(self.df_test.text.values)
        sumtfidf = np.vstack(
            (X_train.toarray(),
             X_dev.toarray(),
             X_test.toarray()))
        self.tf_idf_num = np.sum(sumtfidf, axis=1)
        print("training    n_samples: %d, n_features: %d" % X_train.shape)
        print("development n_samples: %d, n_features: %d" % X_dev.shape)
        print("test        n_samples: %d, n_features: %d" % X_test.shape)
    # 分类！！

    def assignClasses(self):
        # kd聚类器
        clusterer = kdtree.KDTreeClustering(bucket_size=self.bucket_size)  # 50
        # <class 'tuple'>: (5685, 2)
        train_locs = self.df_train[['lat', 'lon']].values
        clusterer.fit(train_locs)
        clusters = clusterer.get_clusters()  # <class 'tuple'>: (5685,)
        cluster_points = defaultdict(list)  # 分类后user的经纬度
        for i, cluster in enumerate(clusters):
            cluster_points[cluster].append(
                train_locs[i])  # train_locs未分类的5686个user的经纬度
        print('# the number of clusterer labels is: %d' % len(cluster_points))
        self.cluster_median = OrderedDict()
        for cluster in sorted(cluster_points):
            points = cluster_points[cluster]
            median_lat = np.median([p[0] for p in points])
            median_lon = np.median([p[1] for p in points])
            self.cluster_median[cluster] = (
                median_lat, median_lon)  # self.cluster_median129个类经纬度
        dev_locs = self.df_dev[['lat', 'lon']].values  # 1895经纬度
        test_locs = self.df_test[['lat', 'lon']].values
        # nnbr = NearestNeighbors(n_neighbors=1, algorithm='brute', leaf_size=1, metric=haversine, n_jobs=4)
        nnbr = NearestNeighbors(
            n_neighbors=1,
            algorithm='brute',
            leaf_size=1,
            metric=haversine)
        nnbr.fit(np.array(list(self.cluster_median.values())))
        '''
		self.dev_classes<class 'tuple'>: (1895,) 样本对应的类？
		self.test_classes<class 'tuple'>: (1895,) 样本对应的类
		self.train_classes<class 'tuple'>: (5685,) 样本对应的类
		'''
        self.dev_classes = nnbr.kneighbors(
            dev_locs, n_neighbors=1, return_distance=False)[:, 0]
        self.test_classes = nnbr.kneighbors(
            test_locs, n_neighbors=1, return_distance=False)[:, 0]

        self.train_classes = clusters

        if self.one_hot_labels:
            num_labels = np.max(self.train_classes) + 1
            y_train = np.zeros(
                (len(
                    self.train_classes),
                    num_labels),
                dtype=np.float32)
            y_train[np.arange(len(self.train_classes)), self.train_classes] = 1
            y_dev = np.zeros(
                (len(
                    self.dev_classes),
                    num_labels),
                dtype=np.float32)
            y_dev[np.arange(len(self.dev_classes)), self.dev_classes] = 1
            y_test = np.zeros(
                (len(
                    self.test_classes),
                    num_labels),
                dtype=np.float32)
            y_test[np.arange(len(self.test_classes)), self.test_classes] = 1
            self.train_classes = y_train
            self.dev_classes = y_dev
            self.test_classes = y_test

    def cleanText(self, corpus, mode):
        documents = []
        for i, text in enumerate(corpus):
            stop_words = [
                'i',
                'me',
                'my',
                'myself',
                'we',
                'our',
                'ours',
                'ourselves',
                'you',
                "you're",
                "you've",
                "you'll",
                "you'd",
                'your',
                'yours',
                'yourself',
                'yourselves',
                'he',
                'him',
                'his',
                'himself',
                'she',
                "she's",
                'her',
                'hers',
                'herself',
                'it',
                "it's",
                'its',
                'itself',
                'they',
                'them',
                'their',
                'theirs',
                'themselves',
                'what',
                'which',
                'who',
                'whom',
                'this',
                'that',
                "that'll",
                'these',
                'those',
                'am',
                'is',
                'are',
                'was',
                'were',
                'be',
                'been',
                'being',
                'have',
                'has',
                'had',
                'having',
                'do',
                'does',
                'did',
                'doing',
                'a',
                'an',
                'the',
                'and',
                'but',
                'if',
                'or',
                'because',
                'as',
                'until',
                'while',
                'of',
                'at',
                'by',
                'for',
                'with',
                'about',
                'against',
                'between',
                'into',
                'through',
                'during',
                'before',
                'after',
                'above',
                'below',
                'to',
                'from',
                'up',
                'down',
                'in',
                'out',
                'on',
                'off',
                'over',
                'under',
                'again',
                'further',
                'then',
                'once',
                'here',
                'there',
                'when',
                'where',
                'why',
                'how',
                'all',
                'any',
                'both',
                'each',
                'few',
                'more',
                'most',
                'other',
                'some',
                'such',
                'no',
                'nor',
                'not',
                'only',
                'own',
                'same',
                'so',
                'than',
                'too',
                'very',
                's',
                't',
                'can',
                'will',
                'just',
                'don',
                "don't",
                'should',
                "should've",
                'now',
                'd',
                'll',
                'm',
                'o',
                're',
                've',
                'y',
                'ain',
                'aren',
                "aren't",
                'couldn',
                "couldn't",
                'didn',
                "didn't",
                'doesn',
                "doesn't",
                'hadn',
                "hadn't",
                'hasn',
                "hasn't",
                'haven',
                "haven't",
                'isn',
                "isn't",
                'ma',
                'mightn',
                "mightn't",
                'mustn',
                "mustn't",
                'needn',
                "needn't",
                'shan',
                "shan't",
                'shouldn',
                "shouldn't",
                'wasn',
                "wasn't",
                'weren',
                "weren't",
                'won',
                "won't",
                'wouldn',
                "wouldn't"]
            interpunction = "||| 《 》 # @ ' —— + - . ! ！ / _ , $ ￥ % ^ * ( ) ] | [ ， 。 ？ ? : ： ； ; 、 ~ …… … & * （ ）".split(
            )
            line = text.lower().split()
            # Remove interpunction.
            line = [w for w in line if w not in interpunction]
            # Remove stopwords.
            line = [w for w in line if w not in stop_words]
            data = ' '.join(line) + "\n"
            documents.append(data)
        data = self.label_sentences(documents, mode)
        return data

    def train_context_vectors(self, dl):
        x_train = dl.cleanText(self.df_train.text.values, mode='Train')
        x_dev = dl.cleanText(self.df_dev.text.values, mode='Dev')
        x_test = dl.cleanText(self.df_test.text.values, mode='Test')
        all_data = x_train + x_dev + x_test
        d2vmodel = self.initialize_model(all_data)
        # 实例1DM和0DBOW模型
        # model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
        # model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)
        # 将训练完成的数据转换为vectors corpus_size, vectors_size, vectors_type
        # 获取数据集的文档向量
        self.X_train = self.get_vectors(
            d2vmodel, x_train, 256, 'Train')  # <class 'tuple'>: (5685, 300)
        self.X_dev = self.get_vectors(
            d2vmodel, x_dev, 256, 'Dev')  # <class 'tuple'>: (5685, 300)
        self.X_test = self.get_vectors(
            d2vmodel, x_test, 256, 'Test')  # <class 'tuple'>: (5685, 300)
        print(
            "training_context_vectors    n_samples: %d, vectors_features: %d" %
            self.X_train.shape)
        print(
            "development_context_vectors n_samples: %d, vectors_features: %d" %
            self.X_dev.shape)
        print(
            "test_context_vectors        n_samples: %d, vectors_features: %d" %
            self.X_test.shape)

    def label_sentences(self, corpus, label_type):
        """
Gensim's Doc2Vec implementation requires each
 document/paragraph to have a label associated with it.
We do this by using the LabeledSentence method.
The format will be "TRAIN_i" or "TEST_i" where "i" is
a dummy index of the review.
"""
        labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            labeled.append(doc2vec.LabeledSentence(v.split(), [label]))
        return labeled

    def initialize_model(self, corpus):
        print("Building Doc2Vec vocabulary")
        '''
		一些参数的选择与对比： 1.skip-gram （训练速度慢，对罕见字有效），CBOW（训练速度快）。一般选择Skip-gram模型； 2.训练方法：Hierarchical Softmax（对罕见字有利），Negative Sampling（对常见字和低维向量有利）； 3.欠采样频繁词可以提高结果的准确性和速度（1e-3~1e-5） 4.Window大小：Skip-gram通常选择10左右，CBOW通常选择5左右。
		1 2-10-512-1e-3-3-5-0-0.75-0-20:36
		2 5-10-512-1e-3-5-5-1-0.75-0-20:36
		3 5-10-512-1e-3-5-         0-15:40.26
		4 5-10-256-1e-3-5-         0-15:29
		'''
        d2vmodel = doc2vec.Doc2Vec(min_count=5,
                                   # Ignores all words with
                                   # total frequency lower than this
                                   window=10,
                                   # The maximum distance between the current
                                   # and predicted word
                                   # within a sentence
                                   vector_size=256,  # Dimensionality of the
                                   # generated feature
                                   # vectors
                                   sample=1e-3,
                                   workers=5,  # Number of worker threads to train the model
                                   # Learning rate will linearly drop to
                                   # min_alpha as training progresses
                                   # dbow_words=1,
                                   negative=5,
                                   hs=1,
                                   ns_exponent=0.75,
                                   dm=0)
        # dm defines the training algorithm.
        #  If dm=1 means 'distributed memory' (PV-DM)
        # and dm =0 means 'distributed bag of words' (PV-DBOW)
        d2vmodel.build_vocab(corpus)
        print("over vocab")
        print("Training Doc2Vec model")
        #  if you have more time/computational power make it 20
        for epoch in range(15):
            print('Training iteration #{0}'.format(epoch))
            d2vmodel.train(
                corpus, total_examples=d2vmodel.corpus_count,
                epochs=d2vmodel.epochs)
            # shuffle the corpus
            random.shuffle(corpus)
            # decrease the learning rate
            d2vmodel.alpha -= 0.0002
            # fix the learning rate, no decay
            d2vmodel.min_alpha = d2vmodel.alpha
        return d2vmodel
    # 读取向量

    def get_vectors(self, model, corpus, vectors_size, vectors_type):
        """
Get vectors from trained doc2vec model
:param doc2vec_model: Trained Doc2Vec model
:param corpus_size: Size of the data
:param vectors_size: Size of the embedding vectors
:param vectors_type: Training or Testing vectors
:return: list of vectors
"""
        vectors = np.zeros((len(corpus), vectors_size))
        for i in range(0, len(corpus)):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = model.docvecs[prefix]
        return vectors


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    dl = DataLoader(data_home='./data/cmu', bucket_size=50, encoding='latin1', celebrity_threshold=5, mindf=10,
                    token_pattern=r'(?u)(?<![@])#?\b\w\w+\b')
    dl.load_data()  # 加载文件
    # dl.tfidf()
    dl.train_context_vectors(dl)
    exit()

    dl.get_taskgraph(args)  # 加载文件
    dl.get_graph(args)  # 构建图

    dl.assignClasses()  # import分类 create the label (number:129)
