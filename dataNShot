import torch
import  numpy as np

import random


class dataNShot:
    def __init__(self, m_trainloader, m_testloader, batchsz, n_way, k_shot, k_query, all_labels, features, Users, classLatMedian, classLonMedian, userLocation):
        self.batchsz = batchsz  # 32
        self.n_way = n_way  # n way5
        self.k_shot = k_shot  # k shot1
        self.k_query = k_query  # k query15
        self.all_labels = all_labels
        self.features = features
        self.Users = Users
        self.classLatMedian = classLatMedian
        self.classLonMedian = classLonMedian
        self.userLocation = userLocation
        assert (k_shot + k_query) <=1668

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": m_trainloader, "test": m_testloader}  # original data cached
        self.datasets_cache = {
                               "train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"])
                               }

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        为N-shot学习收集几批数据
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :返回:一个列表，其中[support_set_x, support_set_y, target_x, target_y]准备好被提供给我们的网络
        """
        #  take 5 way 1 shot as example: 5 * 1
        # setsz = self.k_shot * self.n_way#5
        # querysz = self.k_query * self.n_way#5
        data_cache = []
        # one_cls, one_data, nbrs_class, selected_cls = self.choose_onedata(data_pack, mode)
        # print('preload next 10 caches of batchsz of batch.')预加载下50批的批处理缓存。
        for sample in range(10):  # num of episodes
            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            spt_locations, qry_locations = [], []
            reallabel = []
            for i in range(self.batchsz):  # one batch means one set
                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                qry_location, spt_location= [], []
                # 5个类 replace = True 在一次抽取中，抽取的样本可重复出现。
                # np.random.choice在data_pack中随机取n_way5类
                selected_cls = random.sample(data_pack['label_index'].keys(), self.n_way)
                for j, cur_class in enumerate(selected_cls):
                    # 16 每个类20取16各
                    selected_data = random.sample(data_pack['label_index'][cur_class][0],self.k_shot + self.k_query)
                    x_spt.append(np.array(self.features[selected_data[:self.k_shot]].cpu()))
                    x_qry.append(np.array(self.features[selected_data[self.k_shot:]].cpu()))
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])
                    # location to compulate correct
                    aa = selected_data[:self.k_shot]
                    for i,v in enumerate(aa):  # support
                        user = self.Users[v]
                        location = self.userLocation[user]
                        spt_location.append(np.array(location))
                    bb = selected_data[self.k_shot:]
                    for i,v in enumerate(bb):  # support
                        user = self.Users[v]
                        location = self.userLocation[user]
                        qry_location.append(np.array(location))
                    # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 512)[perm]  # 3256
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                spt_location = np.array(spt_location).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 512)[perm]  # 3256
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]
                qry_location = np.array(qry_location).reshape(self.n_way * self.k_query)[perm]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)
                spt_locations.append(spt_location)
                qry_locations.append(qry_location)
                reallabel.append(selected_cls)
                # [b, setsz, 2, 16, 16]
            x_spts = np.array(x_spts).astype(np.float32)
            y_spts = np.array(y_spts).astype(np.int)
            spt_locations = np.array(spt_locations)
            # [b, qrysz, 2, 16, 16]
            x_qrys = np.array(x_qrys).astype(np.float32)
            y_qrys = np.array(y_qrys).astype(np.int)
            qry_locations = np.array(qry_locations)
            data_cache.append([x_spts, y_spts, x_qrys, y_qrys,spt_locations,qry_locations,reallabel])

            del x_spts
            del y_spts
            del x_qrys
            del y_qrys
            return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        从具有名称的数据集中获取下一批数据。
        :param模式:拆分名称(其中“train”、“val”、“test”)
        """
        # update cache if indexes is larger cached num 如果索引缓存的num较大，则更新缓存
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])
        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch


def dealdata1(args, labels):
    d = {}
    oldtr = labels  # 5685
    for k, v in enumerate(oldtr.cpu()):
        d.setdefault(v.item(), []).append(k)
    g = {k: v for k, v in d.items() if len(v) != 1}  # 128
    valset = {}
    train_cls = np.random.choice(labels.max().item() + 1,args.splt_len,False)
    # print(sorted(train_cls))
    metatraincount = 0
    metatestcount = 0
    train_index = {}
    val_index = {}
    for k, v in g.items():
        # if k in train_cls:
        # sss.append(len(v))
        if k > args.splt_len:
            metatraincount = metatraincount + len(v)
            train_index.setdefault(k, []).append(v)
            trainset = {'label_index': train_index}
        else:
            metatestcount = metatestcount + len(v)
            val_index.setdefault(k, []).append(v)
            valset = {'label_index': val_index}
    print("metatraincount：", metatraincount)
    print("metatestcount：", metatestcount)
    return trainset, valset


def dealdata(args, labels):
    d = {}
    oldtr = labels  # 5685
    for k, v in enumerate(oldtr.cpu()):
        d.setdefault(v.item(), []).append(k)
    g = {k: v for k, v in d.items() if len(v) != 1}  # 128
    valset = {}
    train_cls = np.random.choice(labels.max().item() + 1, args.splt_len, False)
    print(sorted(train_cls))
    metatraincount = 0
    metatestcount = 0
    train_index = {}
    val_index = {}
    for k, v in g.items():
        # if k in train_cls:
        # sss.append(len(v))
        if k > args.splt_len:
            metatraincount = metatraincount + len(v)
            train_index.setdefault(k, []).append(v)
            trainset = {'label_index': train_index}
        else:
            metatestcount = metatestcount + len(v)
            val_index.setdefault(k, []).append(v)
            valset = {'label_index': val_index}
    print("metatraincount：", metatraincount)
    print("metatestcount：", metatestcount)
    return trainset, valset

