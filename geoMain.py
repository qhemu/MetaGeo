#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import torch
import time
import numpy as np
from meta import Meta  # maml
from haversine import haversine
from models import get_model
from utils import sgc_precompute, parse_args
from dataNShot import dealdata, dealdata1
from dataNShot import dataNShot
from dataProcess import preprocess_data, process_data, load_obj
import os

# def train_regression(model, train_features, train_labels, val_features,
# U_dev,


def train_regression(maml, labels, features, users, trainset, valset, classLatMedian, classLonMedian, userLocation,
                     epochs=300, patience=10, model_file='myModel.pkl'):

    train_patience = 0
    val_acc_best = -1
    epoch_best = 0
    db_train = dataNShot(
        m_trainloader=trainset,
        m_testloader=valset,
        batchsz=args.task_num,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry,
        all_labels=labels,
        features=features,
        Users=users,
        classLatMedian=classLatMedian,
        classLonMedian=classLonMedian,
        userLocation=userLocation
    )
    # exit()

    for epoch in range(0, epochs):
        # X, Y = shufflelists([X, Y])
        x_spt, y_spt, x_qry, y_qry, spt_locations, qry_locations, reallabel = db_train.next(
            'train')
        b_idx = 0           # batch计数
        while b_idx <= 350:
            maml.xttrain()
            loss, trainaccs = maml(epoch, x_spt, y_spt, x_qry, y_qry)
            b_idx += 1     # 更新batch计数
            if b_idx % 10 == 0:
                step = epoch * 350 + b_idx
                print(
                    "step:",
                    step,
                    "train loss:",
                    loss,
                    "meta_train classification acc:",
                    trainaccs)
        '''verification, no gradient descent验证，微调'''
        all_clc_accs, all_means, all_medians, all_acc161s = [], [], [], []
        for _ in range(220 // args.task_num):
            # d_test
            x_spt, y_spt, x_qry, y_qry, spt_locations, qry_locations, reallabel = db_train.next(
                'test')
            # all_loss,testaccs = maml(epoch,x_spt, y_spt, x_qry, y_qry)
            # split to single task each time
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one, spt_locations_one, qry_locations_one, reallabel_one in zip(
                    x_spt, y_spt, x_qry, y_qry, spt_locations, qry_locations, reallabel):
                # 		#一个task的acc
                clc_accs, means, medians, acc161s = maml.finetunning(
                    x_spt_one, y_spt_one, x_qry_one, y_qry_one, spt_locations_one, qry_locations_one, reallabel_one)
                # print(clc_accs)
                all_clc_accs.append(clc_accs)
                all_means.append(means)
                all_medians.append(medians)
                all_acc161s.append(acc161s)
        # [b, update_step+1]
        class_acc = np.array(all_clc_accs).mean(axis=0).astype(np.float16)
        meanDis = np.array(all_means).mean(axis=0).astype(np.float16)
        MedianDis = np.array(all_medians).mean(axis=0).astype(np.float16)
        accAT161 = np.array(all_acc161s).mean(axis=0).astype(np.float16)
        # print('meta_Test classification acc:', class_acc,'mean:',meanDis,'median:',MedianDis,'acc161:',accAT161)
        print(
            'Train epoch:',
            epoch,
            'meta_Test mean:',
            meanDis,
            'median:',
            MedianDis,
            'acc161:',
            accAT161)

        '''apply early stop using val_acc_best, and save model'''
        val_acc = max(accAT161)
        if val_acc >= val_acc_best:
            val_acc_best = val_acc
            epoch_best = epoch
            train_patience = 0
            maml.statedict(model_file)
        else:
            train_patience += 1
        '''show val_acc,val_acc_best every 50 epoch'''
        if train_patience == patience:
            # if epoch == epochs-1:
            print(
                "epoch_best:{}\t \t \tval_acc_best:{}".format(
                    epoch_best, val_acc_best))
            break
    return class_acc, meanDis, MedianDis, accAT161


def main():
    """
    preprocess_data() :     load data from dataset and precess data into numpy format
    process_data() :        port the data to pyTorch and convert to cuda
    U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation : only use when Valid and Test
    """
    data = preprocess_data(args)
    data = process_data(data, args, args.normalization, args.usecuda)

    (adj, features, labels, idx_train, idx_val, idx_test, U_train, U_dev, U_test,
     classLatMedian, classLonMedian, userLocation) = data
    # exit()
    model_file = "./result_cmu_desce/{}way{}shot{}query-update_lr:{}-weight_decay:{}.pkl".format(
        args.n_way, args.k_spt, args.k_qry, args.update_lr, args.weight_decay)

    device = torch.device('cuda')
    # maml = Meta(args, config).to(device)
    maml = Meta(
        args,
        features.shape[1],
        labels.max().item() + 1,
        classLatMedian,
        classLonMedian).to(device)
    if args.model == "SGC":
        feature_dump_file = os.path.join(args.dir, 'feature_dump.pkl')
        # if os.path.exists(feature_dump_file):
        # 	print("load features")
        # 	features = load_obj(feature_dump_file)
        # else:
        features = sgc_precompute(args, features, adj, args.degree)

        print(args.dataset)
        if args.splt == True:
            trainset, valset = dealdata1(args, labels)
        else:
            trainset, valset = dealdata(args, labels)

        users = U_train + U_dev + U_test
        class_acc, meanDis, MedianDis, accAT161 = train_regression(maml, labels, features, users, trainset, valset,
                                                                   classLatMedian, classLonMedian, userLocation,
                                                                   args.epochs, args.patience,
                                                                   model_file)
    # load model from file and test the model
    timeStr = time.strftime("%Y-%m-%d %H:%M:%S\t", time.localtime(time.time()))
    argsStr = "-dir:{}\t-{}way{}shot{}query\t-update_lr{}\t-decay:{}".format(
        args.dir, args.n_way, args.k_spt, args.k_qry, args.update_lr, args.weight_decay)
    resultStr = "Test:\tclassification_acc:{}\t\tMean:{}\t\tMedian:{}\t\tAcc@161:{}".format(class_acc, meanDis,
                                                                                            MedianDis, accAT161)
    content = "\n" + timeStr + "\n" + argsStr + "\n" + resultStr + "\n"
    with open('./result_cmu_desce/result.txt', 'a') as f:
        f.write(content)
    f.close()


if __name__ == '__main__':
    # update learning rate and restart
    args = parse_args(sys.argv[1:])
    print(args)

    for i in range(3):
        # 5-1-10-32
        if i == 0:
            args.n_way = 5
            args.k_spt = 5
            args.k_qry = 30
        elif i == 1:
            # args.splt = False
            args.n_way = 5
            args.k_spt = 5
            args.k_qry = 30
        elif i == 2:
            args.n_way = 5
            args.k_spt = 5
            args.k_qry = 30

        print("lr:{}"
              "".format(args.lr), "weight_decay:{}".format(args.weight_decay),
              "update_lr:{}".format(args.update_lr),
              "{}way{}shot{}query".format(
                  args.n_way, args.k_spt, args.k_qry), "task_num:{}".format(
                  args.task_num),
              "splt_len:{}".format(args.splt_len))
        main()


