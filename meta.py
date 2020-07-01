import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
from models import get_model
from learner import Learner
from copy import deepcopy
from haversine import haversine
import math


class Meta(nn.Module):
    """
    Meta Learner
    """
    # def __init__(self, args, config):

    def __init__(self, args, nfeat, nclass, classLatMedian, classLonMedian):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.classLatMedian = classLatMedian
        self.classLonMedian = classLonMedian
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.number_of_training_steps_per_iter = args.number_of_training_steps_per_iter  # 10
        self.multi_step_loss_num_epochs = args.multi_step_loss_num_epochs  # 5
        self.net = get_model(
            args.model,
            args.n_way,
            nfeat,
            nclass,
            usecuda=args.usecuda)
        # self.meta_optim = optim.Adagrad(self.net.parameters(), lr= args.lr, weight_decay=args.weight_decay)
        self.meta_optim = optim.Adam(
            self.net.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def add_loss_weight(self, classLatMedian, classLonMedian):
        mean_lat = np.mean(list(classLatMedian.values()))
        mean_lon = np.mean(list(classLonMedian.values()))
        distances = [None] * 129
        for i, v in classLatMedian.items():
            lat = v
            lon = classLonMedian[i]
            distance = haversine((lat, lon), (mean_lat, mean_lon))
            distances[int(i)] = distance
        dis_mean = np.mean(distances)
        max_num = np.max(distances, axis=0)
        min_num = np.min(distances, axis=0)
        sigma = np.std(distances, axis=0)  # 返回标准偏差
        var_num = np.var(distances, axis=0)  # 返回标准偏差
        print("mean", dis_mean)
        print("var_num", var_num)
        print("sigma", sigma)
        print("max_num-min_num", max_num - min_num)
        # weights =(max_num - distances+1)/(max_num-min_num)#56.5/53.98小->大63
        # weights =(max_num - distances+1)/dis_mean#53.5小->大 63.43
        # weights =(max_num - distances+1)/sigma#53.5小->大 63.43
        weights = (distances - min_num + 1) / sigma  # 53.5小->大 63.43
        # for i in range(0,len(distances)):
        #     if distances[i] <= dis_mean:
        #         weights[i] = (dis_mean + 1 - distances[i])/sigma#55
        # weights[i] = 1#55.67282321899736
        # else:
        #     weights[i] = (max_num - distances[i] + 1) / sigma
        # weights[i] = (distances[i]-dis_mean + 1 )/sigma#

        # weights[i] = math.exp(dis_mean - distances[i])
        print(weights)
        # exit()
        weights = torch.Tensor(weights)
        weights = weights.cuda()
        # return None
        return weights

    def get_per_step_loss_importance_vector(self, epoch):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        生成维数张量(num_inner_loop_steps)，指示每个步骤的目标的重要性向优化方向损失。
        :返回:一个张量，用来计算损失的加权平均，对MSL(多步损耗)机制。
        """
        # "number_of_training_steps_per_iter":5,
        # "multi_step_loss_num_epochs": 10,
        # [0.2 0.2 0.2 0.2 0.2]
        loss_weights = np.ones(shape=(self.number_of_training_steps_per_iter)) * (
            1.0 / self.number_of_training_steps_per_iter)
        '''
        "number_of_training_steps_per_iter":5,
        "multi_step_loss_num_epochs": 10,
        '''
        decay_rate = 1.0 / self.number_of_training_steps_per_iter / \
            self.multi_step_loss_num_epochs  # 0.02
        min_value_for_non_final_losses = 0.03 / \
            self.number_of_training_steps_per_iter  # 0.03???0.006
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(
                loss_weights[i] - (epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] +
            (epoch * (self.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        device = torch.device('cuda')
        loss_weights = torch.Tensor(loss_weights).to(device=device)
        return loss_weights

    def forward(self, epoch, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        x_spt = torch.from_numpy(x_spt).cuda()
        y_spt = torch.from_numpy(y_spt).cuda()
        x_qry = torch.from_numpy(x_qry).cuda()
        y_qry = torch.from_numpy(y_qry).cuda()
        # self.net.train()
        # losses_q = [0 for _ in range(self.update_step)]  # losses_q[i] is the
        # loss on step i
        total_losses = []
        self.meta_optim.zero_grad()
        self.zero_grad()
        corrects = [0 for _ in range(self.update_step)]
        querysz = x_qry.size(1)

        for i in range(self.task_num):
            losses_q = []  # losses_q[i] is the loss on step i
            # x_spt[i]=x_spt[i].cuda()
            # y_spt[i]=y_spt[i].cuda()
            # x_qry[i]=x_qry[i].cuda()
            # y_qry[i]=y_qry[i].cuda()
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector(
                epoch)
            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i])  # torch.Size([15, 129])

            loss = F.cross_entropy(logits, y_spt[i])

            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            for k in range(self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q = self.net(x_qry[i], fast_weights)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                if epoch < self.multi_step_loss_num_epochs:  # 10
                    # logits_q = self.net(x_qry[i], fast_weights)
                    # loss_q = F.cross_entropy(logits_q, y_qry[i])
                    # 用来计算损失的加权平均，对MSL(多步损耗)机制。
                    losses_q.append(
                        per_step_loss_importance_vectors[k] * loss_q)
                else:
                    #  "number_of_training_steps_per_iter":5,
                    if k == (self.number_of_training_steps_per_iter - 1):
                        losses_q.append(loss_q)  # 5
                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(
                        pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k] = corrects[k] + correct
            task_losses = torch.sum(torch.stack(losses_q))
            total_losses.append(task_losses)
        # 评价指标
        # end of all tasks
        # sum over all losses on query set across all tasks
        # loss_q = losses_q[-1] / self.task_num
        # losses = torch.sum(torch.stack(total_losses))/(self.task_num*self.update_step)
        losses = torch.sum(torch.stack(total_losses))
        # optimize theta parameters
        losses.backward()
        self.meta_optim.step()
        losses = torch.sum(torch.stack(total_losses)) / self.task_num
        loss = losses.item()
        del losses
        del task_losses
        del total_losses
        del losses_q
        del loss_q
        accs = np.array(corrects) / (querysz * self.task_num)

        return loss, accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry,
                    spt_locations_one, qry_locations_one, reallabel_one):
        x_spt = torch.from_numpy(x_spt).cuda()
        y_spt = torch.from_numpy(y_spt).cuda()
        x_qry = torch.from_numpy(x_qry).cuda()
        y_qry = torch.from_numpy(y_qry).cuda()
        clc_accs = [0 for _ in range(self.update_step_test + 1)]
        means = [0 for _ in range(self.update_step_test + 1)]
        acc161s = [0 for _ in range(self.update_step_test + 1)]
        medians = [0 for _ in range(self.update_step_test + 1)]
        # finloss = [0 for _ in range(self.update_step_test + 1)]
        querysz = x_qry.size(0)

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt, mode=False)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            #     [setsz, nway]
            logits_q = net(x_qry, net.parameters(), mode=False)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            clc_acc, mean, median, acc161 = self.correct(
                y_qry, logits_q, spt_locations_one, qry_locations_one, reallabel_one)
            # correct = torch.eq(pred_q, y_qry).sum().item()
            clc_accs[0] = clc_accs[0] + clc_acc
            means[0] = means[0] + mean
            medians[0] = medians[0] + median
            acc161s[0] = acc161s[0] + acc161
            # loss = F.cross_entropy(logits_q, y_qry)
            # finloss[0] = finloss[0] + loss.item()

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, mode=False)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            clc_acc, mean, median, acc161 = self.correct(
                y_qry, logits_q, spt_locations_one, qry_locations_one, reallabel_one)
            clc_accs[1] = clc_accs[1] + clc_acc
            means[1] = means[0] + mean
            medians[1] = medians[1] + median
            acc161s[1] = acc161s[1] + acc161
            # loss = F.cross_entropy(logits_q, y_qry)
            # finloss[1] = finloss[1] + loss.item()

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, mode=False)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = net(x_qry, fast_weights, mode=False)
            # loss_q will be overwritten and just keep the loss_q on last
            # update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(
                    pred_q, y_qry).sum().item()  # convert to numpy
                clc_acc, mean, median, acc161 = self.correct(
                    y_qry, logits_q, spt_locations_one, qry_locations_one, reallabel_one)
                clc_accs[k + 1] = clc_accs[k + 1] + clc_acc
                means[k + 1] = means[k + 1] + mean
                medians[k + 1] = medians[k + 1] + median
                acc161s[k + 1] = acc161s[k + 1] + acc161
                # corrects[k + 1] = corrects[k + 1] + correct
                # finloss[k + 1] = finloss[k + 1] + loss_q.item()
        del net
        clc_accs = np.array(clc_accs) / querysz
        means = np.array(means)
        medians = np.array(medians)
        acc161s = np.array(acc161s)
        return clc_accs, means, medians, acc161s

    def correct(self, y_qry, y_pred, spt_locations_one,
                qry_locations_one, reallabel_one):
        # with torch.no_grad():
        #     self.net.eval()
        #     y_pred = self.net(features,mode=False)

        pred_q = F.softmax(y_pred, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
        y_pred = y_pred.data.cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)  # 1代表行

        distances = []
        latlon_pred = []
        latlon_true = []
        for i in range(0, len(y_pred)):
            location = qry_locations_one[i].split(',')
            lat, lon = float(location[0]), float(location[1])
            latlon_true.append([lat, lon])
            prediction = y_pred[i]
            index = reallabel_one[prediction]
            lat_pred, lon_pred = self.classLatMedian[str(
                index)], self.classLonMedian[str(index)]
            latlon_pred.append([lat_pred, lon_pred])
            distance = haversine((lat, lon), (lat_pred, lon_pred))
            distances.append(distance)

        acc_at_161 = 100 * \
            len([d for d in distances if d < 161]) / float(len(distances))
        # return np.mean(distances), np.median(distances), acc_at_161,
        # distances, latlon_true, latlon_pred
        return correct, np.mean(distances), np.median(distances), acc_at_161

    def geo_eval(self, labels, features, U_test,
                 classLatMedian, classLonMedian, userLocation):
        with torch.no_grad():
            self.net.eval()
            y_pred = self.net(features, mode=False)
        pred_q = F.softmax(y_pred, dim=1).argmax(dim=1)
        y_pred = y_pred.data.cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)  # 1代表行

        distances = []
        latlon_pred = []
        latlon_true = []
        class_acc = []
        for i in range(0, len(y_pred)):
            user = U_test[i]
            location = userLocation[user].split(',')
            lat, lon = float(location[0]), float(location[1])
            latlon_true.append([lat, lon])
            prediction = str(y_pred[i])
            lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
            latlon_pred.append([lat_pred, lon_pred])
            distance = haversine((lat, lon), (lat_pred, lon_pred))
            distances.append(distance)
            correct = torch.eq(
                pred_q[i], labels[i]).sum().item()  # convert to numpy
            class_acc.append(correct)

        acc_at_161 = 100 * \
            len([d for d in distances if d < 161]) / float(len(distances))
        # return np.mean(distances), np.median(distances), acc_at_161,
        # distances, latlon_true, latlon_pred
        return np.mean(class_acc), np.mean(
            distances), np.median(distances), acc_at_161

    def statedict(self, model_file):
        torch.save(self.net.state_dict(), model_file)  # ？？

    def load_state_dict(self, model_file):
        self.net.load_state_dict(torch.load(model_file))

    def xttrain(self):
        self.net.train()


def main():
    pass


if __name__ == '__main__':
    main()
