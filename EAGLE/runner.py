from EAGLE.utils.mutils import *
from EAGLE.utils.inits import prepare
from EAGLE.utils.loss import EnvLoss
from EAGLE.utils.util import init_logger, logger
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm

import os
import sys
import time
import torch
import numpy as np
import torch.optim as optim
import pandas as pd
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class NeibSampler:
    def __init__(self, graph, nb_size, include_self=False):
        n = graph.number_of_nodes()
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        if include_self:
            nb_all = torch.zeros(n, nb_size + 1, dtype=torch.int64)
            nb_all[:, 0] = torch.arange(0, n)
            nb = nb_all[:, 1:]
        else:
            nb_all = torch.zeros(n, nb_size, dtype=torch.int64)
            nb = nb_all
        popkids = []
        for v in range(n):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self):
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all


class Runner(object):
    def __init__(self, args, model, cvae, data, writer=None, **kwargs):
        seed_everything(args.seed)
        self.args = args
        self.data = data
        self.model = model
        self.cvae = cvae
        self.writer = writer
        self.len = len(data['train']['edge_index_list'])
        self.len_train = self.len - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength
        self.nbsz = args.nbsz
        self.n_factors = args.n_factors
        self.delta_d = args.delta_d
        self.d = self.n_factors * self.delta_d
        self.interv_size_ratio = args.interv_size_ratio

        x = data['x'].to(args.device).clone().detach()
        self.x = [x for _ in range(self.len)] if len(x.shape) <= 2 else x
        self.edge_index_list_pre = [data['train']['edge_index_list'][ix].long().to(args.device) for ix in
                                    range(self.len)]
        neighbors_all = []
        for t in range(self.len):
            graph_data = Data(x=self.x[t], edge_index=self.edge_index_list_pre[t])
            graph = to_networkx(graph_data)
            sampler = NeibSampler(graph, self.nbsz)
            neighbors = sampler.sample().to(args.device)
            neighbors_all.append(neighbors)
        self.neighbors_all = torch.stack(neighbors_all).to(args.device)

        self.loss = EnvLoss(args)
        print('total length: {}, test length: {}'.format(self.len, args.testlength))

    def cal_fact_rank(self, x_all):
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        x_all = x_all.view(times, n, k, delta_d)
        x_all_trans = x_all.permute(1, 2, 0, 3)
        points = torch.var(x_all_trans, dim=[2, 3])
        rank = torch.argsort(points, 0, descending=True)
        return rank

    def cal_fact_var(self, x_all):
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        x_all_trans = x_all.permute(1, 2, 0, 3)
        points = torch.var(x_all_trans, dim=[2, 3]).view(n, k)
        return points

    def intervention(self, x_all):
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        m = int(self.interv_size_ratio * n)
        indices = torch.randperm(n)[:m]
        x_m = x_all[:, indices, :].view(times, m, k, delta_d)
        mask = self.cal_mask(x_m)
        mask_expand = torch.repeat_interleave(mask, delta_d, dim=1).view(m, k * delta_d)
        mask_expand = torch.stack([mask_expand] * times)

        saved_env = self.saved_env(x_m, mask)
        sampled_env = self.gen_env(saved_env)

        x_m = x_m.view(times, m, k * delta_d)
        embeddings_interv = x_m * mask_expand + sampled_env * (1 - mask_expand)
        x_all[:, indices, :] = embeddings_interv.to(torch.float32)
        return x_all, indices

    def intervention_faster(self, x_all):
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        x_all = x_all.view(times, n, k, delta_d)
        mask = self.cal_mask_faster(x_all)
        mask_expand = torch.repeat_interleave(mask, delta_d, dim=1).view(n, k * delta_d)
        mask_expand = torch.stack([mask_expand] * times)

        saved_env = self.saved_env(x_all, mask)
        sampled_env = self.gen_env(saved_env)

        x_all = x_all.view(times, n, k * delta_d)
        embeddings_interv = x_all * mask_expand + sampled_env * (1 - mask_expand)
        embeddings_interv = embeddings_interv.to(torch.float32)
        return embeddings_interv

    def intervention_final(self, x_all_original):
        x_all = x_all_original.clone()
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        m = int(self.interv_size_ratio * n)
        indices = torch.randperm(n)[:m]
        x_m = x_all[:, indices, :].view(times, m, k, delta_d)
        mask = self.cal_mask_faster(x_m)
        mask_expand = torch.repeat_interleave(mask, delta_d, dim=1).view(m, k * delta_d)
        mask_expand = torch.stack([mask_expand] * times)

        saved_env = self.saved_env(x_m, mask)
        sampled_env = self.gen_env(saved_env)

        x_m = x_m.view(times, m, k * delta_d)
        embeddings_interv = x_m * mask_expand + sampled_env * (1 - mask_expand)
        x_all[:, indices, :] = embeddings_interv.to(torch.float32)
        return x_all, indices

    def cal_mask(self, x_m):
        times = len(x_m)
        m = len(x_m[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        var = self.cal_fact_var(x_m).cpu().detach().numpy()

        def split_array(arr):
            n = len(arr)
            total_sum = sum(arr)
            dp = [[False for _ in range(total_sum + 1)] for __ in range(n + 1)]
            dp[0][0] = True
            for i in range(1, n + 1):
                for j in range(total_sum + 1):
                    if j >= arr[i - 1]:
                        dp[i][j] = dp[i - 1][j] or dp[i - 1][j - arr[i - 1]]
                    else:
                        dp[i][j] = dp[i - 1][j]
            min_diff = float('inf')
            for j in range(total_sum // 2, -1, -1):
                if dp[n][j]:
                    min_diff = total_sum - 2 * j
                    break
            return min_diff

        def process_matrix(matrix):
            matrix = adjust_matrix(matrix)
            n, k = matrix.shape
            result = np.zeros((n, k))
            for i in range(n):
                row = matrix[i]
                min_diff = split_array(row)
                avg = sum(row) / k
                for j in range(k):
                    if row[j] <= avg - min_diff / 2:
                        result[i][j] = 1
                    else:
                        result[i][j] = 0
                if np.sum(result[i]) == 0:
                    index = np.argmin(result[i])
                    result[i][index] = 1
            return result

        def adjust_matrix(matrix):
            for row in matrix:
                while np.min(row) < 1:
                    row *= 10
            matrix_min = matrix.min(axis=1).astype(int)
            matrix_min = np.expand_dims(matrix_min, axis=1)
            matrix_min = np.tile(matrix_min, (1, len(matrix[1])))
            matrix = matrix - matrix_min

            for row in matrix:
                while np.min(row) < 1:
                    row *= 10

            return matrix.astype(int)

        mask = process_matrix(var)
        return torch.from_numpy(mask).to(self.args.device)

    def cal_mask_faster(self, x_m):
        times = len(x_m)
        m = len(x_m[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        var = self.cal_fact_var(x_m)

        def max_avg_diff_index(sorted_tensor):
            n, k = sorted_tensor.shape
            result = np.zeros(n)
            for i in range(n):
                row = sorted_tensor[i]
                max_diff = 0
                max_index = 0
                for j in range(1, k):
                    avg1 = sum(row[:j]) / j
                    avg2 = sum(row[j:]) / (k - j)
                    diff = abs(avg1 - avg2)
                    if diff <= max_diff:
                        break
                    if diff > max_diff:
                        max_diff = diff
                        max_index = j
                result[i] = max_index - 1
            return result

        var_sorted = torch.sort(var, dim=1)
        var_sorted_index = var_sorted.indices
        indices = max_avg_diff_index(var_sorted.values).astype(int)
        for i in range(var.shape[0]):
            sort_indices = var_sorted_index[i]
            values = var[i, sort_indices]
            mask = torch.zeros_like(values)
            mask[:indices[i] + 1] = 1
            var[i, sort_indices] = mask

        return var

    def saved_env(self, x_all, mask):
        times = len(x_all)
        n = len(x_all[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        m = mask.shape[0]
        x_all = x_all.view(times, n, d)
        mask_env = 1 - mask
        mask_expand = torch.repeat_interleave(mask_env, delta_d, dim=1).view(m, k * delta_d)
        mask_expand = torch.stack([mask_expand] * times)

        extract_env = (x_all * mask_expand).view(times, n, k, delta_d).permute(2, 0, 1, 3)
        extract_env = extract_env.view(k, times * n, delta_d)
        extract_env = extract_env[:, torch.randperm(times * n), :]
        for i in range(k):
            zero_rows = (extract_env[i].sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            non_zero_rows = (extract_env[i].sum(dim=1) != 0).nonzero(as_tuple=True)[0]
            if len(non_zero_rows) > 0:
                replacement_rows = non_zero_rows[torch.randint(0, len(non_zero_rows), (len(zero_rows),))]
                extract_env[i][zero_rows] = extract_env[i][replacement_rows]

        return extract_env.view(times, n, d)

    def gen_env(self, extract_env):
        times = len(extract_env)
        n = len(extract_env[0])
        k = self.n_factors
        d = self.d
        delta_d = self.delta_d
        n_gen = int(self.args.gen_ratio * n)

        z = torch.randn(n_gen * k, self.args.d_for_cvae).to(self.args.device)
        y = torch.ones(n_gen, k)
        for i in range(k):
            y[:, i:i + 1] = y[:, i:i + 1] * i
        y_T = y.transpose(0, 1)
        y = (F.one_hot(y_T.reshape(-1).to(torch.int64))).to(self.args.device)
        gen_env = self.cvae.decode(z, y).view(n_gen, k * delta_d)

        random_indices = torch.randperm(n)[:n_gen]
        extract_env[:, random_indices] = gen_env
        return extract_env.view(times, n, d)

    def loss_cvae(self, recon, x, mu, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * (1 + 2 * log_std - mu.pow(2) - torch.exp(2 * log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss

    def train(self, epoch, data):
        args = self.args
        self.model.train()
        optimizer = self.optimizer

        embeddings = self.model([data['edge_index_list'][ix].long().to(args.device) for ix in range(self.len)], self.x,
                                self.neighbors_all)

        y = torch.ones(len(embeddings[0]) * len(embeddings), args.n_factors)
        for i in range(args.n_factors):
            y[:, i:i + 1] = y[:, i:i + 1] * i
        y_T = y.transpose(0, 1)
        y = (F.one_hot(y_T.reshape(-1).to(torch.int64))).to(args.device)

        embeddings_view = embeddings.view(len(embeddings), len(embeddings[0]), args.n_factors, args.delta_d)
        embeddings_trans = embeddings_view.permute(2, 0, 1, 3)
        x_flatten = torch.flatten(embeddings_trans, start_dim=0, end_dim=2)
        recon, mu, log_std = self.cvae(x_flatten, y)
        cvae_loss = self.loss_cvae(recon, x_flatten, mu, log_std) / (len(embeddings[0] * len(embeddings[0])))

        device = embeddings[0].device

        val_auc_list = []
        test_auc_list = []
        train_auc_list = []
        for t in range(self.len - 1):
            z = embeddings[t]
            _, pos_edge, neg_edge = prepare(data, t + 1)[:3]
            auc, ap = self.loss.predict(z, pos_edge, neg_edge, self.model.edge_decoder)
            if t < self.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)

        edge_index = []
        pos_edge_index_all = []
        neg_edge_index_all = []
        edge_label = []
        tsize = []
        for t in range(self.len_train - 1):
            z = embeddings[t]
            pos_edge_index = prepare(data, t + 1)[0]
            if args.dataset == 'yelp':
                neg_edge_index = bi_negative_sampling(pos_edge_index, args.num_nodes, args.shift)
            else:
                neg_edge_index = negative_sampling(pos_edge_index,
                                                   num_neg_samples=pos_edge_index.size(1) * args.sampling_times)
            edge_index.append(torch.cat([pos_edge_index, neg_edge_index], dim=-1))
            pos_edge_index_all.append(pos_edge_index)
            neg_edge_index_all.append(neg_edge_index)
            pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
            neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
            edge_label.append(torch.cat([pos_y, neg_y], dim=0))
            tsize.append(pos_edge_index.shape[1] * 2)

        edge_label = torch.cat(edge_label, dim=0)

        def generate_edge_index_interv(pos_edge_index_all, neg_edge_index_all, indices):
            edge_label = []
            edge_index = []
            pos_edge_index_interv = pos_edge_index_all.copy()
            neg_edge_index_interv = neg_edge_index_all.copy()
            index = indices.cpu().numpy()
            for t in range(self.len_train - 1):
                mask_pos = np.logical_and(np.isin(pos_edge_index_interv[t].cpu()[0], index),
                                          np.isin(pos_edge_index_interv[t].cpu()[1], index))
                pos_edge_index = pos_edge_index_interv[t][:, mask_pos]
                mask_neg = np.logical_and(np.isin(neg_edge_index_interv[t].cpu()[0], index),
                                          np.isin(neg_edge_index_interv[t].cpu()[1], index))
                neg_edge_index = neg_edge_index_interv[t][:, mask_neg]
                pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
                neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
                edge_label.append(torch.cat([pos_y, neg_y], dim=0))
                edge_index.append(torch.cat([pos_edge_index, neg_edge_index], dim=-1))
            edge_label = torch.cat(edge_label, dim=0)
            return edge_label, edge_index

        def cal_y(embeddings, decoder):
            preds = torch.tensor([]).to(device)
            for t in range(self.len_train - 1):
                z = embeddings[t]
                pred = decoder(z, edge_index[t])
                preds = torch.cat([preds, pred])
            return preds

        def cal_y_interv(embeddings, decoder, edge_index_interv, indices):
            index = indices.cpu().numpy()
            preds = torch.tensor([]).to(device)
            for t in range(self.len_train - 1):
                z = embeddings[t]
                pred = decoder(z, edge_index_interv[t])
                preds = torch.cat([preds, pred])
            return preds

        criterion = torch.nn.BCELoss()
        criterion_var = torch.nn.MSELoss()

        def cal_loss(y, label):
            return criterion(y, label)

        def cal_loss_var(y, label):
            return criterion_var(y, label)

        pred_y = cal_y(embeddings, self.model.edge_decoder)
        main_loss = cal_loss(pred_y, edge_label)

        intervention_times = args.n_intervene
        env_loss = torch.tensor([]).to(device)
        for i in range(intervention_times):
            embeddings_interv, indices = self.intervention_final(embeddings)
            edge_label_interv, edge_index_interv = generate_edge_index_interv(pos_edge_index_all, neg_edge_index_all,
                                                                              indices)
            pred_y_interv = cal_y_interv(embeddings_interv, self.model.edge_decoder, edge_index_interv, indices)
            env_loss = torch.cat([env_loss, cal_loss(pred_y_interv, edge_label_interv).unsqueeze(0)])

        var_loss = torch.var(env_loss)

        alpha = args.alpha
        beta = args.beta

        if epoch % args.every_epoch == 0:
            loss = main_loss + alpha * var_loss + beta * cvae_loss
        else:
            loss = main_loss + alpha * var_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        average_epoch_loss = loss.item()

        return average_epoch_loss, train_auc_list, val_auc_list, test_auc_list

    def run(self):
        args = self.args
        min_epoch = args.min_epoch
        max_patience = args.patience
        patience = 0
        self.optimizer = optim.Adam([p for n, p in self.model.named_parameters() if 'ss' not in n], lr=args.lr,
                                    weight_decay=args.weight_decay)

        t_total0 = time.time()
        max_auc = 0
        max_test_auc = 0
        max_train_auc = 0

        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                t0 = time.time()
                epoch_losses, train_auc_list, val_auc_list, test_auc_list = self.train(
                    epoch, self.data['train'])
                average_epoch_loss = epoch_losses
                average_train_auc = np.mean(train_auc_list)

                average_val_auc = np.mean(val_auc_list)
                average_test_auc = np.mean(test_auc_list)

                if average_val_auc > max_auc:
                    max_auc = average_val_auc
                    max_test_auc = average_test_auc
                    max_train_auc = average_train_auc

                    test_results = self.test(epoch, self.data['test'])

                    metrics = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc".split(',')
                    measure_dict = dict(
                        zip(metrics,
                            [max_train_auc, max_auc, max_test_auc] + test_results))

                    patience = 0

                    filepath = '../checkpoint/' + self.args.dataset + '.pth'
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'cvae_state_dict': self.cvae.state_dict()}, filepath)

                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break
                if epoch == 1 or epoch % self.args.log_interval == 0:
                    print("Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(epoch, average_epoch_loss, time.time() - t0))
                    print(
                        f"Current: Epoch:{epoch}, Train AUC:{average_train_auc:.4f}, Val AUC: {average_val_auc:.4f}, Test AUC: {average_test_auc:.4f}"
                    )

                    print(
                        f"Train: Epoch:{test_results[0]}, Train AUC:{max_train_auc:.4f}, Val AUC: {max_auc:.4f}, Test AUC: {max_test_auc:.4f}"
                    )
                    print(
                        f"Test: Epoch:{test_results[0]}, Train AUC:{test_results[1]:.4f}, Val AUC: {test_results[2]:.4f}, Test AUC: {test_results[3]:.4f}"
                    )

        epoch_time = (time.time() - t_total0) / (epoch - 1)
        metrics = [max_train_auc, max_auc, max_test_auc
                   ] + test_results + [epoch_time]
        metrics_des = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc,epoch_time".split(',')
        metrics_dict = dict(zip(metrics_des, metrics))
        df = pd.DataFrame([metrics], columns=metrics_des)
        print(df)
        return metrics_dict

    def test(self, epoch, data):
        args = self.args

        train_auc_list = []

        val_auc_list = []
        test_auc_list = []
        self.model.eval()
        embeddings = self.model([data['edge_index_list'][ix].long().to(args.device) for ix in range(self.len)], self.x,
                                self.neighbors_all)

        for t in range(self.len - 1):
            z = embeddings[t]
            edge_index, pos_edge, neg_edge = prepare(data, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            auc, ap = self.loss.predict(z, pos_edge, neg_edge,
                                        self.model.edge_decoder)
            if t < self.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)

        return [
            epoch,
            np.mean(train_auc_list),
            np.mean(val_auc_list),
            np.mean(test_auc_list)
        ]

    def re_run(self):
        args = self.args
        data = self.data['test']

        self.optimizer = optim.Adam(
            [p for n, p in self.model.named_parameters() if 'ss' not in n],
            lr=args.lr,
            weight_decay=args.weight_decay)

        filepath = '../saved_model/' + self.args.dataset + '.pth'
        checkpoint = torch.load(filepath)

        self.cvae.load_state_dict(checkpoint['cvae_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])

        train_auc_list = []

        val_auc_list = []
        test_auc_list = []
        self.model.eval()
        embeddings = self.model([data['edge_index_list'][ix].long().to(args.device) for ix in range(self.len)], self.x,
                                self.neighbors_all)

        for t in range(self.len - 1):
            z = embeddings[t]
            edge_index, pos_edge, neg_edge = prepare(data, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            auc, ap = self.loss.predict(z, pos_edge, neg_edge,
                                        self.model.edge_decoder)
            if t < self.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)

        test_res = [
            0,
            np.mean(train_auc_list),
            np.mean(val_auc_list),
            np.mean(test_auc_list)
        ]

        metrics = "epoch,test_train_auc,test_val_auc,test_test_auc".split(',')
        metrics_dict = dict(zip(metrics, test_res))
        df = pd.DataFrame([test_res], columns=metrics)
        print(df)
        return metrics_dict
