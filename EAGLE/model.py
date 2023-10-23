from torch_geometric.nn.inits import glorot
from torch import nn
from torch.nn import Parameter
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_scatter import scatter

import torch.nn.functional as F
import networkx as nx
import numpy as np
import torch
import math


class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid, max_len=50, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()

        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.emb(t))


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, z, e):
        x_i = z[e[0]]
        x_j = z[e[1]]
        x = torch.cat([x_i, x_j], dim=1)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x).squeeze()


class MultiplyPredictor(torch.nn.Module):
    def __init__(self):
        super(MultiplyPredictor, self).__init__()

    def forward(self, z, e):
        x_i = z[e[0]]
        x_j = z[e[1]]
        x = (x_i * x_j).sum(dim=1)
        return torch.sigmoid(x)


class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()

        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.mm(x, self.weight) + self.bias


class EAConv(nn.Module):
    def __init__(self, dim, n_factors, agg_param, use_RTE=False):
        super(EAConv, self).__init__()

        assert dim % n_factors == 0
        self.d = dim
        self.k = n_factors
        self.delta_d = self.d // self.k
        self.dk = self.d - self.delta_d
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_dk = torch.zeros(1, self.dk)
        self._cache_zero_k = torch.zeros(1, self.k)
        self._cache_zero_kk = torch.zeros(1, self.k - 1)
        self.use_RTE = use_RTE
        self.rte = RelTemporalEncoding(self.d)
        self.agg_param = agg_param

    def time_encoding(self, x_all):
        if self.use_RTE:
            times = len(x_all)
            for t in range(times):
                x_all[t] = self.rte(x_all[t], torch.LongTensor([t]).to(x_all[t].device))
        return x_all

    def aggregate_former(self, x, neighbors, max_iter):
        n, m = x.size(0), neighbors.size(0) // x.size(0)
        d, k, delta_d = self.dk, self.k - 1, self.delta_d
        x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = torch.cat([x, self._cache_zero_dk], dim=0)
        z = z[neighbors].view(n, m, k, delta_d)
        u = None
        for clus_iter in range(max_iter):
            if u is None:
                p = self._cache_zero_kk.expand(n * m, k).view(n, m, k)
            else:
                p = torch.sum(z * u.view(n, 1, k, delta_d), dim=3)
            p = F.softmax(p, dim=2)
            u = torch.sum(z * p.view(n, m, k, 1), dim=1)
            u += x.view(n, k, delta_d)
            if clus_iter < max_iter - 1:
                u = F.normalize(u, dim=2)
        u = u.view(n, k * delta_d)
        return u

    def aggregate_former_v2(self, x, neighbors, max_iter):
        n, m = x.size(0), neighbors.size(0) // x.size(0)
        d, k, delta_d = self.d, self.k, self.delta_d
        x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = torch.cat([x, self._cache_zero_d], dim=0)
        z = z[neighbors].view(n, m, k, delta_d)
        u = None
        for clus_iter in range(max_iter):
            if u is None:
                p = self._cache_zero_k.expand(n * m, k).view(n, m, k)
            else:
                p = torch.sum(z * u.view(n, 1, k, delta_d), dim=3)
            p = F.softmax(p, dim=2)
            u = torch.sum(z * p.view(n, m, k, 1), dim=1)
            u += x.view(n, k, delta_d)
            if clus_iter < max_iter - 1:
                u = F.normalize(u, dim=2)
        u = u.view(n, k * delta_d)
        return u

    def aggregate_lastt(self, x0, neighbors_previous, t, m):
        d, k, delta_d = self.d, self.k, self.delta_d
        n = len(x0)
        x0 = F.normalize(x0.view(n, k, delta_d), dim=2).view(n, d)
        fac_t = x0.view(n, k, delta_d)
        for t0 in range(t):
            z0 = torch.cat([x0, self._cache_zero_d], dim=0)
            z0 = z0[neighbors_previous[t0]].view(n, m, k, delta_d)
            u0 = None
            p0 = self._cache_zero_k.expand(n * m, k).view(n, m, k)
            p0 = F.softmax(p0, dim=2)
            u0 = torch.sum(z0 * p0.view(n, m, k, 1), dim=1)
            fac_t += u0
        fac_t = fac_t.view(n, d)
        layer = nn.Linear(d, delta_d).to(x0[0].device)
        fac_t_emb = F.sigmoid(layer(fac_t))
        return fac_t_emb

    def aggregate_lastt_v2(self, x_all, t):
        x_all_toagg = torch.sum(x_all[: t + 1])
        layer = nn.Linear(self.d, self.delta_d).to(x_all[0].device)
        fac_t_emb = F.sigmoid(layer(x_all_toagg))
        return fac_t_emb

    def aggregate_lastt_v2_weighted(self, x_all, t):
        weights = F.sigmoid(torch.tensor(list(range(t))).to(x_all[0].device))
        x_all_toagg = torch.sum(weights * x_all[:t]) * self.agg_param + x_all[t]
        layer = nn.Linear(self.d, self.delta_d).to(x_all[0].device)
        fac_t_emb = F.sigmoid(layer(x_all_toagg))
        return fac_t_emb

    def forward(self, x_all, neighbors_all, max_iter):
        dev = x_all[0].device
        n = len(x_all[0])
        m = len(neighbors_all[0][0])
        self._cache_zero_d = self._cache_zero_d.to(dev)
        self._cache_zero_dk = self._cache_zero_dk.to(dev)
        self._cache_zero_k = self._cache_zero_k.to(dev)
        self._cache_zero_kk = self._cache_zero_kk.to(dev)
        times = len(x_all)  # nums of time slices
        emb = torch.zeros((times, x_all[0].size(0), self.d)).to(dev)

        for t in range(times):
            x_temp = self.aggregate_former_v2(
                x_all[t], neighbors_all[t].view(-1), max_iter
            )
            if t > 0:
                weights = F.sigmoid(
                    torch.tensor(list(range(t))).view(t, 1, 1).to(x_all[0].device)
                )
                emb[t] = (
                    torch.sum(weights * emb[:t], dim=0) / t
                ) * self.agg_param + x_temp * (1 - self.agg_param)
            else:
                emb[t] = x_temp
            emb[t] = emb[t].view(n, self.d)

        return emb.to(dev)


class EADGNN(nn.Module):
    def __init__(self, args=None):
        super(EADGNN, self).__init__()

        self.args = args
        self.n_layers = args.n_layers
        self.n_factors = args.n_factors
        self.delta_d = args.delta_d
        self.in_dim = args.nfeat
        self.hid_dim = self.n_factors * self.delta_d
        self.norm = args.norm
        self.maxiter = args.maxiter
        self.use_RTE = args.use_RTE
        self.agg_param = args.agg_param
        self.feat = Parameter(
            (torch.ones(args.num_nodes, args.nfeat)).to(args.device), requires_grad=True
        )
        self.linear = SparseInputLinear(self.in_dim, self.hid_dim)
        self.layers = nn.ModuleList(
            EAConv(self.hid_dim, self.n_factors, self.agg_param, self.use_RTE)
            for i in range(self.n_layers)
        )
        self.relu = F.relu
        self.LeakyReLU = nn.LeakyReLU()
        self.dropout = args.dropout
        self.reset_parameter()
        self.device = args.device
        self.edge_decoder = MultiplyPredictor()

    def reset_parameter(self):
        glorot(self.feat)

    def forward(self, edge_index_list, x_list, neighbors_all):
        times = len(edge_index_list)
        if x_list is None:
            x_list = [self.linear(self.feat) for i in range(len(edge_index_list))]
        else:
            x_list = [self.linear(x) for x in x_list]

        for i, layer in enumerate(self.layers):
            x_list = layer(x_list, neighbors_all, self.maxiter)
            if i != len(self.layers) - 1:
                x_list = x_list.view(
                    len(x_list), len(x_list[0]), self.n_factors, self.delta_d
                )
                x_list = self.LeakyReLU(x_list.to(self.device))
                x_list = [
                    F.dropout(
                        input=F.normalize(x, dim=2),
                        p=self.dropout,
                        training=self.training,
                    )
                    for x in x_list
                ]

        return x_list


class ECVAE(nn.Module):
    def __init__(self, args=None):
        super(ECVAE, self).__init__()

        delta_d = args.delta_d
        n_factors = args.n_factors
        latent_size = args.d_for_cvae
        self.fc1_mu = nn.Linear(delta_d + n_factors, latent_size)
        self.fc1_log_std = nn.Linear(delta_d + n_factors, latent_size)
        self.fc2 = nn.Linear(latent_size + n_factors, delta_d)

    def encode(self, x, y):
        h1 = F.relu(torch.cat([x, y], dim=1))
        mu = self.fc1_mu(h1)
        log_std = self.fc1_log_std(h1)
        return mu, log_std

    def decode(self, z, y):
        h3 = F.relu(torch.cat([z, y], dim=1))
        recon = self.fc2(h3)
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, y):
        mu, log_std = self.encode(x, y)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z, y)
        return recon, mu, log_std
