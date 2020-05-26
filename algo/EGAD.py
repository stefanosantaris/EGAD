import torch
from torch.nn.modules.module import  Module
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F


class EGAD(Module):
    def __init__(self, num_cells, layer_1_in, layer_1_out, layer_2_out, n_heads, dropout, alpha):
        super(EGAD, self).__init__()

        self.num_cells = num_cells

        self.weight_l1_init = Parameter(torch.FloatTensor(layer_1_in, layer_1_out), requires_grad=True)
        nn.init.xavier_uniform_(self.weight_l1_init)

        self.gcn_cells = [GCNCell(layer_1_out, layer_2_out, dropout, alpha, n_heads) for _ in range(self.num_cells)]
        for i, gcn in enumerate(self.gcn_cells):
            self.add_module('gcn_cell_{}'.format(i), gcn)

    def forward(self, input, adj_norm):
        layer_1_weights = self.weight_l1_init
        for i,gcn in enumerate(self.gcn_cells):
            layer_1_weights, h2 = gcn(input[i], adj_norm[i], layer_1_weights)
        return h2


class GCNCell(Module):
    def __init__(self, layer_1_out, layer_2_out, dropout, alpha, n_heads):
        super(GCNCell, self).__init__()

        self.attention_l1 = [GraphAttentionLayer(layer_1_out, layer_1_out, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attention_l1):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att_l1 = GraphAttentionLayer(layer_1_out * n_heads, layer_1_out, dropout=dropout, alpha=alpha, concat=False)

        self.l2_weight = Parameter(torch.FloatTensor(layer_1_out, layer_2_out), requires_grad=True)
        nn.init.xavier_uniform_(self.l2_weight)


    def forward(self, input, adj_norm, previous_l1):
        x_1 = torch.cat([att(previous_l1, adj_norm) for att in self.attention_l1], dim=1)
        x_1 = F.elu(self.out_att_l1(x_1, adj_norm))

        h1 = F.relu(torch.spmm(adj_norm, torch.mm(input, x_1)))
        h2 = torch.spmm(adj_norm, torch.mm(h1, self.l2_weight))

        return x_1, h2


class GraphAttentionLayer(Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.H = nn.Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
        nn.init.xavier_uniform_(self.H)
        self.a = Parameter(torch.FloatTensor(2 * out_features, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.a)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.H)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1,
                                                                                          2 * self.out_features)
        e_lin = torch.matmul(a_input, self.a).squeeze(2)
        e = self.leakyrelu(e_lin)

        attention = torch.spmm(adj, e)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime