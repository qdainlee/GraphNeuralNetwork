import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GCNConv(nn.Module):
    ''''Graph Convolutional Network, H = AXW'''
    def __init__(self, in_dim, out_dim, bias=True):
        super(GCNConv, self).__init__()
        
        self.w = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, adj):
        x = self.w(x)
        x = torch.matmul(adj, x)
        return x


class GINConv(nn.Module):
    '''Graph Isomorphism Network (without self loop epsilon), H = MLP(AX)'''
    def __init__(self, in_dim, hid_dim, num_layer, bias=True):
        super(GINConv, self).__init__()
        self.num_layer = num_layer

        self.MLP = nn.ModuleList()
        for i in range(num_layer):
            self.MLP.append(nn.Linear(in_dim if i==0 else hid_dim,
                                      hid_dim,
                                      bias=bias))
        
    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        for i, layer in enumerate(self.MLP):
            x = layer(x)
            if i != self.num_layer-1: 
                x = F.relu(x)
        return x


class GATConv(nn.Module):
    '''Multi-head Graph Attentional Network'''
    def __init__(self, in_dim, out_dim, n_head):
        super(GATConv, self).__init__()

        self.n_head = n_head

        self.ws = nn.ModuleList()
        self.attentions = nn.ParameterList()
        for i in range(n_head):
            self.ws.append(nn.Linear(in_dim, out_dim, bias=bias))
            self.attentions.append(nn.Parameter(torch.rand(size=(out_dim, out_dim))))

        self.a = nn.Linear(out_dim * n_head, out_dim, bias=False)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.gate1 = nn.Linear(out_dim, out_dim, bias=False)
        self.gate2 = nn.Linear(out_dim, out_dim, bias=False)
        self.gatebias = nn.Parameter(torch.tand(size=(out_dim,)))

    def forward(self, x, adj):
        xs = []
        for i in range(self.n_head):
            _x = self.ws[i](x)
            a = self.attention_matrix(_x, adj, self.attentions[i])
            _x = torch.matmul(a, _x)
            _x = F.relu(_x)
            xs.append(_x)
        
        _x = F.relu(torch.cat(xs, 2))
        _x = self.a(_x)

        x = self.fc(x)

        n = x.size(1)
        coeff = torch.sigmoid(self.gate1(x) + self.gate2(_x)) + \
                self.gatebias.repeat(1, n).reshape(n, -1)
        x = torch.mul(x, coeff) + torch.mul(_x, 1.-coeff)
        return x

    @staticmethod
    def attention_matrix(x, adj, attention):
        x1 = torch.einsum('ij,ajk->aik', attention, torch.transpose(x, 1, 2))
        x2 = torch.matmul(x, x1)
        adj = torch.mul(adj, x1)
        adj = torch.tanh(adj)
        return adj


class Readout(nn.Module):
    def __init__(self, in_dim, out_dim, method='sum'):
        super(Readout, self).__init__()
        if method == 'sum':
            self.readout = torch.sum
        elif method == 'mean':
            self.readout = torch.mean

        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.fc(x)
        x = self.readout(x, dim=1)
        return x


class Readout3M(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Readout3M, self).__init__()
        
        self.fc = nn.Linear(in_dim, out_dim * 3, bias=False)

    def forward(self, x):
        x = self.fc(x)

        x_sum = torch.sum(x, dim=1)
        x_mean = torch.mean(x, dim=1)
        x_max = torch.max(x, dim=1).values

        x = torch.cat(x, dim=1)
        return x