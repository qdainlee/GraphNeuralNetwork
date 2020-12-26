import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCNConv, GINConv, GATConv, Readout, Readout3M


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args

        self.gnns = nn.ModuleList()
        for i in range(args.num_gnn):
            self.gnns.append(GCNConv(args.dim_af if i == 0 else args.dim_gnn,
                                     args.dim_gnn))

        self.readout = Readout(args.dim_gnn, args.dim_fc, 'sum')

        self.fcs = nn.ModuleList()
        for i in range(args.num_fc):
            self.fcs.append(nn.Linear(args.dim_fc,
                                      args.dim_fc if i != args.num_fc-1 else 1))

    def forward(self, data):
        x, adj = data['x'], data['adj']
        
        for layer in self.gnns:
            x = F.relu(layer(x, adj))
        
        x = self.readout(x)

        for i, layer in enumerate(self.fcs):
            x = F.dropout(x, self.args.dropout, training=self.training)
            x = layer(x)
            if i != self.args.num_fc - 1:
                x = F.relu(x)
    
        return torch.sigmoid(x)


