# import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_geometric.nn as gnn
from rgcn import RGCNLayer


class DQN(nn.Module):
    def __init__(self, node_dim=3, edge_dim=2, action_dim=5):
        super(DQN, self).__init__()
        # node encoder
        self.n_enc = nn.Linear(node_dim, 64)
        # edge encoder
        self.e_enc = nn.Linear(edge_dim, 32)

        self.conv1 = RGCNLayer(64, 64, 2, 32)
        self.conv2 = gnn.RGCNConv(64, 64, 2, aggr='max')

        self.v = nn.Linear(64, 1)
        self.a = nn.Linear(64, action_dim)

    def forward(self, x, edge_index, edge_attr, edge_type):

        n = f.relu(self.n_enc(x))
        e = f.relu(self.e_enc(edge_attr))

        h = f.relu(self.conv1(n, edge_index, edge_type, e))
        h = f.relu(self.conv2(h, edge_index, edge_type))

        # Apply a final (linear) classifier.
        value = self.v(h)
        action = self.a(h)

        return value, action
