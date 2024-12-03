import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_geometric.nn as gnn
from rgcn import RGCNLayer


class Actor(nn.Module):
    def __init__(self, node_dim=3, edge_dim=2, action_dim=1):
        super(Actor, self).__init__()
        # node encoder
        self.n_enc = nn.Linear(node_dim, 64)
        # edge encoder
        self.e_enc = nn.Linear(edge_dim, 32)
        # omega encoder
        self.o_enc = nn.Linear(2, 64)

        self.conv1 = RGCNLayer(64, 64, 8, 32)
        self.conv2 = gnn.RGCNConv(64, 64, 8, aggr='max')
        self.agg = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, action_dim)

    def forward(self, x, edge_index, edge_attr, edge_type, omega):

        n = f.relu(self.n_enc(x))
        e = f.relu(self.e_enc(edge_attr))
        o = f.relu(self.o_enc(omega))

        h = f.relu(self.conv1(n, edge_index, edge_type, e))
        h = f.relu(self.conv2(h, edge_index, edge_type))

        out = f.relu(self.agg(torch.cat([h, o], dim=1)))

        # Apply a final (linear) classifier.
        out = self.classifier(out)
        out = out.tanh() * 5.0
        return out


class Critic(nn.Module):
    def __init__(self, node_dim=3, edge_dim=2, action_dim=1):
        super(Critic, self).__init__()
        # node encoder
        self.n_enc1 = nn.Linear(node_dim + action_dim, 64)
        # edge encoder
        self.e_enc1 = nn.Linear(edge_dim, 32)
        # omega encoder
        self.o_enc1 = nn.Linear(2, 64)

        self.conv1 = RGCNLayer(64, 64, 8, 32)
        self.conv2 = gnn.RGCNConv(64, 64, 8, aggr='max')
        self.agg1 = nn.Linear(128, 64)
        self.classifier1 = nn.Linear(64, 1)

        # node encoder
        self.n_enc2 = nn.Linear(node_dim + action_dim, 64)
        # edge encoder
        self.e_enc2 = nn.Linear(edge_dim, 32)
        # omega encoder
        self.o_enc2 = nn.Linear(2, 64)

        self.conv3 = RGCNLayer(64, 64, 8, 32)
        self.conv4 = gnn.RGCNConv(64, 64, 8, aggr='max')
        self.agg2 = nn.Linear(128, 64)
        self.classifier2 = nn.Linear(64, 1)

    def forward(self, data, action, key, omega):
        if key == 's':
            x = data.x_s
            edge_index = data.edge_index_s
            edge_attr = data.edge_attr_s
            edge_type = data.edge_type_s
            batch = data.x_s_batch
        else:
            x = data.x_t
            edge_index = data.edge_index_t
            edge_attr = data.edge_attr_t
            edge_type = data.edge_type_t
            batch = data.x_t_batch

        n1 = f.relu(self.n_enc1(torch.cat([x, action], dim=1)))
        e1 = f.relu(self.e_enc1(edge_attr))
        o1 = f.relu(self.o_enc1(omega))

        h1 = f.relu(self.conv1(n1, edge_index, edge_type, e1))
        h1 = f.relu(self.conv2(h1, edge_index, edge_type))

        out1 = f.relu(self.agg1(torch.cat([h1, o1], dim=1)))

        # Apply a final (linear) classifier.
        out1 = self.classifier1(out1)
        out1 = gnn.global_mean_pool(out1, batch)

        n2 = f.relu(self.n_enc2(torch.cat([x, action], dim=1)))
        e2 = f.relu(self.e_enc2(edge_attr))
        o2 = f.relu(self.o_enc2(omega))

        h2 = f.relu(self.conv3(n2, edge_index, edge_type, e2))
        h2 = f.relu(self.conv4(h2, edge_index, edge_type))

        out2 = f.relu(self.agg2(torch.cat([h2, o2], dim=1)))

        # Apply a final (linear) classifier.
        out2 = self.classifier2(out2)
        out2 = gnn.global_mean_pool(out2, batch)

        return out1, out2

    def q1(self, data, action, key, omega):
        if key == 's':
            x = data.x_s
            edge_index = data.edge_index_s
            edge_attr = data.edge_attr_s
            edge_type = data.edge_type_s
            batch = data.x_s_batch
        else:
            x = data.x_t
            edge_index = data.edge_index_t
            edge_attr = data.edge_attr_t
            edge_type = data.edge_type_t
            batch = data.x_t_batch

        n1 = f.relu(self.n_enc1(torch.cat([x, action], dim=1)))
        e1 = f.relu(self.e_enc1(edge_attr))
        o1 = f.relu(self.o_enc1(omega))

        h1 = f.relu(self.conv1(n1, edge_index, edge_type, e1))
        h1 = f.relu(self.conv2(h1, edge_index, edge_type))

        out1 = f.relu(self.agg1(torch.cat([h1, o1], dim=1)))

        # Apply a final (linear) classifier.
        out1 = self.classifier1(out1)
        out1 = gnn.global_mean_pool(out1, batch)

        return out1
