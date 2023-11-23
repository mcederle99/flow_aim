import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from rgcn import RGCNLayer
from utils import Graph

torch.cuda.set_device(2)
device = torch.device('cuda:2')

class Actor(nn.Module):
    def __init__(self, node_dim=4, edge_dim=2, action_dim=1, max_action=5):
        super(Actor, self).__init__()
        
        # node encoder
        self.n_enc = nn.Linear(node_dim, 64)
        # edge encoder
        self.e_enc = nn.Linear(edge_dim, 32)
        
        # first RGCN layer
        self.RGCN1 = RGCNLayer(96, 64, 2)
        # GAT layer
        self.GAT = gnn.GATConv(64, 64, add_self_loops=False, edge_dim=32)
        # second RGCN layer
        self.RGCN2 = RGCNLayer(96, 64, 2)
        
        # node decoder
        self.n_dec = nn.Linear(64, action_dim)
        
        self.max_action = max_action
        
        self.to(device)
        
    def forward(self, nodes, edges, edges_type):
        n_feat = list(nodes.values())
        n_feat = torch.as_tensor(n_feat, dtype=torch.float32, device=device)
        e_feat = list(edges.values())
        e_feat = torch.as_tensor(e_feat, dtype=torch.float32, device=device)
        
        # node encoding
        if n_feat.size()[0] == 0:
            n = n_feat
        else:
            n = self.n_enc(n_feat) # n should be num_nodes*64
            n = F.relu(n)
        
        # edge encoding
        if e_feat.size()[0] == 0:
            e = e_feat
        else:
            e = self.e_enc(e_feat) # e should be num_edges*32
            e = F.relu(e)
        
        # graph embedding
        g = Graph(list(nodes.keys()), list(edges.keys()))
        g.insert_node_features(n)
        g.insert_edge_features(e, edges_type)
        
        # first RGCN layer
        h = self.RGCN1(g)
        h = F.relu(h)
        
        # GAT layer
        h = self.GAT(h, g.sparse_adj, e)
        h = F.relu(h)
        
        # graph embedding
        g = Graph(list(nodes.keys()), list(edges.keys()))
        g.insert_node_features(h)
        g.insert_edge_features(e, edges_type)
        
        # second RGCN layer
        h = self.RGCN2(g)
        h = F.relu(h)
        
        # decoding
        out = self.n_dec(h)
        out = self.max_action*torch.tanh(out)
        
        return out
    
class Critic(nn.Module):
    def __init__(self, node_dim=4, edge_dim=2, action_dim=1, max_action=5, aggr_func='mean'):
        super(Critic, self).__init__()
        
        # node encoder
        self.n_enc = nn.Linear(node_dim+action_dim, 64)
        # edge encoder
        self.e_enc = nn.Linear(edge_dim, 32)
        
        # first RGCN layer
        self.RGCN1 = RGCNLayer(96, 64, 2)
        # GAT layer
        self.GAT = gnn.GATConv(64, 64, add_self_loops=False, edge_dim=32)
        # second RGCN layer
        self.RGCN2 = RGCNLayer(96, 64, 2)
        
        # node decoder
        self.n_dec = nn.Linear(64, 1)
        
        self.max_action = max_action
        self.aggr_func = aggr_func
        
        self.to(device)
        
    def forward(self, nodes, edges, edges_type, actions):
        n_feat = list(nodes.values())
        n_feat = torch.as_tensor(n_feat, dtype=torch.float32, device=device)
        e_feat = list(edges.values())
        e_feat = torch.as_tensor(e_feat, dtype=torch.float32, device=device)
        
        # node encoding
        if n_feat.size()[0] == 0:
            n = n_feat
        else:
            n = torch.cat((n_feat,actions), 1)
            n = self.n_enc(n) # n should be num_nodes*64
            n = F.relu(n)
        
        # edge encoding
        if e_feat.size()[0] == 0:
            e = e_feat
        else:
            e = self.e_enc(e_feat) # e should be num_edges*32
            e = F.relu(e)
        
        # graph embedding
        g = Graph(list(nodes.keys()), list(edges.keys()))
        g.insert_node_features(n)
        g.insert_edge_features(e, edges_type)
        
        # first RGCN layer
        h = self.RGCN1(g)
        h = F.relu(h)
        
        # GAT layer
        h = self.GAT(h, g.sparse_adj, e)
        h = F.relu(h)

        # graph embedding
        g = Graph(list(nodes.keys()), list(edges.keys()))
        g.insert_node_features(h)
        g.insert_edge_features(e, edges_type)
        
        # second RGCN layer
        h = self.RGCN2(g)
        h = F.relu(h)

        # decoding
        if self.aggr_func == 'mean':
            if g.num_nodes() > 0:
                h = torch.sum(h, dim=0)/g.num_nodes()
            else:
                h = torch.sum(h, dim=0)
        out = self.n_dec(h)
        
        return out
