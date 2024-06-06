import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat # encoded_nodes_features_dim + encoded_edges_features_dim
        self.out_feat = out_feat # encoded_nodes_features_dim
        self.num_rels = num_rels # 2 per ora
        
        # weight tensors
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat,
                                                self.in_feat))
        self.weight_0 = nn.Parameter(torch.Tensor(self.out_feat, self.out_feat))
            
        # initialize trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight_0,
                                gain=nn.init.calculate_gain('relu'))
            
    def forward(self, g):
        
        #weight = self.weight
        
        enh_adj = {}
        for i in range(g.num_nodes()):
            for j in range(g.num_nodes()):
                if (j,i) in g.edges:
                    enh_adj[(i,j)] = torch.cat((g.ndata['x'][j], g.edata['x'][g.edges.index((j,i))]))
                else:
                    enh_adj[(i,j)] = torch.zeros([1,], device=device)
        
        types = ('same_lane', 'crossing')
        out = torch.zeros([g.num_nodes(),64], device=device)
        for i in range(g.num_nodes()):
            message = torch.zeros([64,], device=device)
            for r in types:
                max_value = -np.inf*torch.ones([64,], device=device)
                for j in range(g.num_nodes()):
                    if torch.sum(enh_adj[(i,j)]) != 0:
                        if g.edata['type'][g.edges.index((j,i))] == r:
                            temp = torch.matmul(self.weight[types.index(r)], enh_adj[(i,j)])
                            max_value = torch.maximum(max_value, temp)
                if torch.sum(max_value) != -np.inf:
                    message = message + max_value
            out[i] = message + torch.matmul(self.weight_0, g.ndata['x'][i])
        
        return out
