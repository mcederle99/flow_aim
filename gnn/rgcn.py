import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Linear
import torch.nn.functional as f


class RGCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, edge_channels):
        super(RGCNLayer, self).__init__(aggr='max')  # Use max aggregation
        self.num_relations = num_relations
        self.edge_channels = edge_channels
        self.out_channels = out_channels

        self.linear_node = Linear(in_channels, out_channels, bias=False)
        self.relation_weights = torch.nn.ModuleList([
            Linear(in_channels + edge_channels, out_channels, bias=False) for _ in range(num_relations)
        ])

    def forward(self, x, edge_index, edge_type, edge_attr):
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        for rel in range(self.num_relations):
            mask = edge_type == rel
            if mask.sum() == 0:
                continue
            edge_index_rel = edge_index[:, mask]
            edge_attr_rel = edge_attr[mask]
            out += self.propagate(edge_index_rel, x=x, edge_attr=edge_attr_rel, rel=rel)
        return f.relu(out + self.linear_node(x))

    def message(self, x_j, edge_attr, rel):
        # Concatenate node feature and edge feature
        edge_msg = torch.cat([x_j, edge_attr], dim=-1)
        return self.relation_weights[rel](edge_msg)

    def update(self, aggr_out):
        # The default behavior is to return the aggregated messages
        return aggr_out


# Example usage
# num_nodes = 100
# num_edges = 200
# in_channels = 16
# out_channels = 32
# num_relations = 4
# edge_channels = 8  # Dimension of the edge feature vector
#
# rgcn_layer = RGCNLayer(in_channels, out_channels, num_relations, edge_channels)
# x = torch.randn((num_nodes, in_channels))  # Node features
# edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Edge indices
# edge_type = torch.randint(0, num_relations, (num_edges,))  # Edge types
# edge_attr = torch.randn((num_edges, edge_channels))  # Edge features

# out = rgcn_layer(x, edge_index, edge_type, edge_attr)
