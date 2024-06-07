import networkx as nx
import torch
from torch_geometric.utils import from_networkx

def create_graph(num_nodes, edge_list, node_attrs, edge_attrs):
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i, **{f'attr{j}': torch.tensor([node_attrs[i][j]], dtype=torch.float) for j in range(len(node_attrs[0]))})
    for (u, v, attrs) in edge_list:
        G.add_edge(u, v, **{f'attr{j}': torch.tensor([attrs[j]], dtype=torch.float) for j in range(len(attrs))})
    return from_networkx(G)

# Example graphs
graph1 = create_graph(
    num_nodes=3,
    edge_list=[(0, 1, [0.1, 0.2]), (1, 2, [0.3, 0.4])],
    node_attrs=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    edge_attrs=[[0.1, 0.2], [0.3, 0.4]]
)

graph2 = create_graph(
    num_nodes=4,
    edge_list=[(0, 1, [0.5, 0.6]), (1, 2, [0.7, 0.8]), (2, 3, [0.9, 1.0])],
    node_attrs=[[10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21]],
    edge_attrs=[[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
)

from torch_geometric.data import Batch

# Create a batch of graphs
batch = Batch.from_data_list([graph1, graph2])
print(batch)
