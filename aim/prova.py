import torch
#from torch_geometric.nn import RGCNConv
#from torch_geometric.data import Data
import numpy as np

ret = np.load('returns_more_more_vehicles.npy')
print(ret)
raise KeyboardInterrupt

# Define a graph with edge types
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
edge_types = torch.tensor([0, 1, 0, 1], dtype=torch.long)  # Edge types: 0 and 1
x = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.float)

# Create a PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index, edge_type=edge_types)

# Define the number of input and output features, as well as the number of relations
num_features = 2
num_classes = 3
num_relations = 2  # Number of unique edge types

# Create an RGCNConv layer
rgcn_conv = RGCNConv(in_channels=num_features, out_channels=num_classes, num_relations=num_relations)

# Apply the RGCNConv layer to the graph data
output = rgcn_conv(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type)

# Print the result
print(data.edge_index)

