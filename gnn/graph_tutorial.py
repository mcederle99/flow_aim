import networkx as nx

# Creating a graph
G = nx.Graph()

# Adding nodes
G.add_node(1)
G.add_nodes_from([2, 3])

# Adding nodes with node attributes
G.add_nodes_from([(4, {"color": "red"}), (5, {"color": "green"})])

# Adding edges
G.add_edge(1, 2)
e = (2, 3)
G.add_edge(*e)  # unpack edge tuple*
G.add_edges_from([(1, 2), (1, 3)])

# Remove everything from the graph
G.clear()

# Properties of the graph
G.number_of_nodes()
G.number_of_edges()
print(list(G.nodes))
print(list(G.edges))
print(list(G))

# Remove elements from a graph
G.remove_node(2)
G.remove_nodes_from("spam")
G.remove_edge(1, 3)

# Node attributes
# Add node attributes using add_node(), add_nodes_from(), or G.nodes
G.add_node(1, time='5pm')
G.add_nodes_from([3], time='2pm')
G.nodes[1]['room'] = 714
# Note that adding a node to G.nodes does not add it to the graph,
# use G.add_node() to add new nodes. Similarly for edges.

# Edge Attributes
# Add/change edge attributes using add_edge(), add_edges_from(), or subscript notation.
G.add_edge(1, 2, weight=4.7 )
G.add_edges_from([(3, 4), (4, 5)], color='red')
G.add_edges_from([(1, 2, {'color': 'blue'}), (2, 3, {'weight': 8})])
G[1][2]['weight'] = 4.7
G.edges[3, 4]['weight'] = 4.2
# The special attribute weight should be numeric as it is used by algorithms requiring weighted edges.

# Directed graphs
DG = nx.DiGraph()
DG.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
DG.out_degree(1, weight='weight')
