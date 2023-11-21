import numpy as np
import torch
from numpy.linalg import inv
from numpy import pi, sin, cos, linspace

torch.cuda.set_device(2)
device = torch.device('cuda:2')

edges_dict = {"b_c": 0, "c_r": 1, "r_c": 2, "c_t": 3, "t_c": 4, "c_l": 5,
              "l_c": 6, "c_b": 7, ":center_0": 8, ":center_1": 9, ":center_2": 10,
              ":center_12": 10, ":center_3": 11, ":center_4": 12, ":center_5": 13,
              ":center_6": 14, ":center_7": 15, ":center_8": 16, ":center_13": 16,
              ":center_9": 17, ":center_10": 18, ":center_11": 19}

routes_dict = {('t_c', 'c_l'): 0, ('t_c', 'c_b'): 1, ('t_c', 'c_r'): 2, ('r_c', 'c_t'): 3,
               ('r_c', 'c_l'): 4, ('r_c', 'c_b'): 5, ('b_c', 'c_r'): 6, ('b_c', 'c_t'): 7,
               ('b_c', 'c_l'): 8, ('l_c', 'c_b'): 9, ('l_c', 'c_r'): 10, ('l_c', 'c_t'): 11}

all_vehicles = {}

conflicting_routes_matrix = np.zeros((12,12))
for i in range(12):
    for j in range(12):
        if i == 0:
            if j in (4,8):
                conflicting_routes_matrix[i][j] = 1
        elif i == 1:
            if j in (4,5,8,9,10,11):
                conflicting_routes_matrix[i][j] = 1
        elif i == 2:
            if j in (4,5,6,7,8,10,11):
                conflicting_routes_matrix[i][j] = 1
        elif i == 3:
            if j in (7,11):
                conflicting_routes_matrix[i][j] = 1
        elif i == 4:
            if j in (0,1,2,7,8,11):
                conflicting_routes_matrix[i][j] = 1
        elif i == 5:
            if j in (1,2,7,8,9,10,11):
                conflicting_routes_matrix[i][j] = 1
        elif i == 6:
            if j in (2,10):
                conflicting_routes_matrix[i][j] = 1
        elif i == 7:
            if j in (2,3,4,5,10,11):
                conflicting_routes_matrix[i][j] = 1
        elif i == 8:
            if j in (0,1,2,4,5,10,11):
                conflicting_routes_matrix[i][j] = 1
        elif i == 9:
            if j in (1,5):
                conflicting_routes_matrix[i][j] = 1
        elif i == 10:
            if j in (1,2,5,6,7,8):
                conflicting_routes_matrix[i][j] = 1
        else:
            if j in (1,2,3,4,5,7,8):
                conflicting_routes_matrix[i][j] = 1
                
routes_edges_matrix = np.zeros((12,20))
for i in range(12):
    if i == 0:
        routes_edges_matrix[i][4] = 1
        routes_edges_matrix[i][8] = 2
        routes_edges_matrix[i][5] = 3
    elif i == 1:
        routes_edges_matrix[i][4] = 1
        routes_edges_matrix[i][9] = 2
        routes_edges_matrix[i][7] = 3
    elif i == 2:
        routes_edges_matrix[i][4] = 1
        routes_edges_matrix[i][10] = 2
        routes_edges_matrix[i][1] = 3
    elif i == 3:
        routes_edges_matrix[i][2] = 1
        routes_edges_matrix[i][11] = 2
        routes_edges_matrix[i][3] = 3
    elif i == 4:
        routes_edges_matrix[i][2] = 1
        routes_edges_matrix[i][12] = 2
        routes_edges_matrix[i][5] = 3
    elif i == 5:
        routes_edges_matrix[i][2] = 1
        routes_edges_matrix[i][13] = 2
        routes_edges_matrix[i][7] = 3
    elif i == 6:
        routes_edges_matrix[i][0] = 1
        routes_edges_matrix[i][14] = 2
        routes_edges_matrix[i][1] = 3
    elif i == 7:
        routes_edges_matrix[i][0] = 1
        routes_edges_matrix[i][15] = 2
        routes_edges_matrix[i][3] = 3
    elif i == 8:
        routes_edges_matrix[i][0] = 1
        routes_edges_matrix[i][16] = 2
        routes_edges_matrix[i][5] = 3
    elif i == 9:
        routes_edges_matrix[i][6] = 1
        routes_edges_matrix[i][17] = 2
        routes_edges_matrix[i][7] = 3
    elif i == 10:
        routes_edges_matrix[i][6] = 1
        routes_edges_matrix[i][18] = 2
        routes_edges_matrix[i][1] = 3
    else:
        routes_edges_matrix[i][6] = 1
        routes_edges_matrix[i][19] = 2
        routes_edges_matrix[i][3] = 3

def compute_edges(env, state):
    
    dimensions_matrix = np.array([[25/4,0], [0,1]])
    edges = {}
    edges_type = {}
    for i in env.k.vehicle.get_ids():
        for j in env.k.vehicle.get_ids():
            if conflicting_routes_matrix[state[i][7]][state[j][7]] == 1:
                if (routes_edges_matrix[state[i][7]][state[i][6]] != 3) and (routes_edges_matrix[state[j][7]][state[j][6]] != 3):
                    # DISTANCE
                    rotation_matrix = np.array([[np.cos(state[i][5]), -np.sin(state[i][5])],
                                               [np.sin(state[i][5]), np.cos(state[i][5])]])
                    sigma_matrix = np.matmul(np.matmul(rotation_matrix, dimensions_matrix),
                                             rotation_matrix.transpose())
                    cartesian_dist = np.array(state[j][4]) - np.array(state[i][4])
                    d_ij = np.matmul(np.matmul(cartesian_dist.transpose(), inv(sigma_matrix)),
                                     cartesian_dist)
                    d_ij = 1/np.sqrt(d_ij)
                
                    # BEARING
                    coord_j = np.array(state[j][4])
                    coord_i = np.array(state[i][4])
                    py = coord_i[1] - coord_j[1]
                    px = coord_i[0] - coord_j[1]
                    chi_ij = np.arctan(py/px) - state[j][5]
                    
                    # PRIORITY
                    # IN STALLO PER ORA
                    
                    edges_type[(i,j)] = 'crossing'
                    
                    edges[(i,j)] = (d_ij, chi_ij)
                
                elif np.argmax(routes_edges_matrix[state[i][7]]) == np.argmax(routes_edges_matrix[state[j][7]]):
                    if state[i][0] > state[j][0]:
                        # DISTANCE
                        rotation_matrix = np.array([[np.cos(state[i][5]), -np.sin(state[i][5])],
                                                   [np.sin(state[i][5]), np.cos(state[i][5])]])
                        sigma_matrix = np.matmul(np.matmul(rotation_matrix, dimensions_matrix),
                                                 rotation_matrix.transpose())
                        cartesian_dist = np.array(state[j][4]) - np.array(state[i][4])
                        d_ij = np.matmul(np.matmul(cartesian_dist.transpose(), inv(sigma_matrix)),
                                         cartesian_dist)
                        d_ij = 1/np.sqrt(d_ij)

                        # BEARING
                        coord_j = np.array(state[j][4])
                        coord_i = np.array(state[i][4])
                        py = coord_i[1] - coord_j[1]
                        px = coord_i[0] - coord_j[1]
                        chi_ij = np.arctan(py/px) - state[j][5]

                        # PRIORITY
                        # IN STALLO PER ORA
                        
                        edges_type[(i,j)] = 'same_lane'
                        
                        edges[(i,j)] = (d_ij, chi_ij)
            
            elif state[i][7] == state[j][7]:
                if state[i][0] > state[j][0]:
                    # DISTANCE
                    rotation_matrix = np.array([[np.cos(state[i][5]), -np.sin(state[i][5])],
                                               [np.sin(state[i][5]), np.cos(state[i][5])]])
                    sigma_matrix = np.matmul(np.matmul(rotation_matrix, dimensions_matrix),
                                             rotation_matrix.transpose())
                    cartesian_dist = np.array(state[j][4]) - np.array(state[i][4])
                    d_ij = np.matmul(np.matmul(cartesian_dist.transpose(), inv(sigma_matrix)),
                                     cartesian_dist)
                    d_ij = 1/np.sqrt(d_ij)

                    # BEARING
                    coord_j = np.array(state[j][4])
                    coord_i = np.array(state[i][4])
                    py = coord_i[1] - coord_j[1]
                    px = coord_i[0] - coord_j[1]
                    chi_ij = np.arctan(py/px) - state[j][5]

                    # PRIORITY
                    # IN STALLO PER ORA
                    
                    edges_type[(i,j)] = 'same_lane'
                    
                    edges[(i,j)] = (d_ij, chi_ij)
            
            elif state[i][6] == state[j][6]:
                if state[i][0] > state[j][0]:
                    # DISTANCE
                    rotation_matrix = np.array([[np.cos(state[i][5]), -np.sin(state[i][5])],
                                               [np.sin(state[i][5]), np.cos(state[i][5])]])
                    sigma_matrix = np.matmul(np.matmul(rotation_matrix, dimensions_matrix),
                                             rotation_matrix.transpose())
                    cartesian_dist = np.array(state[j][4]) - np.array(state[i][4])
                    d_ij = np.matmul(np.matmul(cartesian_dist.transpose(), inv(sigma_matrix)),
                                     cartesian_dist)
                    d_ij = 1/np.sqrt(d_ij)

                    # BEARING
                    coord_j = np.array(state[j][4])
                    coord_i = np.array(state[i][4])
                    py = coord_i[1] - coord_j[1]
                    px = coord_i[0] - coord_j[1]
                    chi_ij = np.arctan(py/px) - state[j][5]

                    # PRIORITY
                    # IN STALLO PER ORA

                    edges_type[(i,j)] = 'same_lane'
                    
                    edges[(i,j)] = (d_ij, chi_ij)
                    
    return edges, edges_type

def compute_rp(edges):
    d = np.inf
    for i in range(len(list(edges.values()))):
        d_i = 1/list(edges.values())[i][0]
        if d_i < d:
            d = d_i
    
    return -1/d


class Graph:
    def __init__(self, nodes_list, edges_list):
        self.nodes = [i for i in range(len(nodes_list))]
        self.edges = []
        for e in edges_list:
            self.edges = self.edges + [(nodes_list.index(e[0]), nodes_list.index(e[1]))]
        
        self.edata = {}
        self.ndata = {}
        self.sparse_adj = torch.zeros([2,len(self.edges)], dtype=torch.int64, device=device)
        for k in range(len(self.edges)):
            self.sparse_adj[0][k] = self.edges[k][0]
            self.sparse_adj[1][k] = self.edges[k][1]
        
    def num_nodes(self):
        return len(self.nodes)
    
    def num_edges(self):
        return len(self.edges)
    
    def insert_node_features(self, nodes_feat):
        self.ndata['x'] = nodes_feat
        
    def insert_edge_features(self, edges_feat, edges_types): 
        self.edata['x'] = edges_feat
        self.edata['type'] = list(edges_types.values())
