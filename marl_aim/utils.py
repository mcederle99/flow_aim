import numpy as np
import torch
from numpy.linalg import inv
from numpy import pi, sin, cos, linspace

#torch.cuda.set_device(2)
device = torch.device('cuda')

edges_dict = {"b_c": 0, "c_r": 1, "r_c": 2, "c_t": 3, "t_c": 4, "c_l": 5,
              "l_c": 6, "c_b": 7, ":center_0": 8, ":center_1": 9, ":center_2": 10,
              ":center_12": 10, ":center_3": 11, ":center_4": 12, ":center_5": 13,
              ":center_6": 14, ":center_7": 15, ":center_8": 16, ":center_13": 16,
              ":center_9": 17, ":center_10": 18, ":center_11": 19}

routes_dict = {('t_c', 'c_l'): 0, ('t_c', 'c_b'): 1, ('t_c', 'c_r'): 2, ('r_c', 'c_t'): 3,
               ('r_c', 'c_l'): 4, ('r_c', 'c_b'): 5, ('b_c', 'c_r'): 6, ('b_c', 'c_t'): 7,
               ('b_c', 'c_l'): 8, ('l_c', 'c_b'): 9, ('l_c', 'c_r'): 10, ('l_c', 'c_t'): 11}


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

confl_routes_dict = {0: [4,8], 1: [4,5,8,9,10,11], 2: [4,5,6,7,8,10,11], 3: [7,11], 4: [0,1,2,7,8,11], 5: [1,2,7,8,9,10,11],
                6: [2,10], 7: [2,3,4,5,10,11], 8: [0,1,2,4,5,10,11], 9: [1,5], 10: [1,2,5,6,7,8], 11: [1,2,3,4,5,7,8]}
                
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

def find_value_index(matrix, value):
    for i in range(len(matrix)):
            if matrix[i] == value:
                return i
    return None

basta_dizionari = {(4,0): 'f', (4,2): 'l', (4,6): 'r',
                   (0,4): 'f', (0,6): 'l', (0,2): 'r',
                   (6,2): 'f', (6,4): 'l', (6,0): 'r'}

def compute_connections(env, state):
    ids = env.k.vehicle.get_ids()
    interesting_ids = []
    boring_ids = []
    out = {}
    for i in ids:
        out[i] = []
        if routes_edges_matrix[state[i][7]][state[i][6]] == 3:
            boring_ids.append(i)
        else:
            interesting_ids.append(i)

    for i in interesting_ids:
        route = state[i][7]
        edge = state[i][6]
        conflicting_routes = confl_routes_dict[i]
        
        opposite_lane = []
        left_lane = []
        right_lane = []
        same_lane = []
        for j in interesting_ids:
            route_j = state[j][7]
            edge_j = state[j][6]
            if j != i:
                if conflicting_routes_matrix[i][j] == 1:
                    start_edge = find_value_index(routes_edges_matrix[route], 1)
                    start_edge_j = find_value_index(routes_edges_matrix[route_j], 1)
                    direction = basta_dizionari[(start_edge, start_edge_j)]
                    if direction == 'f':
                        opposite_lane.append(j)
                    elif direction == 'l':
                        left_lane.append(j)
                    else:
                        right_lane.append(j)

                elif edge == edge_j or route == route_j:
                    if state[j][0] > state[i][0]:
                        same_lane.append(j)

        if len(opposite_lane) > 0:
            closest = opposite_lane[0]
            for j in opposite_lane:
                if state[j][0] > state[closest][0]:
                    closest = j
            out[i].append(closest)

        if len(left_lane) > 0:
            closest = left_lane[0]
            for j in left_lane:
                if state[j][0] > state[closest][0]:
                    closest = j
            out[i].append(closest)
        if len(right_lane) > 0:
            closest = right_lane[0]
            for j in right_lane:
                if state[j][0] > state[closest][0]:
                    closest = j
            out[i].append(closest)
        if len(same_lane) > 0:
            closest = same_lane[0]
            for j in same_lane:
                if state[j][0] > state[closest][0]:
                    closest = j
            out[i].append(closest)

    for i in boring_ids:
        edge = state[i][6]
        
        same_lane = []
        for j in boring_ids:
            edge_j = state[j][6]
            if (edge == edge_j) and (state[j][0]>state[i][0]):
                same_lane.append(j)

        if len(same_lane) > 0:
            closest = same_lane[0]
            for j in same_lane:
                if state[j][0] > state[closest][0]:
                    closest = j
            out[i].append(closest)

    return out # this is a dictionary where the keys are vehicles IDs and the value of each
               # key corresponds to its connected vehicles (up to four)

def compute_augmented_state(env, state, idx, connections):
    dimensions_matrix = np.array([[25/4,0], [0,1]])
    out = [state[idx][0], state[idx][1], state[idx][2]]
    for veh in connections:
        out.append(state[veh][1])
        out.append(state[veh][2])
        # DISTANCE
        rotation_matrix = np.array([[np.cos(state[idx][5]), -np.sin(state[idx][5])],
                                [np.sin(state[idx][5]), np.cos(state[idx][5])]])
        sigma_matrix = np.matmul(np.matmul(rotation_matrix, dimensions_matrix),
                                rotation_matrix.transpose())
        cartesian_dist = np.array(state[veh][4]) - np.array(state[idx][4])
        d_ij = np.matmul(np.matmul(cartesian_dist.transpose(), inv(sigma_matrix)),
                        cartesian_dist)
        d_ij = 1/np.sqrt(d_ij)
        out.append(d_ij)
                
        # BEARING
        coord_j = np.array(state[veh][4])
        coord_i = np.array(state[idx][4])
        py = coord_i[1] - coord_j[1]
        px = coord_i[0] - coord_j[1]
        chi_ij = np.arctan(py/px) - state[veh][5]
        out.append(chi_ij)

    return out   

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
        self.sparse_adj = torch.zeros([2,len(self.edges)], dtype=torch.long, device=device)
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
        self.edata['type_bin'] = list(edges_types.values())
        for i in range(len(self.edata['type_bin'])):
            if self.edata['type_bin'][i] == 'same_lane':
                self.edata['type_bin'][i] = 0
            else:
                self.edata['type_bin'][i] = 1
        self.edata['type_bin'] = torch.tensor(self.edata['type_bin'])
