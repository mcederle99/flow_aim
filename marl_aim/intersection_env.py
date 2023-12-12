edges_dict = {"b_c": 0, "c_r": 1, "r_c": 2, "c_t": 3, "t_c": 4, "c_l": 5,
              "l_c": 6, "c_b": 7, ":center_0": 8, ":center_1": 9, ":center_2": 10,
              ":center_12": 10, ":center_3": 11, ":center_4": 12, ":center_5": 13,
              ":center_6": 14, ":center_7": 15, ":center_8": 16, ":center_13": 16,
              ":center_9": 17, ":center_10": 18, ":center_11": 19}

routes_dict = {('t_c', 'c_l'): 0, ('t_c', 'c_b'): 1, ('t_c', 'c_r'): 2, ('r_c', 'c_t'): 3,
               ('r_c', 'c_l'): 4, ('r_c', 'c_b'): 5, ('b_c', 'c_r'): 6, ('b_c', 'c_t'): 7,
               ('b_c', 'c_l'): 8, ('l_c', 'c_b'): 9, ('l_c', 'c_r'): 10, ('l_c', 'c_t'): 11}

all_vehicles = {}

import numpy as np

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


from flow.envs.base import Env
from gym.spaces.box import Box
from gym.spaces import Tuple
from gym.spaces import Discrete
import numpy as np
from numpy.linalg import inv

ADDITIONAL_ENV_PARAMS = {
    "max_accel": 3,
    "max_decel": -3,
}

class myEnv(Env):

    @property
    def action_space(self):
        num_actions = self.initial_vehicles.num_rl_vehicles
        accel_ub = self.env_params.additional_params["max_accel"]
        accel_lb = - abs(self.env_params.additional_params["max_decel"])

        return Box(low=accel_lb,
                   high=accel_ub,
                   shape=(num_actions,))
    
    @property
    def observation_space(self):
        nodes = {}
        for i in self.k.vehicle.get_ids():
            nodes[i] = (Box(low=-float("inf"), high=float("inf"), shape=(1,)), # POSITION
                        Box(low=0, high=float("inf"), shape=(1,)),             # VELOCITY 
                        Box(low=0, high=float("inf"), shape=(1,)),             # ACCELERATION
                        Discrete(2),                                           # CONTROLLABLE
                        Box(low=-float("inf"), high=float("inf"), shape=(2,)), # COORDINATES
                        Box(low=-float("inf"), high=float("inf"), shape=(1,)), # HEADING ANGLE
                        Discrete(20),                                          # EDGE
                        Discrete(12),                                          # ROUTE
                       )
            
        return nodes
    
    def _apply_rl_actions(self, rl_actions):
        # the names of all autonomous (RL) vehicles in the network
        rl_ids = self.k.vehicle.get_rl_ids()
        # use the base environment method to convert actions into accelerations for the rl vehicles
        self.k.vehicle.apply_acceleration(rl_ids, rl_actions)
        
    def get_state(self, **kwargs):
        
        # the get_ids() method is used to get the names of all vehicles in the network
        ids = self.k.vehicle.get_ids()
        state = {}
        
        for q in ids:
            
            # POSITION
            if q not in all_vehicles.keys():
                all_vehicles[q] = False
            
            pos = -42
            old_pos = -12
            raw_pos = self.k.vehicle.get_position(q)
            if self.k.vehicle.get_route(q) == '': # just to fix a simulator bug
                i = 0
            else:
                i = routes_dict[self.k.vehicle.get_route(q)]
            if self.k.vehicle.get_edge(q) == '': # just to fix a simulator bug
                j = 5
            else:
                j = edges_dict[self.k.vehicle.get_edge(q)]
            if routes_edges_matrix[i][j] == 1:
                pos = raw_pos - 42
            elif routes_edges_matrix[i][j] == 2:
                if i in (1,4,7,10):
                    pos = raw_pos - 12
                elif i in (2,5,8,11):
                    if not all_vehicles[q]:
                        if abs(pos+12-raw_pos) > 3:
                            all_vehicles[q] = True
                            pos = pos + raw_pos
                        else:
                            pos = raw_pos - 12
                    else:
                        pos = raw_pos + 4 - 12
                else:
                    old_pos = pos
                    pos = raw_pos - 12
                    ang_coeff = 12/7
                    rel_displ = ang_coeff*(pos-old_pos)
                    pos = old_pos + rel_displ
            else:
                pos = raw_pos
                
            # VELOCITY
            vel = self.k.vehicle.get_speed(q)
            
            # ACCELERATION
            acc = self.k.vehicle.get_realized_accel(q)
            if acc == None:
                acc = 0
            
            # CONTROLLABLE
            if self.k.vehicle.get_type(q) == 'human':
                contr = 0
            else:
                contr = 1
                
            # COORDINATES
            coord = self.k.vehicle.get_2d_position(q)
            
            # HEADING ANGLE
            angle = self.k.vehicle.get_orientation(q)[2]
            
            # EDGE
            if self.k.vehicle.get_edge(q) == '': # just to fix a simulator bug
                edge = 5
            else:
                edge = edges_dict[self.k.vehicle.get_edge(q)]
                              
            # ROUTE
            if self.k.vehicle.get_route(q) == '': # just to fix a simulator bug
                route = 0
            else:
                route = routes_dict[self.k.vehicle.get_route(q)]

                                
            state[q] = (pos, vel, acc, contr, coord, angle, edge, route)
                                
        return state
                                
    def compute_reward(self, rl_actions, state, **kwargs):
        #speed_limit = 25
        w_v = 0.5
        #w_a = 0.01
        w_i = 0.5 # before it was 0.5
        #w_c = 1
        
        # the get_ids() method is used to get the names of all vehicles in the network
        ids = self.k.vehicle.get_ids()
        crash = self.k.simulation.check_collision()
       
        rewards = []
        for i in list(state.keys()):
            if i in ids:

                # VELOCITY TERM
                speed = self.k.vehicle.get_speed(i)

                if crash:
                    Rv = 0
                else:
                    Rv = speed
            
                ## ACTION TERM
                #if speeds == [] or len(rl_actions) == 0:
                #    Ra = 0
                #else:
                #    Ra = -np.mean(np.abs(rl_actions))

                # IDLE TERM
                if speed < 0.3:
                    Ri = -1
                else:
                    Ri = 0
        
                R = w_v*Rv + w_i*Ri

            else:
                R = 0

            rewards.append(R)

        return rewards
