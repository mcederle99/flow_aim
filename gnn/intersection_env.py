from utils import edges_dict, routes_dict, routes_edges_matrix, compute_edges, from_networkx_multigraph
from flow.envs.base import Env
from gym.spaces.box import Box
from gym.spaces import Discrete
import numpy as np
import networkx as nx
import torch

all_vehicles = {}

ADDITIONAL_ENV_PARAMS = {
    "max_accel": 1,
    "max_decel": -1,
}


class MyEnv(Env):

    @property
    def action_space(self):
        num_actions = len(self.k.vehicle.get_rl_ids())
        accel_ub = self.env_params.additional_params["max_accel"]
        accel_lb = -abs(self.env_params.additional_params["max_decel"])

        return Box(low=accel_lb,
                   high=accel_ub,
                   shape=(num_actions,))
    
    @property
    def observation_space(self):
        nodes = {}
        for i in self.k.vehicle.get_ids():
            nodes[i] = (Box(low=-float("inf"), high=float("inf"), shape=(1,)),  # POSITION
                        Box(low=0, high=float("inf"), shape=(1,)),              # VELOCITY
                        Box(low=0, high=float("inf"), shape=(1,)),              # ACCELERATION
                        Box(low=-float("inf"), high=float("inf"), shape=(2,)),  # COORDINATES
                        Box(low=0, high=1, shape=(1,)),                         # HEADING ANGLE
                        Discrete(20),                                           # EDGE
                        Discrete(12))                                           # ROUTE
            
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

        graph = nx.MultiDiGraph()
        
        for q in ids:
            
            # POSITION
            if q not in all_vehicles.keys():
                all_vehicles[q] = 0.5

            raw_pos = self.k.vehicle.get_position(q)
            if self.k.vehicle.get_route(q) == '':  # just to fix a simulator bug
                i = 0
            else:
                i = routes_dict[self.k.vehicle.get_route(q)]
            if self.k.vehicle.get_edge(q) == '':  # just to fix a simulator bug
                j = 5
            else:
                j = edges_dict[self.k.vehicle.get_edge(q)]
            if routes_edges_matrix[i][j] == 1:
                pos = raw_pos - 70
            elif routes_edges_matrix[i][j] == 2:
                if i in (1, 4, 7, 10):  # straight paths
                    pos = raw_pos - 20
                elif i in (2, 8):  # left turns part I
                    if np.abs(raw_pos-all_vehicles[q]) > 2.5:
                        pos = raw_pos + 7 - 20
                        all_vehicles[q] = 50
                    else:
                        pos = raw_pos - 20
                        all_vehicles[q] = raw_pos
                elif i in (5, 11):  # left turns part II
                    pos = raw_pos - 20
                else:  # right turns
                    pos = 20 / 14.5 * raw_pos - 20
            else:
                pos = raw_pos

            # VELOCITY
            vel = self.k.vehicle.get_speed(q)
            if routes_dict[self.k.vehicle.get_route(q)] in (0, 3, 6, 9):
                vel = vel / 13.0
            elif routes_dict[self.k.vehicle.get_route(q)] in (1, 4, 7, 10):
                vel = vel / 17.0
            else:
                vel = vel / 14.0

            # ACCELERATION
            acc = self.k.vehicle.get_realized_accel(q)
            acc = np.clip(acc, -1, 1)
            if acc is None:
                acc = 0

            # COORDINATES
            coordx = (self.k.vehicle.get_2d_position(q)[0] - 50.0) / 100.0
            coordy = (self.k.vehicle.get_2d_position(q)[1] - 50.0) / 100.0
            coord = (coordx, coordy)

            # HEADING ANGLE
            angle = self.k.vehicle.get_orientation(q)[2] / 360.0
            
            # EDGE
            if self.k.vehicle.get_edge(q) == '':  # just to fix a simulator bug
                edge = 5
            else:
                edge = edges_dict[self.k.vehicle.get_edge(q)]
                              
            # ROUTE
            if self.k.vehicle.get_route(q) == '':  # just to fix a simulator bug
                route = 0
            else:
                route = routes_dict[self.k.vehicle.get_route(q)]

            state[q] = (pos, vel, acc, coord, angle, edge, route)

            graph.add_node(q, pos=torch.tensor([state[q][0]], dtype=torch.float),
                           vel=torch.tensor([state[q][1]], dtype=torch.float),
                           acc=torch.tensor([state[q][2]], dtype=torch.float))

        edges, edges_type = compute_edges(self, state)

        for edge in list(edges.keys()):
            graph.add_edge(edge[0], edge[1], key=edges_type[edge],
                           dist=torch.tensor([edges[edge][0]], dtype=torch.float),
                           bearing=torch.tensor([edges[edge][1]], dtype=torch.float))

        state = from_networkx_multigraph(graph)

        return state
                                
    def compute_reward(self, rl_actions, state=None, **kwargs):
        w_v = 0.03
        w_a = 0.01
        w_i = 0.1
        w_c = 1
        
        # the get_ids() method is used to get the names of all vehicles in the network
        ids = self.k.vehicle.get_ids()
        if len(ids) > 0:
            not_empty = True
        else:
            not_empty = False

        crash = self.k.simulation.check_collision()
        
        # VELOCITY TERM
        speeds = 0
        max_vel = 0
        for q in ids:
            vel = self.k.vehicle.get_speed(q)
            if routes_dict[self.k.vehicle.get_route(q)] in (0, 3, 6, 9):
                vel = vel / 13.0
            elif routes_dict[self.k.vehicle.get_route(q)] in (1, 4, 7, 10):
                vel = vel / 17.0
            else:
                vel = vel / 14.0
            speeds += vel
            if vel > max_vel:
                max_vel = vel

        mean_speed = speeds / len(ids) if not_empty else 0.0

        if mean_speed <= 0.8:
            rv = mean_speed * 1.25
        elif mean_speed <= 1.0:
            rv = 1.0
        else:
            rv = 6.0 - 5.0 * mean_speed
        if crash:
            rv = 0

        # ACTION TERM
        if not not_empty or len(rl_actions) == 0:
            ra = 0
        else:
            ra = -np.mean(np.abs(rl_actions))

        # IDLE TERM
        if not_empty:
            if max_vel < 0.3:
                ri = -1
            else:
                ri = 0
        else:
            ri = 0
        
        # COLLISION TERM
        if crash:
            rc = -1
        else:
            rc = 0
            
        r = w_v * rv + w_a * ra + w_i * ri + w_c * rc

        return r
