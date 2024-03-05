from flow.envs.base_gpt import Env
import torch
from gym.spaces.box import Box
from gym.spaces import MultiBinary

import numpy as np

def order_vehicles(state):
    distances = {}
    ordered_vehicles = []

    for veh in list(state.keys()):
        perturbation = 1e-6*np.random.randn()
        dist = np.sqrt(state[veh][0]**2 + state[veh][1]**2) + perturbation
        distances[dist] = veh

    for _ in list(state.keys()):
        min_dist = min(list(distances.keys()))
        ordered_vehicles.append(distances[min_dist])
        distances.pop(min_dist)

    return ordered_vehicles

lanes = { 0: [0.0, 0.0, 1.0],
          1: [0.0, 1.0, 0.0],
          2: [1.0, 0.0, 0.0]
        }
ways = { ('t_c', 'c_l'): [1.0, 0.0, 0.0], ('t_c', 'c_b'): [0.0, 1.0, 0.0], ('t_c', 'c_r'): [0.0, 0.0, 1.0],
         ('r_c', 'c_t'): [1.0, 0.0, 0.0], ('r_c', 'c_l'): [0.0, 1.0, 0.0], ('r_c', 'c_b'): [0.0, 0.0, 1.0],
         ('b_c', 'c_r'): [1.0, 0.0, 0.0], ('b_c', 'c_t'): [0.0, 1.0, 0.0], ('b_c', 'c_l'): [0.0, 0.0, 1.0],
         ('l_c', 'c_b'): [1.0, 0.0, 0.0], ('l_c', 'c_r'): [0.0, 1.0, 0.0], ('l_c', 'c_t'): [0.0, 0.0, 1.0]
       }
queues = { 't_c': [1.0, 0.0, 0.0, 0.0],
           'b_c': [0.0, 1.0, 0.0, 0.0],
           'r_c': [0.0, 0.0, 1.0, 0.0],
           'l_c': [0.0, 0.0, 0.0, 1.0],
         }

ADDITIONAL_ENV_PARAMS = {
    # maximum velocity for autonomous vehicles, in m/s
    'max_speed': 13.9,
    'max_accel': 1,
    'max_decel': 1
}

def goal_position(way, lane, queue):
    if way[0]*queue[2] == 1.0:
        return 0.08, 1
    elif way[2]*queue[3] == 1.0:
        return 0.016, 1
    elif way[1]*queue[0] == 1.0:
        return -0.048, -1
    elif way[0]*queue[0] == 1.0:
        return -1, 0.08
    elif way[1]*queue[3] == 1.0:
        return 1, -0.048
    elif way[2]*queue[1] == 1.0:
        return -1, 0.016
    elif way[1]*queue[1] == 1.0:
        return 0.048, 1
    elif way[2]*queue[2] == 1.0:
        return -0.016, -1
    elif way[0]*queue[3] == 1.0:
        return -0.08, -1
    elif way[0]*queue[1] == 1.0:
        return 1, -0.08
    elif way[1]*queue[2] == 1.0:
        return -1, 0.048
    elif way[2]*queue[0] == 1.0:
        return 1, -0.016


class SpeedEnv(Env):
    """Fully observed velocity environment.

    This environment used to train autonomous vehicles to improve traffic flows
    when velocity actions are permitted by the rl agent.

    Required from env_params:

    * max_speed: maximum speed for autonomous vehicles, in m/s^2
    * sort_vehicles: specifies whether vehicles are to be sorted by position
      during a simulation step. If set to True, the environment parameter
      self.sorted_ids will return a list of all vehicles sorted in accordance
      with the environment

    States
        The state consists of (for each vehicle in the network):
        - relative position to the center of the intersection on the x-axis
        - relative position to the center of the intersection on the y-axis
        - vehicle speed
        - vehicle orientation angle
        - lane of approach (one-hot)
        - way the vehicle will follow (one-hot)
        - intersection branch through which the vehicle is approaching (one-hot)

    Actions
        Actions are a list of speeds for each rl vehicle
        
    Rewards
        The reward function is a summation of three terms (for each vehicle):
        - -100 if there was a collision
        - +100 if the intersection was crossed
        - -timestep to encourage crossing as fast as possible
        
    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))
        
        super().__init__(env_params, sim_params, network, simulator)

        #self.last_positions = {"rl_0": 0.0, "rl_1": 0.0, "rl_2": 0.0, "rl_3": 0.0}
        self.last_positions = {}
        self.state_dim = 14

    @property
    def action_space(self):
        """See class definition."""
        num_vehicles = len(self.k.vehicle.get_ids())
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(num_vehicles, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        vehs = len(self.k.vehicle.get_ids())
        obs_space = Box(low=-1, high=1, shape=(vehs,self.state_dim*vehs))
            
        return obs_space

    def _apply_rl_actions(self, rl_actions, vehs):
        """See class definition."""
        self.k.vehicle.apply_acceleration(vehs, rl_actions)

    def compute_reward(self, vehs, **kwargs):
        """See class definition."""
        
        # CURRENTLY TESTING:
        # 1. displ - 1 + finished(10) - collision(100)
        # 2. displ + finished(10) - collision(10)

        ids = self.k.vehicle.get_ids()
        # collided_vehicles
        coll_veh = self.k.simulation.collided_vehicles()
        # successful_vehicles
        #succ_veh = self.k.simulation.successful_vehicles()
        
        crash = self.k.simulation.check_collision()

        reward = torch.tensor([0.0])
        not_done = torch.tensor([1])

        if crash:
            reward += torch.tensor([-100.0])
            not_done = torch.tensor([0])
            return reward, not_done

        for i in vehs:
            if i in ids:
                #pos = self.k.vehicle.get_position(i)
                #displ = pos - self.last_positions[i]
                pos = self.k.vehicle.get_distance(i)
                if i in list(self.last_positions.keys()):
                    displ = pos - self.last_positions[i]
                else:
                    displ = pos
                self.last_positions[i] = pos
                #if displ < 0:
                #    displ = pos
                reward += torch.tensor([displ - 1])
            else:
                reward += torch.tensor([10.0])
                #return reward, not_done
            
        return reward, not_done

    def get_state(self):
        """See class definition."""
        
        ids = self.k.vehicle.get_ids()
        state_dict = {}
        
        for q in ids:
            obs = []
            
            # POSITION
            pos = self.k.vehicle.get_2d_position(q)
            obs.append(pos[0])
            obs.append(pos[1])

            # VELOCITY
            vel = self.k.vehicle.get_speed(q)
            obs.append(vel)
            
            # ACCELERATION
            acc = self.k.vehicle.get_realized_accel(q)
            if acc is None:
                acc = 0
            else:
                acc = np.clip(acc, -1, 1)
            obs.append(acc)

            # HEADING ANGLE
            angle = self.k.vehicle.get_orientation(q)[2]
            obs.append(angle)

            # DISTANCE TRAVELLED
            distance = self.k.vehicle.get_distance(q)
            obs.append(distance)

            # COLLISION
            if q in self.k.simulation.collided_vehicles():
                coll = 1
            else:
                coll = 0
            obs.append(coll)

            # LANE, WAY AND QUEUE
            if self.k.vehicle.get_route(q) == '': # just to fix a simulator bug
                #lane = [0.0, 0.0, 0.0]
                way = [0.0, 0.0, 0.0]
                queue = [0.0, 0.0, 0.0, 0.0]
            else:
                way = ways[self.k.vehicle.get_route(q)]
                #lane = [way[2], way[1], way[0]]
                queue = queues[self.k.vehicle.get_route(q)[0]]
            
            obs = obs + way + queue
            
            state_dict[q] = obs
        
        ord_vehs = order_vehicles(state_dict)
        state = torch.tensor([], dtype=torch.float32)
        for k in range(len(ord_vehs)):
            ego_state = torch.as_tensor(state_dict[ord_vehs[k]], dtype=torch.float32)
            state = torch.cat((state, ego_state))
        
        num_arrived = self.k.vehicle.get_num_arrived()
        if num_arrived > 0:
            if len(ids) > 0:
                aug_col = torch.zeros(self.state_dim*num_arrived, dtype=torch.float32)
                state = torch.cat((state, aug_col))
            else:
                state = torch.zeros(self.state_dim*num_arrived, dtype=torch.float32)
                        
        return state, ord_vehs
