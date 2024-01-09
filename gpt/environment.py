from flow.envs.base_gpt import Env
import torch
from gym.spaces.box import Box
from gym.spaces import MultiBinary
from utils import order_vehicles

import numpy as np

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
}


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

    @property
    def action_space(self):
        """See class definition."""
        num_vehicles = len(self.k.vehicle.get_ids())
        return Box(
            low=0,
            high=self.env_params.additional_params['max_speed'],
            shape=(num_vehicles, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        vehs = len(self.k.vehicle.get_ids())
        obs_space = Box(low=-1, high=1, shape=(vehs,14*vehs))
            
        return obs_space

    def _apply_rl_actions(self, rl_actions, vehs):
        """See class definition."""
        self.k.vehicle.apply_velocity(vehs, rl_actions)

    def compute_reward(self, vehs, **kwargs):
        """See class definition."""
        ids = self.k.vehicle.get_ids()
        # collided_vehicles
        coll_veh = self.k.simulation.collided_vehicles()
        # successful_vehicles
        succ_veh = self.k.simulation.successful_vehicles()
        
        rewards = torch.tensor([])
        dones = torch.tensor([])
        
        for i in vehs:
            if i in ids:
                if i in coll_veh:
                    reward = torch.tensor([-100.0])
                    done = torch.tensor([1.0])
                elif i in succ_veh:
                    reward = torch.tensor([100.0])
                    done = torch.tensor([1.0])
                else:
                    reward = torch.tensor([-0.25])
                    done = torch.tensor([0.0])
            else:
                reward = torch.tensor([100.0])
                done = torch.tensor([1.0])
            
            rewards = torch.cat((rewards, reward))
            dones = torch.cat((dones, done))

        return rewards, dones

    def get_state(self):
        """See class definition."""
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        ids = self.k.vehicle.get_ids()
        state_dict = {}
        
        for q in ids:
            obs = []
            
            # POSITION
            pos = self.k.vehicle.get_2d_position(q)
            obs.append(np.clip((pos[0]-100)/100, -1, 1))
            obs.append(np.clip((pos[1]-100)/100, -1, 1))
            
            # VELOCITY
            vel = np.clip((self.k.vehicle.get_speed(q) - 13.9/2)/(13.9/2), -1, 1)
            obs.append(vel)
            
            # HEADING ANGLE
            angle = np.clip((self.k.vehicle.get_orientation(q)[2]-180)/180, -1, 1)
            obs.append(angle)
            
            # LANE, WAY AND QUEUE
            if self.k.vehicle.get_route(q) == '': # just to fix a simulator bug
                lane = [0.0, 0.0, 0.0]
                way = [0.0, 0.0, 0.0]
                queue = [0.0, 0.0, 0.0, 0.0]
            else:
                way = ways[self.k.vehicle.get_route(q)]
                lane = [way[2], way[1], way[0]]
                queue = queues[self.k.vehicle.get_route(q)[0]]
            
            obs = obs + lane + way + queue
            
            state_dict[q] = obs
            
        ord_vehs = order_vehicles(state_dict)
        state = torch.zeros((len(ord_vehs), 14*len(ord_vehs)))
        for k in range(len(ord_vehs)):
            ego_state = torch.as_tensor([state_dict[ord_vehs[k]]])
            for q in range(len(ord_vehs)):
                if k != q:
                    other_state = torch.as_tensor([state_dict[ord_vehs[q]]])
                    ego_state = torch.cat((ego_state, other_state), dim=1)
            state[k] = ego_state
        
        num_arrived = self.k.vehicle.get_num_arrived()
        if num_arrived > 0:
            if len(ids) > 0:
                aug_col = torch.zeros((len(ids), 14*num_arrived))
                aug_row = torch.zeros((num_arrived, 14*(state.shape[0]+num_arrived)))
                state = torch.cat((state, aug_col), dim=1)
                state = torch.cat((state, aug_row), dim=0)
            else:
                state = torch.zeros((num_arrived, 14*num_arrived))
                
        state = state.to(torch.float32)

        return state.to(device), ord_vehs
