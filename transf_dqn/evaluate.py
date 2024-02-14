from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
import numpy as np
import torch
from utils_ddqn import flow_params, trim, order_vehicles, map_actions
from agent_ddqn import DQN

num_eps = 100
total_steps = 0
returns_list = []
returns_per_veh_list = []
ep_steps_list = []

state_dim = 15
action_dim = 1

aim = DQN(15, 1, epsilon=0.0, filename='models/ddqn/aim')
aim.load()

for i in range(num_eps):

    random_seed = np.random.choice(1000)
    sim_params = SumoParams(sim_step=0.1, render=True, seed=random_seed)
    flow_params['sim'] = sim_params
    # Get the env name and a creator for the environment.
    create_env, _ = make_create_env(flow_params)
    # Create the environment.
    env = create_env()
    max_ep_steps = env.env_params.horizon
    
    returns = 0
    ep_steps = 0
    
    # state is a 2-dim tensor
    state = env.reset() # (F*V) where V: number of vehicles and F: number of features of each vehicle 

    for j in range(max_ep_steps):    
        
        # actions: (V,) ordered tensor
        actions = aim.select_action(state)
        env_actions, actions = map_actions(actions)
        
        # next_state: (F*V) ordered tensor
        # reward: (V,) ordered tensor
        # done: (V,) ordered tensor
        # crash: boolean
        
        state, reward, done, crash = env.step(np.abs(env_actions)*10)
        state = trim(state)
        input("")
        if state.shape[0] > 0:
            ep_steps += 1
        elif ep_steps > 0:
            break
        returns += sum(reward.tolist())
        
#        if crash:
#            break
    
    returns_list.append(returns)
    ep_steps_list.append(ep_steps)
    returns_per_veh = returns/sum(env.k.vehicle._num_departed)
    returns_per_veh_list.append(returns_per_veh)
    print('Episode number: {}, Episode steps: {}, Episode total return: {}, Returns per vehicle: {}, Epsilon: {}'.format(i, ep_steps, returns, returns_per_veh, aim.epsilon))
   
    env.terminate()
