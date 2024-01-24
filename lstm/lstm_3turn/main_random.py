from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
import numpy as np
import torch
from utils import flow_params, trim, order_vehicles

num_eps = 1000 
total_steps = 0
returns_list = []
returns_per_veh_list = []
ep_steps_list = []

state_dim = 15
action_dim = 1
max_action = 1.5

def rl_actions(state):
    num = state.shape[0]
    actions = torch.randn((num,), device="cuda").clamp(-1, 1)
    return actions.detach().cpu()

for i in range(num_eps):

    random_seed = np.random.choice(1000)
    sim_params = SumoParams(sim_step=0.25, render=False, seed=random_seed)
    flow_params['sim'] = sim_params
    # Get the env name and a creator for the environment.
    create_env, _ = make_create_env(flow_params)
    # Create the environment.
    env = create_env()
    max_ep_steps = env.env_params.horizon
    
    returns = 0
    ep_steps = 0
    
    # state is a 2-dim tensor
    state = env.reset() # (V, F*V) where V: number of vehicles and F: number of features of each vehicle 

    for j in range(max_ep_steps):    
        # actions: (V,) ordered tensor
        actions = rl_actions(state)
        
        # next_state: (V, F*V) ordered tensor
        # reward: (V,) ordered tensor
        # done: (V,) ordered tensor
        # crash: boolean
        
        next_state, reward, done, crash = env.step(actions*max_action + max_action)
         
        state = next_state
        state = trim(state)
        
        returns += sum(reward.tolist())
        ep_steps += 1
        
        if crash:
            break
        
    returns_list.append(returns)
    ep_steps_list.append(ep_steps)
    returns_per_veh = returns/sum(env.k.vehicle._num_departed)
    returns_per_veh_list.append(returns_per_veh)
    print('Episode number: {}, Episode steps: {}, Episode total return: {}, Returns per vehicle: {}'.format(i, ep_steps, returns, returns_per_veh))
    np.save('results/returns_random_bigr.npy', returns_list)
    np.save('results/ep_steps_random_bigr.npy', ep_steps_list)
    np.save('results/returns_per_veh_random_bigr.npy', returns_per_veh_list)
    
    env.terminate()