from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
import numpy as np
import torch
from utils_batch import flow_params, trim, order_vehicles, choose_actions
from per_batch import PrioritizedReplayBuffer
from agent_batch import TD3

num_eps = 1000 
total_steps = 0
returns_list = []
returns_per_veh_list = []
ep_steps_list = []

state_dim = 15
action_dim = 1
max_action = 3

aim_straight = TD3(
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=4e-3,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename='models/batch/LSTM_AIM_straight')
aim_left = TD3(
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=4e-3,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename='models/batch/LSTM_AIM_left')
aim_right = TD3(
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=4e-3,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename='models/batch/LSTM_AIM_right')

aim_straight.load()
aim_left.load()
aim_right.load()

crash_counter = 0

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
        actions = choose_actions(state, aim_straight, aim_left, aim_right)
        # next_state: (V, F*V) ordered tensor
        # reward: (V,) ordered tensor
        # done: (V,) ordered tensor
        # crash: boolean
        
        next_state, reward, done, crash = env.step(actions*max_action)
         
        state = next_state
        state = trim(state)
        
        returns += sum(reward.tolist())
        ep_steps += 1
        
        if crash:
            crash_counter += 1
            break
        
    returns_list.append(returns)
    ep_steps_list.append(ep_steps)
    returns_per_veh = returns/sum(env.k.vehicle._num_departed)
    returns_per_veh_list.append(returns_per_veh)
    print('Episode number: {}, Episode steps: {}, Episode total return: {}, Returns per vehicle: {}'.format(i, ep_steps, returns, returns_per_veh))
#    np.save('results/returns.npy', returns_list)
#    np.save('results/ep_steps.npy', ep_steps_list)
#    np.save('results/returns_per_veh.npy', returns_per_veh_list)
    
    env.terminate()

print(crash_counter/1000*100)
