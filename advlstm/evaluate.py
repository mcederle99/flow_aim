from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
import numpy as np
import torch
from utils import flow_params, trim, order_vehicles
from memory import ReplayBuffer
from agent import TD3

num_eps = 1000 
total_steps = 0

state_dim = 15
action_dim = 1
max_action = 3

aim = TD3(
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=4e-3,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename='models/LSTM_AIM')

#aim.load()

crash_counter = 0
for i in range(num_eps):

    random_seed = np.random.choice(1000)
    sim_params = SumoParams(sim_step=0.1, render=False, seed=random_seed)
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
        actions = aim.select_action(state.unsqueeze(dim=0))
        # next_state: (F*V) ordered tensor
        # reward: (V,) ordered tensor
        # done: (V,) ordered tensor
        # crash: boolean
        
        next_state, reward, done, crash = env.step(actions*max_action)
         
        state = next_state
        state = trim(state)
        #print(state)
        #print(actions)
        #print(reward)
        #print(done)
        #print('----------------------------')
        #input('')
        returns += sum(reward.tolist())
        ep_steps += 1
        
        if crash:
            crash_counter += 1
            break
        
    returns_per_veh = returns/sum(env.k.vehicle._num_departed)
    print('Episode number: {}, Episode steps: {}, Episode total return: {}, Returns per vehicle: {}'.format(i, ep_steps, returns, returns_per_veh))
   
    env.terminate()

print(crash_counter/1000*100)
