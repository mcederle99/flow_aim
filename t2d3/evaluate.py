from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
import numpy as np
import torch
from utils_simp import flow_params, trim, order_vehicles, evaluate, rl_actions
from agent import TD3

num_eps = 10 
total_steps = 0
best_return = -100
returns_list = []
ep_steps_list = []

state_dim = 15
action_dim = 1

aim = TD3(
        state_dim,
        action_dim,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename='models/AIM_T2D3_simp')

aim.load()
aim.actor.eval()

for i in range(num_eps):

    random_seed = np.random.choice(1000)
    sim_params = SumoParams(sim_step=0.25, render=False, seed=random_seed)
    flow_params['sim'] = sim_params
    # Get the env name and a creator for the environment.
    create_env, _ = make_create_env(flow_params)
    # Create the environment.
    env = create_env()
    max_ep_steps = env.env_params.horizon

    # state is a 2-dim tensor
    state = env.reset()
    
    returns = 0
    ep_steps = 0
    
    for j in range(max_ep_steps):    
        # actions: (V,) ordered tensor
        actions = aim.select_action(state.view(-1, 15).unsqueeze(dim=0))
        
        # next_state: (V, F) ordered tensor
        # reward: (1,) ordered tensor
        # done: (1,) ordered tensor
        # crash: boolean
        
        state, reward, not_done, crash = env.step(actions)
        
        state = trim(state)
          
        ep_steps += 1
        returns += reward

        if crash:
            break
        
    env.terminate()
    print('Episode steps: {}, Returns: {}'.format(ep_steps, returns))
