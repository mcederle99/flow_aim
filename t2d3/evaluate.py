import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--random_speed", default=True, type=bool)      # Random speed for entering vehicles
parser.add_argument("--scenario", default="simple")                 # Intersection scenario considered
parser.add_argument("--memories", default=1, type=int)              # Number of replay buffers
parser.add_argument("--max_eps", default=10, type=int)             # Max episodes to run environment
args = parser.parse_args()

from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
import numpy as np
import torch
if args.scenario == "simple":
    if args.random_speed:
        from utils_simp import flow_params, trim, order_vehicles, evaluate, rl_actions
    else:
        from utils_simp_noran import flow_params, trim, order_vehicles, evaluate, rl_actions
else:
    if args.random_speed:
        from utils import flow_params, trim, order_vehicles, evaluate, rl_actions
    else:
        from utils_noran import flow_params, trim, order_vehicles, evaluate, rl_actions
from agent import TD3

num_eps = args.max_eps 

state_dim = 12
action_dim = 1

aim = TD3(
        state_dim,
        action_dim,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename=f'models/AIM_T2D3_{args.random_speed}_{args.scenario}_{args.memories}')

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
        actions = aim.select_action(state.view(-1, 12).unsqueeze(dim=0))
        
        # next_state: (V, F) ordered tensor
        # reward: (1,) ordered tensor
        # done: (1,) ordered tensor
        # crash: boolean
        state, reward, not_done, crash = env.step(actions)
        state = trim(state)
        
        if state.shape[0] > 0:
            ep_steps += 1
            returns += reward
        else:
            if ep_steps > 0:
                break

        if crash:
            break
        
    env.terminate()
    print('Episode steps: {}, Returns: {}'.format(ep_steps, returns))
