import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--initial_speed", default="random")      # Random speed for entering vehicles
parser.add_argument("--scenario", default="simple")                 # Intersection scenario considered
parser.add_argument("--memories", default=1, type=int)              # Number of replay buffers
parser.add_argument("--max_eps", default=10, type=int)             # Max episodes to run environment
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()

from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
import numpy as np
import torch
if args.scenario == "simple":
    if args.initial_speed == "random":
        from utils_simp import flow_params, trim, order_vehicles, evaluate, rl_actions, map_actions
    else:
        from utils_simp_noran import flow_params, trim, order_vehicles, evaluate, rl_actions, map_actions
else:
    if args.initial_speed == "random":
        from utils import flow_params, trim, order_vehicles, evaluate, rl_actions, map_actions
    else:
        from utils_noran import flow_params, trim, order_vehicles, evaluate, rl_actions, map_actions
if args.memories == 1:
    from agent import TD3
else:
    from agent_2mem import TD3

num_eps = args.max_eps 
total_steps = 0

state_dim = 14
action_dim = 11

aim = TD3(
        state_dim,
        action_dim,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename=f'models/AIM_T2D3_th_{args.initial_speed}_{args.scenario}_{args.memories}_{args.seed}')

aim.load()
aim.actor.eval()
total_crashes = 0

for i in range(num_eps):
    #random_seed = np.random.choice(1000)
    sim_params = SumoParams(sim_step=0.25, render=False, seed=i)
    flow_params['sim'] = sim_params
    # Get the env name and a creator for the environment.
    create_env, _ = make_create_env(flow_params)
    # Create the environment.
    env = create_env()
    max_ep_steps = env.env_params.horizon

    # state is a 2-dim tensor
    state = env.reset()
    ep_steps = 0
    returns = 0
    
    for j in range(max_ep_steps):    
        # actions: (V,) ordered tensor
        actions = aim.select_action(state.view(-1, state_dim).unsqueeze(dim=0))
        env_actions = map_actions(actions)
        # next_state: (V, F) ordered tensor
        # reward: (1,) ordered tensor
        # done: (1,) ordered tensor
        # crash: boolean
        next_state, reward, not_done, crash = env.step(env_actions)
        
        returns += reward

        if state.shape[0] > 0:
            ep_steps += 1

        elif ep_steps > 0:
            break

        state = next_state
        state = trim(state)
          
        total_steps += 1
        
        if crash:
            total_crashes += 1
            break
        
    print('Training ep. number: {}, Avg. Ev. steps: {}, Avg. Ev. total return: {}'.format(i+1, ep_steps, returns.item()))
    env.terminate()

print(total_crashes)
