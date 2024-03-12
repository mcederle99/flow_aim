import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--initial_speed", default="random")      # Random speed for entering vehicles
parser.add_argument("--scenario", default="simple")                 # Intersection scenario considered
parser.add_argument("--memories", default=1, type=int)              # Number of replay buffers
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--dimension", default="small")
args = parser.parse_args()

from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
import numpy as np
import torch
if args.scenario == "simple":
    if args.initial_speed == "random":
        from utils_simp import flow_params, trim, order_vehicles, evaluate, rl_actions, map_actions
#    else:
#        from utils_simp_noran import flow_params, trim, order_vehicles, evaluate, rl_actions, map_actions
#else:
#    if args.initial_speed == "random":
#        from utils import flow_params, trim, order_vehicles, evaluate, rl_actions, map_actions
#    else:
#        from utils_noran import flow_params, trim, order_vehicles, evaluate, rl_actions, map_actions
if args.memories == 1:
    if args.dimension == "small":
        from TD3 import TD3
    else:
        from TD3_big import TD3
#else:
#    from memory_2mem import ReplayBuffer
#    from agent_2mem import TD3

# Set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = 9*4
action_dim = 4

aim = TD3(
        state_dim,
        action_dim,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename=f'models/AIM_TD3_{args.dimension}_{args.initial_speed}_{args.scenario}_{args.memories}_{args.seed}')

#aim.load()
returns_list = []

north_routes = np.array([('t_c', 'c_b'), ('t_c', 'c_l'), ('t_c', 'c_r')])
south_routes = np.array([('b_c', 'c_t'), ('b_c', 'c_l'), ('b_c', 'c_r')])
east_routes = np.array([('r_c', 'c_l'), ('r_c', 'c_t'), ('r_c', 'c_b')])
west_routes = np.array([('l_c', 'c_r'), ('l_c', 'c_t'), ('l_c', 'c_b')])

for i in range(10):

    routes_chosen = False

    sim_params = SumoParams(sim_step=0.25, render=True, seed=i)
    flow_params['sim'] = sim_params
    # Get the env name and a creator for the environment.
    create_env, _ = make_create_env(flow_params)
    # Create the environment.
    env = create_env()
    max_ep_steps = env.env_params.horizon
    returns = 0

    # state is a 2-dim tensor
    state = env.reset() 
    ep_steps = 0
    
    for j in range(max_ep_steps):    
        # actions: (V,) ordered tensor
        actions = aim.select_action(np.array(state))
        # next_state: (V, F) ordered tensor
        # reward: (1,) ordered tensor
        # done: (1,) ordered tensor
        # crash: boolean
        state, reward, not_done, crash = env.step(actions)
        
        if len(env.k.vehicle.get_ids()) > 0 and not routes_chosen:
            env.k.vehicle.choose_routes("rl_0", north_routes[np.random.randint(0, high=3)])
            env.k.vehicle.choose_routes("rl_1", south_routes[np.random.randint(0, high=3)])
            env.k.vehicle.choose_routes("rl_2", east_routes[np.random.randint(0, high=3)])
            env.k.vehicle.choose_routes("rl_3", west_routes[np.random.randint(0, high=3)])
            routes_chosen = True

        returns += reward
        ep_steps += 1

        if crash:
            break
    
    returns_list.append(returns)
    print(f"Average episode T: {ep_steps} Average reward: {returns.item():.3f}")
    print(len(env.k.vehicle.get_ids()))
    env.terminate()
print(sum(returns_list)/10)
