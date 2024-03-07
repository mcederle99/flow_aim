import argparse
import time 

parser = argparse.ArgumentParser()
parser.add_argument("--initial_speed", default="random")      # Random speed for entering vehicles
parser.add_argument("--scenario", default="simple")                 # Intersection scenario considered
parser.add_argument("--memories", default=1, type=int)              # Number of replay buffers
parser.add_argument("--start_timesteps", default=25e3, type=int)    # Time steps initial random policy is used
parser.add_argument("--eval_freq", default=5000, type=int)             # How often (episodes) we evaluate
parser.add_argument("--max_steps", default=1e6, type=int)             # Max episodes to run environment
parser.add_argument("--expl_noise", default=0.1, type=float)        # Std of Gaussian exploration noise
parser.add_argument("--seed", default=0, type=int)
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
    from memory import ReplayBuffer
    from TD3_big import TD3
#else:
#    from memory_2mem import ReplayBuffer
#    from agent_2mem import TD3

# Set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

num_steps = args.max_steps
total_steps = 0
best_return = -2000
returns_list = []
ep_steps_list = []

state_dim = 14*12
action_dim = 12

memory = ReplayBuffer(state_dim, action_dim)
#if args.memories == 2:
#    memory_col = ReplayBuffer(state_dim, action_dim, max_size=int(2**19))

aim = TD3(
        state_dim,
        action_dim,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename=f'models/AIM_TD3_big_{args.initial_speed}_{args.scenario}_{args.memories}_{args.seed}')

total_params = sum(p.numel() for p in aim.actor.parameters())
print(total_params)

ep_steps, returns = evaluate(aim, flow_params)
returns_list.append(returns)
ep_steps_list.append(ep_steps)
np.save(f'results/returns_big_{args.initial_speed}_{args.scenario}_{args.memories}_{args.seed}.npy', returns_list)
np.save(f'results/ep_steps_big_{args.initial_speed}_{args.scenario}_{args.memories}_{args.seed}.npy', ep_steps_list)
print(f"Total T: {total_steps} Training episodes: {0} Average episode T: {ep_steps} Average reward: {returns:.3f}")

for i in range(int(1e6)):

    sim_params = SumoParams(sim_step=0.25, render=False, seed=10+i)
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
        if total_steps > args.start_timesteps:
            actions = aim.select_action(np.array(state))
            noise = np.random.normal(0, args.expl_noise, size=action_dim)
            actions = (actions + noise).clip(-1, 1)
        else:
            actions = env.action_space.sample()
        
        # next_state: (V, F) ordered tensor
        # reward: (1,) ordered tensor
        # done: (1,) ordered tensor
        # crash: boolean
        next_state, reward, not_done, crash = env.step(actions)

        if args.memories == 1:
            if len(env.k.vehicle.get_ids()) > 0:
                ep_steps += 1
                memory.add(state, actions, next_state, reward, not_done)
                if total_steps > args.start_timesteps:
                    aim.train(memory)
#        else:
#            if state.shape[0] > 0:
#                ep_steps += 1
#                if reward != -100:
#                    memory.add(state, actions, next_state, reward, not_done)
#                else:
#                    memory_col.add(state, actions, next_state, reward, not_done)
#                if total_steps > args.start_timesteps:
#                    aim.train(memory, memory_col)
        
        state = next_state
        #state = trim(state)
          
        total_steps += 1
        returns += reward

        if total_steps % args.eval_freq == 0:
            ev_ep_steps, ev_returns = evaluate(aim, flow_params)
            returns_list.append(ev_returns)
            ep_steps_list.append(ev_ep_steps)
            np.save(f'results/returns_big_{args.initial_speed}_{args.scenario}_{args.memories}_{args.seed}.npy', returns_list)
            np.save(f'results/ep_steps_big_{args.initial_speed}_{args.scenario}_{args.memories}_{args.seed}.npy', ep_steps_list)
            
            if total_steps > args.start_timesteps and ev_returns > best_return:
                aim.save()
                best_return = ev_returns
            
            print('---------------------------------------')
            print(f"Total T: {total_steps} Training episodes: {i+1} Average episode T: {ev_ep_steps} Average reward: {ev_returns:.3f}")
            print('---------------------------------------')

        if len(env.k.vehicle.get_ids()) == 0 and ep_steps > 0:
            break
        if crash:
            break
    

    print(f"Total T: {total_steps} Training episodes: {i+1} Episode T: {ep_steps} Reward: {returns.item():.3f}")
    env.terminate()
    if total_steps >= 1e6:
        break
