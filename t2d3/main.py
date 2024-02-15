import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--random_speed", default=True, type=bool)      # Random speed for entering vehicles
parser.add_argument("--scenario", default="simple")                 # Intersection scenario considered
parser.add_argument("--memories", default=1, type=int)              # Number of replay buffers
parser.add_argument("--start_timesteps", default=25e3, type=int)    # Time steps initial random policy is used
parser.add_argument("--eval_freq", default=5, type=int)             # How often (episodes) we evaluate
parser.add_argument("--max_eps", default=10000, type=int)             # Max episodes to run environment
parser.add_argument("--expl_noise", default=0.1, type=float)        # Std of Gaussian exploration noise
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
if args.memories == 1:
    from memory import ReplayBuffer
    from agent import TD3
else:
    from memory_2mem import ReplayBuffer
    from agent_2mem import TD3

num_eps = args.max_eps 
total_steps = 0
best_return = -100
returns_list = []
ep_steps_list = []

state_dim = 12
action_dim = 1

memory = ReplayBuffer(state_dim, action_dim)
if args.memories == 2:
    memory_col = ReplayBuffer(state_dim, action_dim, max_size=int(2**19))

aim = TD3(
        state_dim,
        action_dim,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename=f'models/AIM_T2D3_{args.random_speed}_{args.scenario}_{args.memories}')

total_params = sum(p.numel() for p in aim.actor.parameters())
print(total_params)

#ep_steps, returns = evaluate(aim, flow_params)
#returns_list.append(returns)
#ep_steps_list.append(ep_steps)
#np.save(f'results/returns_{args.random_speed}_{args.scenario}_{args.memories}.npy', returns_list)
#np.save(f'results/ep_steps_{args.random_speed}_{args.scenario}_{args.memories}.npy', ep_steps_list)
#print('Training ep. number: {}, Avg. Ev. steps: {}, Avg. Ev. total return: {}'.format(0, ep_steps, returns))

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
    ep_steps = 0

    for j in range(max_ep_steps):    
        # actions: (V,) ordered tensor
        if total_steps > args.start_timesteps:
            actions = aim.select_action(state.view(-1, 12).unsqueeze(dim=0))
            noise = torch.randn_like(actions) * args.expl_noise
            actions = (actions + noise).clamp(-1, 1)
        else:
            actions = rl_actions(state)
        
        # next_state: (V, F) ordered tensor
        # reward: (1,) ordered tensor
        # done: (1,) ordered tensor
        # crash: boolean
        next_state, reward, not_done, crash = env.step(actions)
       
        if args.memories == 1:
            if state.shape[0] > 0:
                ep_steps += 1
                memory.add(state, actions, next_state, reward, not_done)
                if total_steps > args.start_timesteps:
                    aim.train(memory)
        else:
            if state.shape[0] > 0:
                ep_steps += 1
                if reward != -100:
                    memory.add(state, actions, next_state, reward, not_done)
                else:
                    memory_col.add(state, actions, next_state, reward, not_done)
                if total_steps > args.start_timesteps:
                    aim.train(memory, memory_col)
        
        if state.shape[0] == 0 and ep_steps > 0:
            break

        state = next_state
        state = trim(state)
          
        total_steps += 1
        
        if crash:
            break
        
    if total_steps > args.start_timesteps and ((i+1) % args.eval_freq == 0):
        ep_steps, returns = evaluate(aim, flow_params)
        returns_list.append(returns)
        ep_steps_list.append(ep_steps)
        np.save(f'results/returns_{args.random_speed}_{args.scenario}_{args.memories}.npy', returns_list)
        np.save(f'results/ep_steps_{args.random_speed}_{args.scenario}_{args.memories}.npy', ep_steps_list)

        if returns > best_return:
            aim.save()
            best_return = returns

        print('Training ep. number: {}, Avg. Ev. steps: {}, Avg. Ev. total return: {}'.format(i+1, ep_steps, returns))
    env.terminate()
