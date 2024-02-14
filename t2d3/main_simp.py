from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
import numpy as np
import torch
from utils_simp import flow_params, trim, order_vehicles, evaluate, rl_actions
from memory import ReplayBuffer
from agent import TD3

num_eps = 10000 
total_steps = 0
best_return = -100
returns_list = []
ep_steps_list = []

state_dim = 15
action_dim = 1

memory = ReplayBuffer(state_dim, action_dim)

aim = TD3(
        state_dim,
        action_dim,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename='models/AIM_T2D3_simp')

ep_steps, returns = evaluate(aim, flow_params)
returns_list.append(returns)
ep_steps_list.append(ep_steps)
np.save('results/returns_simp.npy', returns_list)
np.save('results/ep_steps_simp.npy', ep_steps_list)

print('Training ep. number: {}, Avg. Ev. steps: {}, Avg. Ev. total return: {}'.format(0, ep_steps, returns))

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
    
    for j in range(max_ep_steps):    
        # actions: (V,) ordered tensor
        if total_steps > 25000:
            actions = aim.select_action(state.view(-1, 15).unsqueeze(dim=0))
            noise = torch.randn_like(actions) * 0.1
            actions = (actions + noise).clamp(-1, 1)
        else:
            actions = rl_actions(state)
        
        # next_state: (V, F) ordered tensor
        # reward: (1,) ordered tensor
        # done: (1,) ordered tensor
        # crash: boolean
        
        next_state, reward, not_done, crash = env.step(actions)
        
        if state.shape[0] > 0:
            memory.add(state, actions, next_state, reward, not_done)
            if total_steps > 25000:
                aim.train(memory)
            
        state = next_state
        state = trim(state)
          
        total_steps += 1
        
        if crash:
            break
        
    if total_steps > 25000:
        ep_steps, returns = evaluate(aim, flow_params)
        returns_list.append(returns)
        ep_steps_list.append(ep_steps)
        np.save('results/returns_simp.npy', returns_list)
        np.save('results/ep_steps_simp.npy', ep_steps_list)

        if returns > best_return:
            aim.save()
            best_return = returns

        print('Training ep. number: {}, Avg. Ev. steps: {}, Avg. Ev. total return: {}'.format(i+1, ep_steps, returns))
    env.terminate()
