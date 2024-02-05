from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
import numpy as np
import torch
from utils_oldrf import flow_params, trim, order_vehicles, evaluate
from memory import ReplayBuffer
from agent import TD3

num_eps = 1000
total_steps = 0
best_ep_steps = 0
best_return = -100
returns_list = []
returns_per_veh_list = []
ep_steps_list = []

state_dim = 15
action_dim = 1
max_action = 3

memory = ReplayBuffer(15, 1)

aim = TD3(
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=4e-3,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename='models/LSTM_AIM_oldrf')

def rl_actions(state):
    num = state.shape[0] // 15
    actions = torch.randn((num,), device="cuda").clamp(-1, 1)
    return actions.detach().cpu()

ep_steps, returns, returns_per_veh = evaluate(aim, flow_params)
returns_list.append(returns)
ep_steps_list.append(ep_steps)
returns_per_veh_list.append(returns_per_veh)
np.save('results/returns_oldrf.npy', returns_list)
np.save('results/ep_steps_oldrf.npy', ep_steps_list)
np.save('results/returns_per_veh_oldrf.npy', returns_per_veh_list)

print('Training ep. number: {}, Avg. Ev. steps: {}, Avg. Ev. total return: {}, Avg. Ev. returns per vehicle: {}, Best ep. steps: {}'.format(0, ep_steps, returns, returns_per_veh, best_ep_steps))

for i in range(num_eps):

    random_seed = np.random.choice(1000)
    sim_params = SumoParams(sim_step=0.1, render=False, seed=random_seed)
    flow_params['sim'] = sim_params
    # Get the env name and a creator for the environment.
    create_env, _ = make_create_env(flow_params)
    # Create the environment.
    env = create_env()
    max_ep_steps = env.env_params.horizon
    
    #returns = 0
    #ep_steps = 0
    learn_steps = 0
    
    # state is a 2-dim tensor
    state = env.reset() # (F*V) where V: number of vehicles and F: number of features of each vehicle 

    for j in range(max_ep_steps):    
        
        # actions: (V,) ordered tensor
        if total_steps > 5000:
            actions = aim.select_action(state.unsqueeze(dim=0))
            noise = (
                torch.randn_like(actions) * 0.1).clamp(-0.5, 0.5)
            actions = (actions + noise).clamp(-1, 1)
        else:
            actions = rl_actions(state)
        
        # next_state: (F*V) ordered tensor
        # reward: (V,) ordered tensor
        # done: (V,) ordered tensor
        # crash: boolean
        
        next_state, reward, done, crash = env.step(actions*max_action)
        
        if state.shape[0] > 0:
           total_steps += 1
           learn_steps += 1
           memory.add(state, actions, next_state, reward, done)
        
        state = next_state
        state = trim(state)
        
        #returns += sum(reward.tolist())
        #ep_steps += 1
        
        if crash:
            break

    if total_steps > 5000:
        aim.train(memory, learn_steps)
    
    ep_steps, returns, returns_per_veh = evaluate(aim, flow_params)
    returns_list.append(returns)
    ep_steps_list.append(ep_steps)
    #returns_per_veh = returns/sum(env.k.vehicle._num_departed)
    returns_per_veh_list.append(returns_per_veh)
    np.save('results/returns_oldrf.npy', returns_list)
    np.save('results/ep_steps_oldrf.npy', ep_steps_list)
    np.save('results/returns_per_veh_oldrf.npy', returns_per_veh_list)
    if best_ep_steps < 3000:
        if ep_steps >= best_ep_steps:
            aim.save()
            best_ep_steps = ep_steps
            best_return = returns_per_veh
    else:
        if ep_steps == 3000 and returns_per_veh > best_return:
            aim.save()
            best_return = returns_per_veh

    print('Training ep. number: {}, Avg. Ev. steps: {}, Avg. Ev. total return: {}, Avg. Ev. returns per vehicle: {}, Best ep. steps: {}'.format(i, ep_steps, returns, returns_per_veh, best_ep_steps))
    env.terminate()
