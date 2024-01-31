from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
import numpy as np
import torch
from utils_batch import flow_params, trim, order_vehicles, evaluate, choose_actions, rl_actions
from per_batch import PrioritizedReplayBuffer
from agent_batch_2 import TD3

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

memory_straight = PrioritizedReplayBuffer(2**20)
memory_left = PrioritizedReplayBuffer(2**20)
memory_right = PrioritizedReplayBuffer(2**20)

aim_straight = TD3(
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=4e-3,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename='models/batch_2/LSTM_AIM_straight')
aim_left = TD3(
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=4e-3,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename='models/batch_2/LSTM_AIM_left')
aim_right = TD3(
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=4e-3,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename='models/batch_2/LSTM_AIM_right')

ep_steps, returns, returns_per_veh = evaluate(aim_straight, aim_left, aim_right, flow_params)
returns_list.append(returns)
ep_steps_list.append(ep_steps)
returns_per_veh_list.append(returns_per_veh)
np.save('results/batch_2/returns.npy', returns_list)
np.save('results/batch_2/ep_steps.npy', ep_steps_list)
np.save('results/batch_2/returns_per_veh.npy', returns_per_veh_list)

print('Training ep. number: {}, Avg. Ev. steps: {}, Avg. Ev. total return: {}, Avg. Ev. returns per vehicle: {}, Best ep. steps: {}'.format(0, ep_steps, returns, returns_per_veh, best_ep_steps))

for i in range(num_eps):

    random_seed = np.random.choice(1000)
    sim_params = SumoParams(sim_step=0.25, render=False, seed=random_seed)
    flow_params['sim'] = sim_params
    # Get the env name and a creator for the environment.
    create_env, _ = make_create_env(flow_params)
    # Create the environment.
    env = create_env()
    max_ep_steps = env.env_params.horizon
    
    #returns = 0
    #ep_steps = 0
    learn_steps_right = 0
    learn_steps_left = 0
    learn_steps_straight = 0

    # state is a 2-dim tensor
    state = env.reset() # (V, F*V) where V: number of vehicles and F: number of features of each vehicle 

    for j in range(max_ep_steps):    
        # actions: (V,) ordered tensor
        if total_steps > 5000:
            actions = choose_actions(state, aim_straight, aim_left, aim_right)
            noise = (
                torch.randn_like(actions) * 0.1).clamp(-0.5, 0.5)
            actions = (actions + noise).clamp(-1, 1)
        else:
            actions = rl_actions(state)
        
        # next_state: (V, F*V) ordered tensor
        # reward: (V,) ordered tensor
        # done: (V,) ordered tensor
        # crash: boolean
        
        next_state, reward, done, crash = env.step(actions*max_action)
         
        if state.shape[0] > 0:
           total_steps += 1
           for k in range(state.shape[0]):
                if state[k][8] == 1.0:
                    memory_right.add(state[k,:], actions[k], reward[k], next_state[k,:], done[k])
                    learn_steps_right += 1
                elif state[k][9] == 1.0:
                    memory_straight.add(state[k,:], actions[k], reward[k], next_state[k,:], done[k])
                    learn_steps_straight += 1
                elif state[k][10] == 1.0:
                    memory_left.add(state[k,:], actions[k], reward[k], next_state[k,:], done[k])
                    learn_steps_left += 1
                    
        state = next_state
        state = trim(state)
        
        #returns += sum(reward.tolist())
        #ep_steps += 1
        
        if crash:
            break
    
    if total_steps > 5000:
        aim_straight.train(memory_straight, learn_steps_straight)
        aim_left.train(memory_left, learn_steps_left)
        aim_right.train(memory_right, learn_steps_right)
    
        ep_steps, returns, returns_per_veh = evaluate(aim_straight, aim_left, aim_right, flow_params)
        returns_list.append(returns)
        ep_steps_list.append(ep_steps)
        #returns_per_veh = returns/sum(env.k.vehicle._num_departed)
        returns_per_veh_list.append(returns_per_veh)
        np.save('results/batch_2/returns.npy', returns_list)
        np.save('results/batch_2/ep_steps.npy', ep_steps_list)
        np.save('results/batch_2/returns_per_veh.npy', returns_per_veh_list)
        if ep_steps >= best_ep_steps and returns_per_veh >= best_return:
            aim_straight.save()
            aim_left.save()
            aim_right.save()
            best_ep_steps = ep_steps
            best_return = returns_per_veh

        print('Training ep. number: {}, Avg. Ev. steps: {}, Avg. Ev. total return: {}, Avg. Ev. returns per vehicle: {}, Best ep. steps: {}'.format(i, ep_steps, returns, returns_per_veh, best_ep_steps))
    env.terminate()
