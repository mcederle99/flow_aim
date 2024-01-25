from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
import numpy as np
import torch
from utils import flow_params, trim, order_vehicles
from per import PrioritizedReplayBuffer
from agent_gru import TD3

num_eps = 1000 
total_steps = 0
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
        filename='GRU_AIM_straight')
aim_left = TD3(
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=4e-3,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename='GRU_AIM_left')
aim_right = TD3(
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=4e-3,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename='GRU_AIM_right')

def rl_actions(state):
    num = state.shape[0]
    actions = torch.randn((num,), device="cuda").clamp(-1, 1)
    return actions.detach().cpu()

# 8: right, 9: straight, 10: left
def choose_actions(state, aim_straight, aim_left, aim_right):
    actions = torch.tensor([])
    for i in range(state.shape[0]):
        if state[i][8] == 1.0:
            action = aim_right.select_action(state[i,:])
        elif state[i][9] == 1.0:
            action = aim_straight.select_action(state[i,:])
        elif state[i][10] == 1.0:
            action = aim_left.select_action(state[i,:])
        actions = torch.cat((actions, action))

    return actions

for i in range(num_eps):

    random_seed = np.random.choice(1000)
    sim_params = SumoParams(sim_step=0.25, render=False, seed=random_seed)
    flow_params['sim'] = sim_params
    # Get the env name and a creator for the environment.
    create_env, _ = make_create_env(flow_params)
    # Create the environment.
    env = create_env()
    max_ep_steps = env.env_params.horizon
    
    returns = 0
    ep_steps = 0
    
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
                elif state[k][9] == 1.0:
                    memory_straight.add(state[k,:], actions[k], reward[k], next_state[k,:], done[k])
                elif state[k][10] == 1.0:
                    memory_left.add(state[k,:], actions[k], reward[k], next_state[k,:], done[k])
        if total_steps % 20 == 0 and total_steps > 5000:
            aim_straight.train(memory_straight)
            aim_left.train(memory_left)
            aim_right.train(memory_right)
        
        state = next_state
        state = trim(state)
        
        returns += sum(reward.tolist())
        ep_steps += 1
        
        if crash:
            break
        
    returns_list.append(returns)
    ep_steps_list.append(ep_steps)
    returns_per_veh = returns/sum(env.k.vehicle._num_departed)
    returns_per_veh_list.append(returns_per_veh)
    print('Episode number: {}, Episode steps: {}, Episode total return: {}, Returns per vehicle: {}'.format(i, ep_steps, returns, returns_per_veh))
    np.save('results/returns_gru_fb.npy', returns_list)
    np.save('results/ep_steps_gru_fb.npy', ep_steps_list)
    np.save('results/returns_per_veh_gru_fb.npy', returns_per_veh_list)
    
    aim_straight.save()
    aim_left.save()
    aim_right.save()
    env.terminate()
