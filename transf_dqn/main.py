from flow.core.params import SumoParams
from flow.utils.registry import make_create_env
import numpy as np
import torch
from utils import flow_params, trim, order_vehicles, map_actions
from memory import ReplayBuffer
from agent import DQN

num_eps = 10000
total_steps = 0
returns_list = []
returns_per_veh_list = []
ep_steps_list = []

state_dim = 15
action_dim = 1

memory = ReplayBuffer(15, 1)

aim = DQN(15, 1, filename='models/dqn/aim')

for i in range(num_eps):

    random_seed = np.random.choice(1000)
    sim_params = SumoParams(sim_step=0.1, render=False, seed=random_seed)
    flow_params['sim'] = sim_params
    # Get the env name and a creator for the environment.
    create_env, _ = make_create_env(flow_params)
    # Create the environment.
    env = create_env()
    max_ep_steps = env.env_params.horizon
    
    returns = 0
    ep_steps = 0
    
    # state is a 2-dim tensor
    state = env.reset() # (F*V) where V: number of vehicles and F: number of features of each vehicle 

    for j in range(max_ep_steps):    
        
        # actions: (V,) ordered tensor
        actions = aim.select_action(state)
        env_actions, actions = map_actions(actions)
        
        # next_state: (F*V) ordered tensor
        # reward: (V,) ordered tensor
        # done: (V,) ordered tensor
        # crash: boolean
        
        next_state, reward, done, crash = env.step(env_actions*3)
        
        if state.shape[0] > 0:
           total_steps += 1
           ep_steps += 1
           memory.add(state, actions, next_state, reward, done)
           aim.train(memory)
        elif ep_steps > 0:
            break
        
        state = next_state
        state = trim(state)
        
        returns += sum(reward.tolist())
        
        if crash:
            break
    
    returns_list.append(returns)
    ep_steps_list.append(ep_steps)
    returns_per_veh = returns/sum(env.k.vehicle._num_departed)
    returns_per_veh_list.append(returns_per_veh)
    print('Episode number: {}, Episode steps: {}, Episode total return: {}, Returns per vehicle: {}, Epsilon: {}'.format(i, ep_steps, returns, returns_per_veh, aim.epsilon))
    np.save('results/dqn/returns.npy', returns_list)
    np.save('results/dqn/ep_steps.npy', ep_steps_list)
    np.save('results/dqn/returns_per_veh.npy', returns_per_veh_list)
    aim.save()
    
    env.terminate()
