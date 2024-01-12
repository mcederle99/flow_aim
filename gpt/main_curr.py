from flow.utils.registry import make_create_env
import numpy as np
import torch
from utils import trim, order_vehicles, variance
from per import PrioritizedReplayBuffer
from agent import TD3

from environment import ADDITIONAL_ENV_PARAMS
from scenario import ADDITIONAL_NET_PARAMS
from flow.core.params import EnvParams, SumoParams, TrafficLightParams, InitialConfig, VehicleParams, SumoCarFollowingParams, InFlows, NetParams
from flow.envs.ring.accel import AccelEnv
from flow.controllers.rlcontroller import RLController
from flow.controllers.routing_controllers import ContinuousRouter
from environment import SpeedEnv
from scenario import IntersectionNetwork

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
random_seed = np.random.choice(1000)
sim_params = SumoParams(sim_step=0.25, render=False, seed=random_seed)
traffic_lights = TrafficLightParams()
initial_config = InitialConfig()
vehicles = VehicleParams()

vehicles.add("rl",
             acceleration_controller=(RLController, {}),
             routing_controller=(ContinuousRouter, {}),
             car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive"),
             num_vehicles=0)

inflow = InFlows()

inflow.add(veh_type="rl",
           edge="t_c",
           depart_lane="best",
           #vehs_per_hour=200,
           #period=18,
           probability=1/36
          )
inflow.add(veh_type="rl",
           edge="b_c",
           depart_lane="best",
           #vehs_per_hour=200,
           #period=18,
           probability=1/36
          )
inflow.add(veh_type="rl",
           edge="r_c",
           depart_lane="best",
           #vehs_per_hour=200
           #period=18,
           probability=1/36
          )
inflow.add(veh_type="rl",
           edge="l_c",
           depart_lane="best",
           #vehs_per_hour=200
           #period=18,
           probability=1/36
          )

net_params = NetParams(inflows=inflow, additional_params=ADDITIONAL_NET_PARAMS)

flow_params = dict(
    exp_tag='test',
    env_name=SpeedEnv,
    network=IntersectionNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
    tls=traffic_lights,
)

# number of time steps
flow_params['env'].horizon = 1200

ep = 0
flow_hour = 100

max_ep_steps = flow_params['env'].horizon
total_steps = 0
returns_list = []
ep_steps_list = []
returns_per_veh_list = []

state_dim = 14
action_dim = 1
max_action = 13.9/2

memory = PrioritizedReplayBuffer(2**20)
aim = TD3(
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=4e-3,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        filename='LSTM_AIM')

def rl_actions(state):
    num = state.shape[0]
    actions = torch.randn((num,), device="cuda").clamp(-1, 1)
    return actions.detach().cpu()


while ep < 8000:
    
    random_seed = np.random.choice(1000) 
    sim_params = SumoParams(sim_step=0.25, render=False, seed=random_seed)
    flow_params['sim'] = sim_params
    inflow = InFlows()

    inflow.add(veh_type="rl",
               edge="t_c",
               depart_lane="best",
               #vehs_per_hour=200,
               #period=18,
               probability=flow_hour/3600
              )
    inflow.add(veh_type="rl",
               edge="b_c",
               depart_lane="best",
               #vehs_per_hour=200,
               #period=18,
               probability=flow_hour/3600
              )
    inflow.add(veh_type="rl",
               edge="r_c",
               depart_lane="best",
               #vehs_per_hour=200
               #period=18,
               probability=flow_hour/3600
              )
    inflow.add(veh_type="rl",
               edge="l_c",
               depart_lane="best",
               #vehs_per_hour=200
               #period=18,
               probability=flow_hour/3600
              )

    net_params = NetParams(inflows=inflow, additional_params=ADDITIONAL_NET_PARAMS)
    flow_params['net'] = net_params
   

    # Get the env name and a creator for the environment.
    create_env, _ = make_create_env(flow_params)
    # Create the environment.
    env = create_env()

    num_eps = 8000 - ep

    for i in range(num_eps):
        returns = 0
        ep_steps = 0
        
        # state is a 2-dim tensor
        state = env.reset() # (V, F*V) where V: number of vehicles and F: number of features of each vehicle 

        for j in range(max_ep_steps):    

            # actions: (V,) ordered tensor
            if total_steps > 10000:
                actions = aim.select_action(state)
                noise = (
                    torch.randn_like(actions) * 0.1).clamp(-0.5, 0.5)
                actions = (actions + noise).clamp(-1, 1)
            else:
                actions = rl_actions(state)
            
            # next_state: (V, F*V) ordered tensor
            # reward: (V,) ordered tensor
            # done: (V,) ordered tensor
            # crash: boolean
            
            next_state, reward, done, crash = env.step(actions*max_action + max_action)
            
            if state.shape[0] > 0:
               total_steps += 1
               for k in range(state.shape[0]):
                    memory.add(state[k,:], actions[k], reward[k], next_state[k,:], done[k])
            if total_steps % 20 == 0 and total_steps > 10000:
                aim.train(memory)

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
        ret_var = -1
        if i >= 50:
            ret_var = variance(returns_per_veh_list[-50:])
        print('Episode number: {}, Episode steps: {}, Episode return: {}, Current flow: {}, Returns variance: {}, ReturnxVeh: {}'.format(i, ep_steps, returns, flow_hour, ret_var, returns_per_veh))
        np.save('results/returns_curr_100.npy', returns_list)
        np.save('results/ep_steps_curr_100.npy', ep_steps_list)
        np.save('results/returns_per_veh_curr_100.npy', returns_per_veh_list)

        if i >= 50 and ret_var <= 0.01*flow_hour:
            flow_hour += 50
            break

        aim.save()

    aim.save()
    env.terminate()
