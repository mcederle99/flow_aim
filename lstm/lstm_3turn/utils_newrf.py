import numpy as np
import torch

def order_vehicles(state):
    distances = {}
    ordered_vehicles = []
    
    for veh in list(state.keys()):
        perturbation = 1e-10*np.random.randn()
        dist = np.sqrt(state[veh][0]**2 + state[veh][1]**2) + perturbation
        distances[dist] = veh
    
    for _ in list(state.keys()):
        min_dist = min(list(distances.keys()))
        ordered_vehicles.append(distances[min_dist])
        distances.pop(min_dist)
        
    return ordered_vehicles

def trim(state):
    if state.shape[0] > 0:
        while torch.sum(state[-1,:]) == 0:
            state = state[:-1,:state.shape[1]-15]
            if state.shape[0] == 0:
                break
        return state
    else:
        return state

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

from environment_newrf import ADDITIONAL_ENV_PARAMS
from scenario import ADDITIONAL_NET_PARAMS
from flow.core.params import EnvParams

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

from flow.core.params import TrafficLightParams

traffic_lights = TrafficLightParams()

from flow.core.params import InitialConfig

initial_config = InitialConfig()

from flow.core.params import VehicleParams

vehicles = VehicleParams()

from flow.controllers.rlcontroller import RLController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.core.params import SumoCarFollowingParams, SumoParams
from flow.utils.registry import make_create_env

vehicles.add("rl",
             acceleration_controller=(RLController, {}),
             routing_controller=(ContinuousRouter, {}),
             car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive"),
             num_vehicles=0)

from flow.core.params import InFlows

inflow_prob = 1/18

inflow = InFlows()

inflow.add(veh_type="rl",
           edge="t_c",
           depart_lane="best",
           #vehs_per_hour=200,
           #period=18,
           probability=inflow_prob
          )
inflow.add(veh_type="rl",
           edge="b_c",
           depart_lane="best",
           #vehs_per_hour=200,
           #period=18,
           probability=inflow_prob
          )
inflow.add(veh_type="rl",
           edge="r_c",
           depart_lane="best",
           #vehs_per_hour=200
           #period=18,
           probability=inflow_prob
          )
inflow.add(veh_type="rl",
           edge="l_c",
           depart_lane="best",
           #vehs_per_hour=200
           #period=18,
           probability=inflow_prob
          )

from flow.core.params import NetParams
from environment_newrf import SpeedEnv
from scenario import IntersectionNetwork

net_params = NetParams(inflows=inflow, additional_params=ADDITIONAL_NET_PARAMS)

flow_params = dict(
    exp_tag='test',
    env_name=SpeedEnv,
    network=IntersectionNetwork,
    simulator='traci',
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
    tls=traffic_lights,
)

# number of time steps
flow_params['env'].horizon = 1200

def evaluate(aim_straight, aim_left, aim_right, flow_params, num_eps=10):
    
    returns_list = []
    ep_steps_list = []
    returns_per_veh_list = []
    
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
            actions = choose_actions(state, aim_straight, aim_left, aim_right)

            # next_state: (V, F*V) ordered tensor
            # reward: (V,) ordered tensor
            # done: (V,) ordered tensor
            # crash: boolean

            state, reward, done, crash = env.step(actions*3)
            state = trim(state)

            returns += sum(reward.tolist())
            ep_steps += 1

            if crash:
                break

        returns_list.append(returns)
        ep_steps_list.append(ep_steps)
        returns_per_veh = returns/sum(env.k.vehicle._num_departed)
        returns_per_veh_list.append(returns_per_veh)
        
    return np.mean(ep_steps_list), np.mean(returns_list), np.mean(returns_per_veh_list)  
