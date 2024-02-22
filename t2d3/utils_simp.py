import numpy as np
import torch
from flow.core.params import SumoCarFollowingParams, SumoParams
from flow.utils.registry import make_create_env

state_dim = 12

def order_vehicles(state):
    distances = {}
    ordered_vehicles = []
    
    for veh in list(state.keys()):
        perturbation = 1e-6*np.random.randn()
        dist = np.sqrt(state[veh][0]**2 + state[veh][1]**2) + perturbation
        distances[dist] = veh
    
    for _ in list(state.keys()):
        min_dist = min(list(distances.keys()))
        ordered_vehicles.append(distances[min_dist])
        distances.pop(min_dist)
        
    return ordered_vehicles

def trim(state):
    if state.shape[0] > 0:
        while torch.sum(state[-state_dim]) == 0:
            state = state[:-state_dim]
            if state.shape[0] == 0:
                break
        return state
    else:
        return state
    
def rl_actions(state):
    num = state.shape[0] // state_dim
    actions = torch.randint(0, 11, (num,), device="cuda")
    return actions.detach().cpu()

def map_actions(actions):
    discretization = np.linspace(0, 1, num=11)
    dim = actions.shape[0]
    env_actions = np.zeros(dim)
    for i in range(dim):
        env_actions[i] = discretization[actions[i]]

    return env_actions

def evaluate(aim, flow_params, num_eps=10):
    aim.actor.eval()
    returns_list = []
    ep_steps_list = []
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
        state = env.reset() # (V, F) where V: number of vehicles and F: number of features of each vehicle 

        for j in range(max_ep_steps):

            # actions: (V,) ordered tensor
            actions = aim.select_action(state.view(-1, state_dim).unsqueeze(dim=0))
            env_actions = map_actions(actions) 
            # next_state: (V, F*V) ordered tensor
            # reward: (V,) ordered tensor
            # done: (V,) ordered tensor
            # crash: boolean
            state, reward, not_done, crash = env.step(torch.tensor(env_actions))
            state = trim(state)
            
            returns += reward
            ep_steps += 1

            if crash:
                break
        
        returns_list.append(returns)
        ep_steps_list.append(ep_steps)
        env.terminate()
        
    aim.actor.train()
    return np.mean(ep_steps_list), np.mean(returns_list)

from environment_simp import ADDITIONAL_ENV_PARAMS
from scenario_simp import ADDITIONAL_NET_PARAMS
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
from flow.controllers.car_following_models import IDMController

vehicles.add("rl",
             acceleration_controller=(RLController, {}),
             routing_controller=(ContinuousRouter, {}),
             car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive"),
             num_vehicles=4,
             )

from flow.core.params import InFlows

inflow_prob = 1/18

inflow = InFlows()

inflow.add(veh_type="rl",
           edge="t_c",
           #depart_lane="best",
           depart_speed="random",
           #vehs_per_hour=200,
           period=1200,
           #probability=inflow_prob,
           end=5
          )
inflow.add(veh_type="rl",
           edge="b_c",
           #depart_lane="best",
           depart_speed="random",
           #vehs_per_hour=200,
           period=1200,
           #probability=inflow_prob
           end=5
          )
inflow.add(veh_type="rl",
           edge="r_c",
           #depart_lane="best",
           depart_speed="random",
           #vehs_per_hour=200
           period=1200,
           #probability=inflow_prob
           end=5
          )
inflow.add(veh_type="rl",
           edge="l_c",
           #depart_lane="best",
           depart_speed="random",
           #vehs_per_hour=200
           period=1200,
           #probability=inflow_prob
           end=5
          )

from flow.core.params import NetParams
from environment_simp import SpeedEnv
from scenario_simp import IntersectionNetwork

net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)

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
