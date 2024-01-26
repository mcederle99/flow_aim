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
        while torch.sum(state[-15]) == 0:
            state = state[:-15]
            if state.shape[0] == 0:
                break
        return state
    else:
        return state

from environment import ADDITIONAL_ENV_PARAMS
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
from flow.core.params import SumoCarFollowingParams

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
from environment import SpeedEnv
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
flow_params['env'].horizon = 3000
