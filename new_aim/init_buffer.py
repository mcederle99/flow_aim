from flow.core.params import VehicleParams
from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.utils.registry import make_create_env
from datetime import datetime
import logging
import time
import numpy as np
import torch

device = torch.device('cuda')

from intersection_network import IntersectionNetwork, ADDITIONAL_NET_PARAMS
from intersection_env import myEnv, ADDITIONAL_ENV_PARAMS
from utils import compute_edges, compute_rp
from memory import ReplayBuffer

vehicles = VehicleParams()

vehicles.add(veh_id="human",
             acceleration_controller=(IDMController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=0,
             color='green')

from flow.core.params import InFlows

inflow = InFlows()

inflow.add(veh_type="human",
           edge="b_c",
           probability=0.05,
           depart_speed="random",
          )
inflow.add(veh_type="human",
           edge="t_c",
           probability=0.1,
           depart_speed="random",
          )
inflow.add(veh_type="human",
           edge="l_c",
           probability=0.1,
           depart_speed="random",
          )
inflow.add(veh_type="human",
           edge="r_c",
           probability=0.05,
           depart_speed="random",
          )


sim_params = SumoParams(sim_step=0.1, render=False)

initial_config = InitialConfig()

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
net_params = NetParams(inflows=inflow, additional_params=additional_net_params)

flow_params = dict(
    exp_tag='test_network',
    env_name=myEnv,
    network=IntersectionNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

# number of time steps
flow_params['env'].horizon = 1000

# Get the env name and a creator for the environment.
create_env, _ = make_create_env(flow_params)

# Create the environment.
env = create_env()

logging.info(" Starting experiment {} at {}".format(
    env.network.name, str(datetime.utcnow())))

logging.info("Initializing environment.")

finished = False

num_steps = env.env_params.horizon

# raise an error if convert_to_csv is set to True but no emission
# file will be generated, to avoid getting an error at the end of the
# simulation
convert_to_csv = False
if convert_to_csv and env.sim_params.emission_path is None:
    raise ValueError(
        'The experiment was run with convert_to_csv set '
        'to True, but no emission file will be generated. If you wish '
        'to generate an emission file, you should set the parameter '
        'emission_path in the simulation parameters (SumoParams or '
        'AimsunParams) to the path of the folder where emissions '
        'output should be generated. If you do not wish to generate '
        'emissions, set the convert_to_csv parameter to False.')

# time profiling information
t = time.time()
times = []

replay_buffer = ReplayBuffer(size=10**6)

st = 0
max_veh = 0
while not finished:
    state = env.reset()
    
    veh_ids = env.k.vehicle.get_ids()
    edges, edges_type = compute_edges(env, state)
    nodes = {}
    for node in list(state.keys()):
        nodes[node] = state[node][:3]
    
    for j in range(num_steps):
        
        t0 = time.time()
        state_, reward, done, _ = env.step(None)

        veh_ids = env.k.vehicle.get_ids()
        if len(veh_ids) > max_veh:
            max_veh = len(veh_ids)
        edges_, edges_type_ = compute_edges(env, state_)
        nodes_ = {}
        for node in list(state_.keys()):
            nodes_[node] = state_[node][:3]
        
        actions = []
        for veh in veh_ids:
            actions.append(env.k.vehicle.get_realized_accel(veh))
        if len(list(nodes.keys())) > len(list(nodes_.keys())):
            dep_veh = len(list(nodes.keys())) - len(list(nodes_.keys()))
            for i in range(dep_veh):
                actions.append(5)
            
        if len(list(nodes.keys())) < len(list(nodes_.keys())):
            new_veh = len(list(nodes_.keys())) - len(list(nodes.keys()))
            actions = actions[:-new_veh]
        actions = torch.tensor(actions, device=device)
        actions = actions.unsqueeze(0)
        actions = actions.transpose(0, 1)

        Rp = compute_rp(edges)
        w_p = 0.2
        reward += Rp*w_p
        
        if nodes != {}:
            replay_buffer.add(nodes, edges, edges_type, actions, reward, nodes_, edges_, edges_type_, done)
        
        st += 1

        nodes = nodes_
        edges = edges_
        edges_type = edges_type_
        
        t1 = time.time()
        times.append(1 / (t1 - t0))
        
        if st == 25000:
            finished = True
            break
        if done:
            break

import pickle

# Save object to a file
def save_object_to_pickle(obj, filename):
    with open(filename, 'wb') as pickle_file:
        pickle.dump(obj, pickle_file)

# Load object from a file
def load_object_from_pickle(filename):
    with open(filename, 'rb') as pickle_file:
        obj = pickle.load(pickle_file)
    return obj

save_object_to_pickle(replay_buffer, 'replay_buffer.pkl')

env.terminate()
