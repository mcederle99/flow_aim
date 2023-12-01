from flow.core.params import VehicleParams
from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.controllers import RLController
from flow.utils.registry import make_create_env
from datetime import datetime
import logging
import time
import numpy as np

from intersection_network import IntersectionNetwork, ADDITIONAL_NET_PARAMS
from intersection_env import myEnv, ADDITIONAL_ENV_PARAMS
from utils import compute_edges, compute_rp
from intersection_agent import AIM
from memory import ReplayBuffer
from networks import Actor, Critic
import torch

#torch.cuda.set_device(2)
device = torch.device('cuda')

vehicles = VehicleParams()

vehicles.add(veh_id="rl",
             acceleration_controller=(RLController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=0,
             color='green')

from flow.core.params import InFlows

inflow = InFlows()

inflow.add(veh_type="rl",
           edge="b_c",
           probability=0.1,
           #depart_speed="random",
          )
inflow.add(veh_type="rl",
           edge="t_c",
           probability=0.05,
           #depart_speed="random",
          )
inflow.add(veh_type="rl",
           edge="l_c",
           probability=0.05,
           #depart_speed="random",
          )
inflow.add(veh_type="rl",
           edge="r_c",
           probability=0.1,
           #depart_speed="random",
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
flow_params['env'].horizon = 500

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

# used to store
returns = []
episode_lengths = []

# time profiling information
t = time.time()
times = []

# RL agent initialization - inizio
actor = Actor()
critic_1 = Critic()
critic_2 = Critic()

lr = 1e-4
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
critic_optimizer_1 = torch.optim.Adam(critic_1.parameters(), lr=lr)
critic_optimizer_2 = torch.optim.Adam(critic_2.parameters(), lr=lr)

explore_noise = 0.1

replay_buffer = ReplayBuffer(size=10**6)

gamma = 0.9

warmup = 25000
# RL agent initialization - fine

aim = AIM(actor,
          actor_optimizer,
          critic_1,
          critic_optimizer_1,
          critic_2,
          critic_optimizer_2,
          explore_noise,
          warmup,
          replay_buffer,
          batch_size=256,
          update_interval=1000,
          update_interval_actor=2,
          target_update_interval=1000,
          soft_update_tau=0.01,
          n_steps=1,
          gamma=gamma,
          model_name='AIM_model')

aim.load_model('../TrainedModels/TD3_more_vehicles')

st = 0
good_eps = 0
for i in range(100):
    ep_steps = 0
    ret = 0
    state = env.reset()
    
    veh_ids = env.k.vehicle.get_ids()
    edges, edges_type = compute_edges(env, state)
    nodes = {}
    for node in list(state.keys()):
        nodes[node] = state[node][:3]
    
    for j in range(num_steps):
        actions = aim.test_action(nodes, edges, edges_type)
        
        t0 = time.time()
        state_, reward, done, _ = env.step(rl_actions=actions.cpu().detach().numpy())

        veh_ids = env.k.vehicle.get_ids()
        edges_, edges_type_ = compute_edges(env, state_)
        nodes_ = {}
        for node in list(state_.keys()):
            nodes_[node] = state_[node][:3]
        
        Rp = compute_rp(edges)
        w_p = 0.2
        reward += Rp*w_p
        
        st += 1
        ep_steps += 1
        
        nodes = nodes_
        edges = edges_
        edges_type = edges_type_
        
        t1 = time.time()
        times.append(1 / (t1 - t0))
        
        ret += reward
        
        if done:
            if ep_steps == num_steps:
                good_eps += 1
            #print(ret)
            #print(ep_steps)
            break

    returns.append(ret)
    episode_lengths.append(ep_steps)

print(np.mean(returns))
print(good_eps)
    # Save emission data at the end of every rollout. This is skipped
    # by the internal method if no emission path was specified.
    #if env.simulator == "traci":
    #    env.k.simulation.save_emission(run_id=i)

# Print the averages/std for all variables in the info_dict.
#print("Total time:", time.time() - t)
#print("steps/second:", np.mean(times))
env.terminate()
