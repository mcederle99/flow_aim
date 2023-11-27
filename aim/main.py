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
           probability=0.05,
           #depart_speed="random",
          )
inflow.add(veh_type="rl",
           edge="t_c",
           probability=0.1,
           #depart_speed="random",
          )
inflow.add(veh_type="rl",
           edge="l_c",
           probability=0.1,
           #depart_speed="random",
          )
inflow.add(veh_type="rl",
           edge="r_c",
           probability=0.05,
           #depart_speed="random",
          )


sim_params = SumoParams(sim_step=0.1, render=False)

initial_config = InitialConfig()

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
net_params = NetParams(inflows=inflow, additional_params=additional_net_params)

def evaluate(aim, env, st):
    num_steps = 1000
    returns = []
    outflows = []
    eval_steps_list = []
    for i in range(10):
        ret = 0
    
        state = env.reset()
    
        veh_ids = env.k.vehicle.get_ids()
        edges, edges_type = compute_edges(env, state)
        nodes = {}
        for node in list(state.keys()):
            nodes[node] = state[node][:3]
        eval_steps = 0
  
        for j in range(num_steps):
        
            actions = aim.test_action(nodes, edges, edges_type)
        
            state_, reward, done, _ = env.step(rl_actions=actions.cpu().detach().numpy())

            veh_ids = env.k.vehicle.get_ids()
            edges_, edges_type_ = compute_edges(env, state_)
            nodes_ = {}
            for node in list(state_.keys()):
                nodes_[node] = state_[node][:3]
        
            proximity_reward = compute_rp(edges)
            w_p = 0.2
            reward += proximity_reward*w_p
        
            nodes = nodes_
            edges = edges_
            edges_type = edges_type_
        
            ret += reward
            eval_steps += 1
        
            if done:
                break
    
        # Store the information from the run in info_dict.
        outflow = env.k.vehicle.get_outflow_rate(int(eval_steps))
        outflows.append(outflow)
        returns.append(ret)
        eval_steps_list.append(eval_steps)

    print("Average return: {0}".format(np.mean(returns)))
    print("Average duration of the episode: {0}".format(np.mean(eval_steps_list)))
    print("Average outflow of the episode: {0}".format(np.mean(outflows)))
    print("Total training steps up to now: {0}".format(st))

    return np.mean(returns)

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
flow_params['env'].horizon = 5000

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
eval_returns = []

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

st = 0
best_eval_return = -1000

while not finished:
    ep_steps = 0
    ret = 0
    state = env.reset()
    
    veh_ids = env.k.vehicle.get_ids()
    edges, edges_type = compute_edges(env, state)
    nodes = {}
    for node in list(state.keys()):
        nodes[node] = state[node][:3]
    
    for j in range(num_steps):
        actions = aim.choose_action(nodes, edges, edges_type)
        
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
        
        if nodes != {}:
            aim.store_transition(nodes, edges, edges_type, actions, reward, nodes_, edges_, edges_type_, done)
            aim.learn()
        st += 1
        ep_steps += 1
        
        nodes = nodes_
        edges = edges_
        edges_type = edges_type_
        
        t1 = time.time()
        times.append(1 / (t1 - t0))
        
        ret += reward
        
        if done:
            break
        if st == 1000000:
            finished = True
            break
        if st % 5000 == 0 and st > 25000:
            print('EVALUATION RUN')
            eval_ret = evaluate(aim, env, st)
            if eval_ret > best_eval_return:
                aim.save_model('../TrainedModels/TD3')
                best_eval_return = eval_ret
                eval_returns.append(eval_ret)
                np.save('returns.npy', eval_returns)
                print('END EVALUATION')
            break
    
    # Save emission data at the end of every rollout. This is skipped
    # by the internal method if no emission path was specified.
    #if env.simulator == "traci":
    #    env.k.simulation.save_emission(run_id=i)

# Print the averages/std for all variables in the info_dict.
print("Total time:", time.time() - t)
print("steps/second:", np.mean(times))
env.terminate()
