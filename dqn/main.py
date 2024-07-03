import numpy as np
import torch
from agent import DQNAgent
from memory import ReplayBuffer
from flow.controllers import ContinuousRouter, RLController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, VehicleParams
from flow.utils.registry import make_create_env
from intersection_network import IntersectionNetwork, ADDITIONAL_NET_PARAMS
from intersection_env import MyEnv, ADDITIONAL_ENV_PARAMS
from utils import compute_rp, eval_policy, get_inflows
import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

vehicles = VehicleParams()
vehicles.add(veh_id="rl",
             acceleration_controller=(RLController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=4,
             color='green')
sim_params = SumoParams(sim_step=0.1, render=False)
initial_config = InitialConfig()
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
additional_net_params = ADDITIONAL_NET_PARAMS.copy()
inflow_rate = 0.1
# net_params = NetParams(additional_params=additional_net_params, inflows=get_inflows(inflow_rate))
net_params = NetParams(additional_params=additional_net_params)

flow_params = dict(
    exp_tag='test_network',
    env_name=MyEnv,
    network=IntersectionNetwork,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

flow_params['env'].horizon = 1000
create_env, _ = make_create_env(flow_params)
env = create_env()

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)              # Sets PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=300, type=int) # Time steps initial random policy is used
parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
parser.add_argument("--epsilon", default=1.0, type=float)       # Initial exploration rate
parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
parser.add_argument("--eps_min", default=0.01)                  # Minimum exploration rate
parser.add_argument("--eps_dec", default=5e-7, type=float)      # Epsilon decrement
parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
args = parser.parse_args()

file_name = f"aim_dqn_{args.seed}_4"
print("---------------------------------------")
print(f"Seed: {args.seed}")
print("---------------------------------------")

if not os.path.exists("./results"):
    os.makedirs("./results")

if args.save_model and not os.path.exists("./models"):
    os.makedirs("./models")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = 3
edge_dim = 2
action_dim = 5

aim = DQNAgent(state_dim, edge_dim, action_dim, discount=args.discount, tau=args.tau, epsilon=args.epsilon,
               eps_min=args.eps_min, eps_dec=args.eps_dec)
memory = ReplayBuffer()

if args.load_model != "":
    policy_file = file_name if args.load_model == "default" else args.load_model
    aim.load(f"./models/{policy_file}")
    _, _, _ = eval_policy(aim, env, eval_episodes=10)
    env.terminate()
    raise KeyboardInterrupt

evaluations = []
ev, num_crashes, not_completed = eval_policy(aim, env, eval_episodes=10)
print(f"Inflow_rate: {inflow_rate}")
print("---------------------------------------")
evaluations.append(ev)
max_evaluations = evaluations[0]
num_steps = env.env_params.horizon
num_evaluations = 1

ep_steps = 0
ep_return = 0
ep_number = 0
state = env.reset()
while state.x is None:
    state, _, _, _ = env.step([])

for t in range(int(args.max_timesteps)):

    ep_steps += 1
    actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type)
    state_, reward, done, _ = env.step(rl_actions=actions)

    done_bool = float(done) if ep_steps < num_steps else 0.0
    if state_.x is None:
        memory.add(state, actions, state, reward, done_bool)
    else:
        # reward = compute_rp(state_, reward)
        memory.add(state, actions, state_, reward, done_bool)

    state = state_
    ep_return += reward

    if t >= args.start_timesteps:
        aim.train(memory, args.batch_size)

    while state.x is None and not done:
        state, _, done, _ = env.step([])
    if done:
        print(f"Total T: {t + 1} Episode Num: {ep_number + 1} Episode T: {ep_steps} Reward: {ep_return:.3f}"
              f" Epsilon: {aim.epsilon:.3f}")
        # Evaluate episode
        if (t + 1) >= args.eval_freq * num_evaluations:
            ev, num_crashes, not_completed = eval_policy(aim, env, eval_episodes=10)
            print(f"Inflow_rate: {inflow_rate}")
            print("---------------------------------------")
            evaluations.append(ev)
            np.save(f"./results/{file_name}", evaluations)
            if evaluations[-1] > max_evaluations and (num_crashes + not_completed) <= 1:
                if args.save_model:
                    aim.save(f"./models/{file_name}")
                max_evaluations = evaluations[-1]
            num_evaluations += 1
            # if num_crashes == 0 and evaluations[-1] > 50:
            #     env.terminate()
            #     inflow_rate *= 1.5
            #     net_params = NetParams(additional_params=additional_net_params, inflows=get_inflows(inflow_rate))
            #     flow_params['net'] = net_params
            #     create_env, _ = make_create_env(flow_params)
            #     env = create_env()

        # Reset environment
        state = env.reset()
        while state.x is None:
            state, _, _, _ = env.step([])
        ep_steps = 0
        ep_return = 0
        ep_number += 1

env.terminate()
print(inflow_rate)
