import numpy as np
import torch
from agent import TD3
from memory import ReplayBuffer
from flow.controllers import ContinuousRouter, RLController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, VehicleParams
from flow.utils.registry import make_create_env
from intersection_network import IntersectionNetwork, ADDITIONAL_NET_PARAMS
from intersection_env import MyEnv, ADDITIONAL_ENV_PARAMS
from utils import compute_rp, eval_policy
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
parser.add_argument("--start_timesteps", default=5e3, type=int)# Time steps initial random policy is used
parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
args = parser.parse_args()

file_name = f"aim_{args.seed}"
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
action_dim = 1
max_action = 5.0

aim = TD3(state_dim, edge_dim, action_dim, discount=args.discount, tau=args.tau, policy_noise=args.policy_noise,
          noise_clip=args.noise_clip, policy_freq=args.policy_freq, max_action=max_action)
memory = ReplayBuffer()

if args.load_model != "":
    policy_file = file_name if args.load_model == "default" else args.load_model
    aim.load(f"./models/{policy_file}")

evaluations = [eval_policy(aim, env)]
max_evaluations = evaluations[0]
num_steps = env.env_params.horizon
num_evaluations = 1

ep_steps = 0
ep_return = 0
ep_number = 0
state = env.reset()

for t in range(int(args.max_timesteps)):

    ep_steps += 1

    if t < args.start_timesteps:
        actions = env.action_space.sample()
    else:
        actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type)
        noise = np.random.normal(0.0, max_action * args.expl_noise, size=len(actions)).astype(np.float32)
        actions = (actions + noise).clip(-max_action, max_action)

    state_, reward, done, _ = env.step(rl_actions=actions)

    if state_.x is None:
        done_bool = 1.0
        memory.add(state, actions, state, reward, done_bool)
    else:
        reward = compute_rp(state_, reward)
        done_bool = float(done) if ep_steps < num_steps else 0.0
        memory.add(state, actions, state_, reward, done_bool)

    state = state_
    ep_return += reward

    if t >= args.start_timesteps:
        aim.train(memory, args.batch_size)

    if done or state.x is None:
        print(f"Total T: {t + 1} Episode Num: {ep_number + 1} Episode T: {ep_steps} Reward: {ep_return:.3f}")
        # Evaluate episode
        if (t + 1) >= args.eval_freq * num_evaluations:
            evaluations.append(eval_policy(aim, env))
            np.save(f"./results/{file_name}", evaluations)
            if evaluations[-1] > max_evaluations:
                if args.save_model:
                    aim.save(f"./models/{file_name}")
                max_evaluations = evaluations[-1]
            num_evaluations += 1
        # Reset environment
        state = env.reset()
        ep_steps = 0
        ep_return = 0
        ep_number += 1

env.terminate()
