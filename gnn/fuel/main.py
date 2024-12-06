import numpy as np
import torch
from flow.controllers import ContinuousRouter, RLController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, VehicleParams
from flow.utils.registry import make_create_env
from intersection_network import IntersectionNetwork, ADDITIONAL_NET_PARAMS
from intersection_env_new import MyEnv, ADDITIONAL_ENV_PARAMS
from utils import eval_policy, get_inflows, eval_policy_inflows  # , eval_policy_pareto
import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)              # Sets PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
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
parser.add_argument("--inflows", default="no")
parser.add_argument("--file_name", default="")
parser.add_argument("--omega_space", default="discrete")
parser.add_argument("--nn_architecture", default="base")
args = parser.parse_args()

if args.nn_architecture == "smart":
    from agent_smart import TD3
    from memory_smart import ReplayBuffer
else:
    from agent import TD3
    from memory import ReplayBuffer

vehicles = VehicleParams()
if args.inflows == "yes":
    vehicles.add(veh_id="rl",
                 acceleration_controller=(RLController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=0,
                 color='green')
else:
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
if args.inflows == "yes":
    net_params = NetParams(additional_params=additional_net_params, inflows=get_inflows(inflow_rate))
else:
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
env.nn_architecture = args.nn_architecture
env.omega_space = args.omega_space

file_name = f"aim_{args.seed}_{args.file_name}"
print("---------------------------------------")
print(f"Seed: {args.seed}")
print("---------------------------------------")

if not os.path.exists("./results"):
    os.makedirs("./results")

if args.save_model and not os.path.exists("./models"):
    os.makedirs("./models")

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.nn_architecture == "smart":
    state_dim = 3
else:
    state_dim = 4
edge_dim = 2
action_dim = 1
max_action = 5.0

aim = TD3(state_dim, edge_dim, action_dim, discount=args.discount, tau=args.tau, policy_noise=args.policy_noise,
          noise_clip=args.noise_clip, policy_freq=args.policy_freq, max_action=max_action)
memory = ReplayBuffer()

if args.load_model != "":
    policy_file = file_name if args.load_model == "default" else args.load_model
    aim.load(f"./models/{policy_file}")
    np.random.seed(np.random.randint(0, int(1e5)))
    if args.inflows == "yes":
        _, _ = eval_policy_inflows(aim, env, eval_episodes=10)
    else:
        _, _, _ = eval_policy(aim, env, eval_episodes=11, test=True, nn_architecture=args.nn_architecture, omega_space=args.omega_space)
    env.terminate()
    raise KeyboardInterrupt

np.random.seed(args.seed)

evaluations = []
if args.inflows == "yes":
    ev, num_crashes = eval_policy_inflows(aim, env, eval_episodes=10)
    print(f"Inflow_rate: {inflow_rate}")
    print("---------------------------------------")
else:
    ev, num_crashes, _ = eval_policy(aim, env, eval_episodes=11, nn_architecture=args.nn_architecture, omega_space=args.omega_space)

evaluations.append(ev)
max_evaluations = -1000
best_num_crashes = 5
num_steps = env.env_params.horizon
num_evaluations = 6
best_num_pareto_sol = 0

ep_steps = 0
ep_return = 0
ep_number = 0
veh_num = 4
state = env.reset()
while state.x is None:
    state, _, _, _ = env.step([])

for t in range(int(args.max_timesteps)):

    ep_steps += 1

    if t < args.start_timesteps:
        actions = env.action_space.sample()
    else:
        if args.nn_architecture == "smart":
            actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type,
                                        torch.tensor([[env.omega, 1 - env.omega]], dtype=torch.float, device=device).repeat(state.x.shape[0], 1))
        else:
            actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type)
        noise = np.random.normal(0.0, max_action * args.expl_noise, size=len(actions)).astype(np.float32)
        actions = (actions + noise).clip(-max_action, max_action)

    state_, reward, done, _ = env.step(rl_actions=actions)
    # while state_.x is None and not done:
    #     state_, _, done, _ = env.step([])

    done_bool = float(done) if ep_steps < num_steps else 0.0

    if state_.x is None:
        if args.nn_architecture == "smart":
            memory.add(state, actions, state, reward, done_bool, env.omega)
        else:
            memory.add(state, actions, state, reward, done_bool)
    else:
        # reward = compute_rp(state_, reward)
        if args.nn_architecture == "smart":
            memory.add(state, actions, state_, reward, done_bool, env.omega)
        else:
            memory.add(state, actions, state_, reward, done_bool)

    state = state_
    ep_return += reward

    if t >= args.start_timesteps:
        aim.train(memory, args.batch_size)

    # if state.x is None:
    if args.inflows == "no":
        # if ep_steps % 150 == 0:
        if state.x is None:
            done = True
            # we may need to put "best" instead of 0 as starting lane (aquarium)
            # env.k.vehicle.add("rl_{}".format(veh_num), "rl", "b_c", 0.0, "best", 0.0)
            # env.k.vehicle.add("rl_{}".format(veh_num + 1), "rl", "t_c", 0.0, "best", 0.0)
            # env.k.vehicle.add("rl_{}".format(veh_num + 2), "rl", "l_c", 0.0, "best", 0.0)
            # env.k.vehicle.add("rl_{}".format(veh_num + 3), "rl", "r_c", 0.0, "best", 0.0)
            # veh_num += 4
            # state, _, _, _ = env.step([])
    while state.x is None and not done:
        state, _, done, _ = env.step([])
    if done:
        print(f"Total T: {t + 1} Episode Num: {ep_number + 1} Episode T: {ep_steps} Reward: {ep_return:.3f}")
        # Evaluate episode
        if (t + 1) >= args.eval_freq * num_evaluations and t >= 30000:
            if args.inflows == "yes":
                ev, num_crashes = eval_policy_inflows(aim, env, eval_episodes=10)
                print(f"Inflow_rate: {inflow_rate}")
                print("---------------------------------------")
            else:
                ev, num_crashes, num_pareto_sol = eval_policy(aim, env, eval_episodes=11, nn_architecture=args.nn_architecture, omega_space=args.omega_space)
            evaluations.append(ev)
            if args.save_model:
                np.save(f"./results/{file_name}", evaluations)
            # if evaluations[-1] > max_evaluations and num_crashes <= best_num_crashes:
            if num_pareto_sol >= best_num_pareto_sol and num_crashes <= best_num_crashes and evaluations[-1] > max_evaluations:
                if args.save_model:
                    aim.save(f"./models/{file_name}")
                max_evaluations = evaluations[-1]
                best_num_pareto_sol = num_pareto_sol
                best_num_crashes = num_crashes
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
        # veh_num = 4

env.terminate()
