from flow.core.params import VehicleParams
from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, SumoCarFollowingParams
from flow.controllers import RLController
from flow.utils.registry import make_create_env
from datetime import datetime
import logging
import time
import numpy as np
import torch
import argparse
import os

import utils
from td3 import TD3

from intersection_network import IntersectionNetwork, ADDITIONAL_NET_PARAMS
from intersection_env import myEnv, ADDITIONAL_ENV_PARAMS
from utils import compute_edges, compute_rp, compute_connections, compute_augmented_state
from memory import ReplayBuffer

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, params, eval_episodes=1):
	avg_reward = 0.
	for _ in range(eval_episodes):
		create_env, _ = make_create_env(params)
		env = create_env()
		state = env.reset()
		done = False
		while not done:
			connections = compute_connections(env, state)

			actions_for_env = []
			num_conn = []
			if len(list(state.keys())) > 0:
				for i, ids in enumerate(list(state.keys())):
					aug_state = compute_augmented_state(env, state, ids, connections[ids])
					num_conn.append((len(aug_state)-3)/4 - 1)
					if num_conn[i] == -1:
						action = max_action
					else:
						action = policy[int(num_conn[i])].select_action(np.array(aug_state))
					actions_for_env.append(action)
				state, reward_list, done, _ = env.step(actions_for_env, state)
				avg_reward += sum(reward_list)
			else:
				state, reward_list, done, _ = env.step([], state)
		avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
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
	parser.add_argument("--save_model", default=True, type=bool)        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--env", default="intersection")
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	vehicles = VehicleParams()
	vehicles.add(veh_id="rl",
                     acceleration_controller=(RLController, {}),
                     routing_controller=(ContinuousRouter, {}),
                     #car_following_params=SumoCarFollowingParams(speed_mode='obey_safe_speed'),
                     num_vehicles=2,
                     color='green')

	from flow.core.params import InFlows
        
	inflow = InFlows()

	inflow.add(veh_type="rl",
                   edge="b_c",
                   probability=0.1,
                   depart_speed="random")
	inflow.add(veh_type="rl",
                   edge="t_c",
                   probability=0.1,
                   depart_speed="random")
	inflow.add(veh_type="rl",
                   edge="l_c",
                   probability=0.1,
                   depart_speed="random")
	inflow.add(veh_type="rl",
                   edge="r_c",
                   probability=0.1,
                   depart_speed="random")
        
	sim_params = SumoParams(sim_step=0.1, render=False, seed=args.seed)
	sim_params_eval = SumoParams(sim_step=0.1, render=False, seed=args.seed+100)

	initial_config = InitialConfig()

	env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

	additional_net_params = ADDITIONAL_NET_PARAMS.copy()
	net_params = NetParams(additional_params=additional_net_params)

	flow_params = dict(
                exp_tag='train_network',
                env_name=myEnv,
                network=IntersectionNetwork,
                simulator='traci',
                sim=sim_params,
                env=env_params,
                net=net_params,
                veh=vehicles,
                initial=initial_config,
	)
	flow_params['env'].horizon = 500
	flow_params_eval = dict(
                exp_tag='eval_network',
                env_name=myEnv,
                network=IntersectionNetwork,
                simulator='traci',
                sim=sim_params_eval,
                env=env_params,
                net=net_params,
                veh=vehicles,
                initial=initial_config,
	)
	flow_params_eval['env'].horizon = 500

	create_env, _ = make_create_env(flow_params)
	env = create_env()
        
	# Set seeds
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
        
	state_dim = []
	action_dim = 1
	max_action = 3

	kwargs = {
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	policy = []
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		for i in range(4):
			state_dim.append(3+4*(i+1))
			policy.append(TD3(state_dim[i], **kwargs))

	if args.load_model != "":
		for i in range(4):
			policy_file = file_name + '_' + i if args.load_model == "default" else args.load_model
			policy.load(f"./models/{policy_file}")

	replay_buffer = []
	for i in range(4):
		replay_buffer.append(ReplayBuffer(state_dim[i], action_dim))
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, flow_params_eval)]

	state = env.reset()
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1
                
		# this is a dictionary where the keys are vehicles IDs and the value of each
		# key corresponds to its connected vehicles (up to four)
		connections = compute_connections(env, state)
                
		aug_states = [[], [], [], []]
		actions = [[], [], [], []]
		actions_for_env = []
		num_conn = []
		for i, ids in enumerate(list(state.keys())):
			aug_state = compute_augmented_state(env, state, ids, connections[ids])
			num_conn.append((len(aug_state)-3)/4 - 1)
			if num_conn[i] == -1:
				action = max_action
			elif t < args.start_timesteps:
				action = np.random.normal(0, 1.7, size=action_dim).clip(-max_action, max_action)
			else:
				action = (policy[int(num_conn[i])].select_action(np.array(aug_state)) + np.random.normal(0, max_action*args.expl_noise, size=action_dim)).clip(-max_action, max_action)
			if num_conn[i] != -1:
				aug_states[int(num_conn[i])].append(aug_state)
				actions[int(num_conn[i])].append(action)
			actions_for_env.append(action)

		state_, reward_list, done, _ = env.step(actions_for_env, state)
		
		#if len(reward_list) < len(list(state.keys())):
		#	d = -len(reward_list) + len(list(state.keys()))
		#	for _ in range(d):
		#		reward_list.append(0)

		connections_ = compute_connections(env, state_)
		aug_states_ = [[], [], [], []]
		rewards = [[], [], [], []]
		num_conn_ = []
		for i, ids in enumerate(list(state.keys())):
			num_conn_.append(-1)
			if ids in list(state_.keys()):
				aug_state = compute_augmented_state(env, state_, ids, connections_[ids])
				num_conn_[i] = ((len(aug_state)-3)/4 - 1)
				if num_conn[i] != -1:
					if num_conn_[i] > num_conn[i]:
						diff = num_conn_[i] - num_conn[i]
						aug_state = aug_state[:-4]
					elif num_conn_[i] < num_conn[i]:
						diff = num_conn[i] - num_conn_[i]
						for j in range(4*int(diff)):
							aug_state.append(0)
					aug_states_[int(num_conn[i])].append(aug_state)
					rewards[int(num_conn[i])].append(reward_list[i])
			else:
				if num_conn[i] != -1:
					aug_state = list(np.zeros(int(num_conn[i])))
					aug_states_[int(num_conn[i])].append(aug_state)
					rewards[int(num_conn[i])].append(reward_list[i])

		done_bool = float(done) if episode_timesteps < 5000 else 0

		# Store data in replay buffer
		train_models = [False, False, False, False]
		for i in range(4):
			for j in range(len(aug_states[i])):
				train_models[i] = True
				replay_buffer[i].add(aug_states[i][j], actions[i][j], aug_states_[i][j], rewards[i][j], done_bool)

		state = state_
		episode_reward += sum(reward_list)

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			for i in range(4):
				if train_models[i] == True:
					policy[i].train(replay_buffer[i], args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			env.terminate()
			create_env, _ = make_create_env(flow_params)
			env = create_env()
			# Reset environment
			state = env.reset()
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, flow_params_eval))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model:
				for i in range(4):
					policy[i].save(f"./models/{file_name}_{i}")

	env.terminate()
