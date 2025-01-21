import numpy as np
import torch
from flow.controllers import ContinuousRouter, RLController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, VehicleParams
from flow.utils.registry import make_create_env
from intersection_network import IntersectionNetwork, ADDITIONAL_NET_PARAMS
from intersection_env_new import MyEnv, ADDITIONAL_ENV_PARAMS
from utils import compute_pareto_front
import argparse
import warnings
import pickle
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--file_name", default="")
parser.add_argument("--nn_architecture", default="base")
args = parser.parse_args()

if args.nn_architecture == "smart":
    from agent_smart import TD3
else:
    from agent import TD3

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
env.nn_architecture = args.nn_architecture
env.omega_space = "continuous"

file_name = f"aim_{args.seed}_{args.file_name}"
print("---------------------------------------")
print(f"Seed: {args.seed}")
print("---------------------------------------")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.nn_architecture == "smart":
    state_dim = 3
else:
    state_dim = 4
edge_dim = 2
action_dim = 1
max_action = 5.0

aim = TD3(state_dim, edge_dim, action_dim, max_action=max_action)
aim.load(f"./models/{file_name}")

boxplot_speeds = []
boxplot_emissions = []

avg_reward = 0.0
tot_num_crashes = 0
avg_speed = []
avg_emissions = []
omegas = np.linspace(0.0, 1.0, num=100, dtype=np.float64)

global_dictionary = {}
for i in range(100):
    global_dictionary[i] = {'dictionary_name_speed': {},
                            'dictionary_name_emission': {}}

for i in range(100):
    # dictionary_name_speed = {}
    # dictionary_name_emission = {}
    speed = []
    emission = []
    for _ in range(10):
        state = env.reset()
        env.omega = omegas[i]
        if args.nn_architecture == 'base':
            state.x[:, -1] = env.omega
        while state.x is None:
            state, _, _, _ = env.step([], evaluate=True)
        done = False
        ep_steps = 0
        while not done:
            ep_steps += 1

            if state.x is None:
                state, _, done, _ = env.step([], evaluate=True)
            else:
                speed_per_veh = 0
                emission_per_veh = 0
                for idx in env.k.vehicle.get_ids():
                    if idx not in global_dictionary[i]['dictionary_name_speed']:
                        global_dictionary[i]['dictionary_name_speed'][idx] = []
                    global_dictionary[i]['dictionary_name_speed'][idx].append(env.k.vehicle.get_speed(idx))
                    if idx not in global_dictionary[i]['dictionary_name_emission']:
                        global_dictionary[i]['dictionary_name_emission'][idx] = []
                    global_dictionary[i]['dictionary_name_emission'][idx].append(env.k.vehicle.kernel_api.vehicle.getCO2Emission(idx) / 50000)

                    speed_per_veh += env.k.vehicle.get_speed(idx)
                    emission_per_veh += env.k.vehicle.kernel_api.vehicle.getCO2Emission(idx) / 50000

                speed.append(speed_per_veh / len(env.k.vehicle.get_ids()))
                emission.append(emission_per_veh / len(env.k.vehicle.get_ids()))

                if args.nn_architecture == 'base':
                    actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type)
                else:
                    actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type,
                                                torch.tensor([[env.omega, 1 - env.omega]], dtype=torch.float,
                                                             device=device).repeat(state.x.shape[0], 1))
                state, reward, done, _ = env.step(rl_actions=actions, evaluate=True)
            if env.k.simulation.check_collision():
                tot_num_crashes += 1

            avg_reward += reward

    avg_speed.append(np.mean(speed))
    avg_emissions.append(np.mean(emission))

    # speeds, emissions = [], []
    # for name_s, name_e in zip(list(dictionary_name_speed.keys()), list(dictionary_name_emission.keys())):
    #     speeds.append(np.mean(dictionary_name_speed[name_s]))
    #     emissions.append(np.mean(dictionary_name_emission[name_e]))
    # boxplot_speeds.append(speeds)
    # boxplot_emissions.append(emissions)

env.terminate()

print(f'Crashes: {tot_num_crashes}')

pareto_front = []
for i in range(len(avg_speed)):
    pareto_front.append((round(avg_speed[i], 9), round(-avg_emissions[i], 9)))
front, indexes = compute_pareto_front(pareto_front)

for i in indexes:
    speeds, emissions = [], []
    for name_s, name_e in zip(list(global_dictionary[i]['dictionary_name_speed'].keys()), list(global_dictionary[i]['dictionary_name_emission'].keys())):
        speeds.append(np.mean(global_dictionary[i]['dictionary_name_speed'][name_s]))
        emissions.append(np.mean(global_dictionary[i]['dictionary_name_emission'][name_e]))
    boxplot_speeds.append(speeds)
    boxplot_emissions.append(emissions)

# Compute the median of each sublist and sort the data accordingly
# sorted_speeds = sorted(boxplot_speeds, key=lambda x: np.median(x))
# sorted_emissions = sorted(boxplot_emissions, key=lambda x: np.median(x))
#
# # Plot the sorted boxplot
# plt.boxplot(sorted_speeds, vert=True, patch_artist=True)
# plt.show()
# plt.close()
#
# plt.boxplot(sorted_emissions, vert=True, patch_artist=True)
# plt.show()
# plt.close()
#
# indexed_bs = list(enumerate(boxplot_speeds))
# sorted_indexed_bs = sorted(indexed_bs, key=lambda x: x[1])
# original_indexes_bs = [idx for idx, _ in sorted_indexed_bs]
#
# indexed_be = list(enumerate(boxplot_emissions))
# sorted_indexed_be = sorted(indexed_be, key=lambda x: x[1])
# original_indexes_be = [idx for idx, _ in sorted_indexed_be]
#
# count = 0
# for i in range(len(original_indexes_bs)):
#     if original_indexes_bs[i] != original_indexes_be[i]:
#         count += 1
# print(count)

with open(f"boxplot_speeds_{args.nn_architecture}.pkl", "wb") as f:
    pickle.dump(boxplot_speeds, f)
with open(f"boxplot_emissions_{args.nn_architecture}.pkl", "wb") as f:
    pickle.dump(boxplot_emissions, f)
