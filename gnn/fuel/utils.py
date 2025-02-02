import numpy as np
import torch
from numpy.linalg import inv
from torch_geometric.utils import from_networkx
from flow.core.params import InFlows

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

edges_dict = {"b_c": 0, "c_r": 1, "r_c": 2, "c_t": 3, "t_c": 4, "c_l": 5,
              "l_c": 6, "c_b": 7, ":center_0": 8, ":center_1": 9, ":center_2": 10,
              ":center_12": 10, ":center_3": 11, ":center_4": 12, ":center_5": 13,
              ":center_6": 14, ":center_7": 15, ":center_8": 16, ":center_13": 16,
              ":center_9": 17, ":center_10": 18, ":center_11": 19}

routes_dict = {('t_c', 'c_l'): 0, ('t_c', 'c_b'): 1, ('t_c', 'c_r'): 2,
               ('r_c', 'c_t'): 3, ('r_c', 'c_l'): 4, ('r_c', 'c_b'): 5,
               ('b_c', 'c_r'): 6, ('b_c', 'c_t'): 7, ('b_c', 'c_l'): 8,
               ('l_c', 'c_b'): 9, ('l_c', 'c_r'): 10, ('l_c', 'c_t'): 11}


conflicting_routes_matrix = np.zeros((12, 12))
for a in range(12):
    for b in range(12):
        if a == 0:
            if b in (4, 8):
                conflicting_routes_matrix[a][b] = 1
        elif a == 1:
            if b in (4, 5, 8, 9, 10, 11):
                conflicting_routes_matrix[a][b] = 1
        elif a == 2:
            if b in (4, 5, 6, 7, 10, 11):
                conflicting_routes_matrix[a][b] = 1
        elif a == 3:
            if b in (7, 11):
                conflicting_routes_matrix[a][b] = 1
        elif a == 4:
            if b in (0, 1, 2, 7, 8, 11):
                conflicting_routes_matrix[a][b] = 1
        elif a == 5:
            if b in (1, 2, 7, 8, 9, 10):
                conflicting_routes_matrix[a][b] = 1
        elif a == 6:
            if b in (2, 10):
                conflicting_routes_matrix[a][b] = 1
        elif a == 7:
            if b in (2, 3, 4, 5, 10, 11):
                conflicting_routes_matrix[a][b] = 1
        elif a == 8:
            if b in (0, 1, 4, 5, 10, 11):
                conflicting_routes_matrix[a][b] = 1
        elif a == 9:
            if b in (1, 5):
                conflicting_routes_matrix[a][b] = 1
        elif a == 10:
            if b in (1, 2, 5, 6, 7, 8):
                conflicting_routes_matrix[a][b] = 1
        else:
            if b in (1, 2, 3, 4, 7, 8):
                conflicting_routes_matrix[a][b] = 1
                
routes_edges_matrix = np.zeros((12, 20))
for c in range(12):
    if c == 0:
        routes_edges_matrix[c][4] = 1
        routes_edges_matrix[c][8] = 2
        routes_edges_matrix[c][5] = 3
    elif c == 1:
        routes_edges_matrix[c][4] = 1
        routes_edges_matrix[c][9] = 2
        routes_edges_matrix[c][7] = 3
    elif c == 2:
        routes_edges_matrix[c][4] = 1
        routes_edges_matrix[c][10] = 2
        routes_edges_matrix[c][1] = 3
    elif c == 3:
        routes_edges_matrix[c][2] = 1
        routes_edges_matrix[c][11] = 2
        routes_edges_matrix[c][3] = 3
    elif c == 4:
        routes_edges_matrix[c][2] = 1
        routes_edges_matrix[c][12] = 2
        routes_edges_matrix[c][5] = 3
    elif c == 5:
        routes_edges_matrix[c][2] = 1
        routes_edges_matrix[c][13] = 2
        routes_edges_matrix[c][7] = 3
    elif c == 6:
        routes_edges_matrix[c][0] = 1
        routes_edges_matrix[c][14] = 2
        routes_edges_matrix[c][1] = 3
    elif c == 7:
        routes_edges_matrix[c][0] = 1
        routes_edges_matrix[c][15] = 2
        routes_edges_matrix[c][3] = 3
    elif c == 8:
        routes_edges_matrix[c][0] = 1
        routes_edges_matrix[c][16] = 2
        routes_edges_matrix[c][5] = 3
    elif c == 9:
        routes_edges_matrix[c][6] = 1
        routes_edges_matrix[c][17] = 2
        routes_edges_matrix[c][7] = 3
    elif c == 10:
        routes_edges_matrix[c][6] = 1
        routes_edges_matrix[c][18] = 2
        routes_edges_matrix[c][1] = 3
    else:
        routes_edges_matrix[c][6] = 1
        routes_edges_matrix[c][19] = 2
        routes_edges_matrix[c][3] = 3


def compute_edges(env, state):
    
    dimensions_matrix = np.array([[25/4, 0], [0, 1]])
    edges = {}
    edges_type = {}
    for i in env.k.vehicle.get_ids():
        for j in env.k.vehicle.get_ids():
            if conflicting_routes_matrix[state[i][6]][state[j][6]] == 1:
                if ((routes_edges_matrix[state[i][6]][state[i][5]] != 3) and
                        (routes_edges_matrix[state[j][6]][state[j][5]] != 3)):
                    # DISTANCE
                    rotation_matrix = np.array([[np.cos(state[i][4]), -np.sin(state[i][4])],
                                               [np.sin(state[i][4]), np.cos(state[i][4])]])
                    sigma_matrix = np.matmul(np.matmul(rotation_matrix, dimensions_matrix),
                                             rotation_matrix.transpose())
                    cartesian_dist = np.array(state[j][3]) - np.array(state[i][3])
                    d_ij = np.matmul(np.matmul(cartesian_dist.transpose(), inv(sigma_matrix)),
                                     cartesian_dist)
                    d_ij = 1/np.sqrt(d_ij)
                
                    # BEARING
                    coord_j = np.array(state[j][3])
                    coord_i = np.array(state[i][3])
                    py = coord_i[1] - coord_j[1]
                    px = coord_i[0] - coord_j[1]
                    chi_ij = np.arctan(py/px) - state[j][4]
                    
                    edges_type[(i, j)] = 'crossing'
                    
                    edges[(i, j)] = (d_ij, chi_ij)
                
                elif np.argmax(routes_edges_matrix[state[i][6]]) == np.argmax(routes_edges_matrix[state[j][6]]):
                    if state[i][0] > state[j][0]:
                        # DISTANCE
                        rotation_matrix = np.array([[np.cos(state[i][4]), -np.sin(state[i][4])],
                                                   [np.sin(state[i][4]), np.cos(state[i][4])]])
                        sigma_matrix = np.matmul(np.matmul(rotation_matrix, dimensions_matrix),
                                                 rotation_matrix.transpose())
                        cartesian_dist = np.array(state[j][3]) - np.array(state[i][3])
                        d_ij = np.matmul(np.matmul(cartesian_dist.transpose(), inv(sigma_matrix)),
                                         cartesian_dist)
                        d_ij = 1/np.sqrt(d_ij)

                        # BEARING
                        coord_j = np.array(state[j][3])
                        coord_i = np.array(state[i][3])
                        py = coord_i[1] - coord_j[1]
                        px = coord_i[0] - coord_j[1]
                        chi_ij = np.arctan(py/px) - state[j][4]
                        
                        edges_type[(i, j)] = 'same_lane'
                        
                        edges[(i, j)] = (d_ij, chi_ij)
            
            elif state[i][6] == state[j][6]:
                if state[i][0] > state[j][0]:
                    # DISTANCE
                    rotation_matrix = np.array([[np.cos(state[i][4]), -np.sin(state[i][4])],
                                               [np.sin(state[i][4]), np.cos(state[i][4])]])
                    sigma_matrix = np.matmul(np.matmul(rotation_matrix, dimensions_matrix),
                                             rotation_matrix.transpose())
                    cartesian_dist = np.array(state[j][3]) - np.array(state[i][3])
                    d_ij = np.matmul(np.matmul(cartesian_dist.transpose(), inv(sigma_matrix)),
                                     cartesian_dist)
                    d_ij = 1/np.sqrt(d_ij)

                    # BEARING
                    coord_j = np.array(state[j][3])
                    coord_i = np.array(state[i][3])
                    py = coord_i[1] - coord_j[1]
                    px = coord_i[0] - coord_j[1]
                    chi_ij = np.arctan(py/px) - state[j][4]
                    
                    edges_type[(i, j)] = 'same_lane'
                    
                    edges[(i, j)] = (d_ij, chi_ij)
            
            elif state[i][5] == state[j][5]:
                if state[i][0] > state[j][0]:
                    # DISTANCE
                    rotation_matrix = np.array([[np.cos(state[i][4]), -np.sin(state[i][4])],
                                               [np.sin(state[i][4]), np.cos(state[i][4])]])
                    sigma_matrix = np.matmul(np.matmul(rotation_matrix, dimensions_matrix),
                                             rotation_matrix.transpose())
                    cartesian_dist = np.array(state[j][3]) - np.array(state[i][3])
                    d_ij = np.matmul(np.matmul(cartesian_dist.transpose(), inv(sigma_matrix)),
                                     cartesian_dist)
                    d_ij = 1/np.sqrt(d_ij)

                    # BEARING
                    coord_j = np.array(state[j][3])
                    coord_i = np.array(state[i][3])
                    py = coord_i[1] - coord_j[1]
                    px = coord_i[0] - coord_j[1]
                    chi_ij = np.arctan(py/px) - state[j][4]

                    edges_type[(i, j)] = 'same_lane'
                    
                    edges[(i, j)] = (d_ij, chi_ij)
                    
    return edges, edges_type


def compute_rp(graph, reward):
    num_edges = graph.edge_index.shape[1]
    if num_edges == 0:
        return reward
    else:
        d = torch.sum(graph.dist, dim=0).item()
        assert graph.dist.shape[0] == num_edges
        rp = -d / num_edges
        w_p = 0.2
        reward += w_p * rp

        return reward


def from_networkx_multigraph(g, nn_architecture):
    data = from_networkx(g)
    if data.pos is None:
        return data

    # Extract node attributes
    if nn_architecture == "base":
        node_attrs = ['pos', 'vel', 'acc', 'omega']
    else:
        node_attrs = ['pos', 'vel', 'acc']
    data.x = torch.cat([torch.stack([g.nodes[n][attr] for n in g.nodes()]) for attr in node_attrs], dim=-1)

    # Extract edge types
    edge_types = []
    edge_attrs = ['dist', 'bearing']
    edge_attr_list = {attr: [] for attr in edge_attrs}
    for u, v, k, attr in g.edges(data=True, keys=True):
        edge_types.append(k)
        for edge_attr in edge_attrs:
            edge_attr_list[edge_attr].append(attr[edge_attr])

    # Map edge types to integers
    edge_type_mapping = {etype: i for i, etype in enumerate(set(edge_types))}
    edge_type_indices = torch.tensor([edge_type_mapping[etype] for etype in edge_types],
                                     dtype=torch.long, device=device)

    # Add edge types and attributes to the data object
    data.edge_type = edge_type_indices
    data.edge_attr = torch.cat([torch.tensor(edge_attr_list[attr],
                                             dtype=torch.float).view(-1, 1) for attr in edge_attrs],
                               dim=-1).to(device)
    data.edge_index = data.edge_index.to(device)

    return data


def eval_policy(aim, env, eval_episodes=10, test=True, nn_architecture='base', omega_space='discrete'):

    avg_reward = 0.0
    num_crashes = 0
    if test:
        avg_speed = [[], [], [], [], [], [], [], [], [], [], []]
        avg_emissions = [[], [], [], [], [], [], [], [], [], [], []]
    if omega_space == 'continuous':
        omegas = np.linspace(0.0, 1.0, num=eval_episodes * 10, dtype=np.float64)
    else:
        omegas = None
    for i in range(eval_episodes * 10):

        state = env.reset()
        if omegas is None:
            env.omega = (i % 11) / 10
        else:
            env.omega = omegas[i]
        if nn_architecture == 'base':
            state.x[:, -1] = env.omega
        while state.x is None:
            state, _, _, _ = env.step([], evaluate=True)
        done = False
        ep_steps = 0
        # veh_num = 4
        while not done:
            ep_steps += 1

            if state.x is None:
                state, _, done, _ = env.step([], evaluate=True)
            else:
                if test:
                    speed = 0
                    emission = 0
                    for idx in env.k.vehicle.get_ids():
                        speed += env.k.vehicle.get_speed(idx)
                        emission += env.k.vehicle.kernel_api.vehicle.getCO2Emission(idx) / 50000
                    if omega_space == 'continuous':
                        avg_speed[i // 10].append(speed / len(env.k.vehicle.get_ids()))
                        avg_emissions[i // 10].append(emission / len(env.k.vehicle.get_ids()))
                    else:
                        avg_speed[i % 11].append(speed / len(env.k.vehicle.get_ids()))
                        avg_emissions[i % 11].append(emission / len(env.k.vehicle.get_ids()))

                if nn_architecture == 'base':
                    actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type)
                else:
                    actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type,
                                                torch.tensor([[env.omega, 1 - env.omega]], dtype=torch.float, device=device).repeat(state.x.shape[0], 1))
                state, reward, done, _ = env.step(rl_actions=actions, evaluate=True)
            if env.k.simulation.check_collision():
                num_crashes += 1

            #  if state.x is None:
            # if ep_steps % 150 == 0:
            #     # we may need to put "best" instead of 0 as starting lane (aquarium)
            #     env.k.vehicle.add("rl_{}".format(veh_num), "rl", "b_c", 0.0, "best", 0.0)
            #     env.k.vehicle.add("rl_{}".format(veh_num + 1), "rl", "t_c", 0.0, "best", 0.0)
            #     env.k.vehicle.add("rl_{}".format(veh_num + 2), "rl", "l_c", 0.0, "best", 0.0)
            #     env.k.vehicle.add("rl_{}".format(veh_num + 3), "rl", "r_c", 0.0, "best", 0.0)
            #     veh_num += 4
            avg_reward += reward

        # print(f"Average speed: {np.mean(avg_speed):.3f}. Average CO2 emission: {np.mean(avg_emissions):.3f}")

        # tot_veh_num += veh_num
    avg_reward /= (eval_episodes * 10)
    # tot_veh_num = tot_veh_num / 2
    if test:
        objectives = []
        for i in range(11):
            objectives.append([-np.mean(avg_speed[i]), np.mean(avg_emissions[i])])
        num_pareto_solutions = compute_pareto_front(objectives)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}. Number of crashes: {num_crashes}."
          f"Number of Pareto solutions: {num_pareto_solutions}")
    print("---------------------------------------")

    return avg_reward, num_crashes, num_pareto_solutions


def compute_pareto_front(objectives):
    # Ensure inputs are numpy arrays
    objectives = np.array(objectives)

    # Determine the Pareto front
    num_points = objectives.shape[0]
    is_pareto = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        for j in range(num_points):
            if all(objectives[j] <= objectives[i]) and any(objectives[j] < objectives[i]):
                is_pareto[i] = False
                break
    pareto_front = objectives[is_pareto]

    return pareto_front


def eval_policy_pareto_continuous(aim, env, eval_episodes=10, nn_architecture='base'):

    avg_reward = 0.0
    num_crashes = 0
    avg_speed = []
    avg_emissions = []
    omegas = np.linspace(0.0, 1.0, num=eval_episodes * 10, dtype=np.float64)
    for i in range(eval_episodes * 10):

        state = env.reset()
        env.omega = omegas[i]
        if nn_architecture == 'base':
            state.x[:, -1] = env.omega
        while state.x is None:
            state, _, _, _ = env.step([], evaluate=True)
        done = False
        ep_steps = 0
        speed = []
        emission = []
        while not done:
            ep_steps += 1

            if state.x is None:
                state, _, done, _ = env.step([], evaluate=True)
            else:
                speed_per_veh = 0
                emission_per_veh = 0
                for idx in env.k.vehicle.get_ids():
                    speed_per_veh += env.k.vehicle.get_speed(idx)
                    emission_per_veh += env.k.vehicle.kernel_api.vehicle.getCO2Emission(idx) / 50000
                speed.append(speed_per_veh/len(env.k.vehicle.get_ids()))
                emission.append(emission_per_veh/len(env.k.vehicle.get_ids()))

                if nn_architecture == 'base':
                    actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type)
                else:
                    actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type,
                                                torch.tensor([[env.omega, 1 - env.omega]], dtype=torch.float, device=device).repeat(state.x.shape[0], 1))
                state, reward, done, _ = env.step(rl_actions=actions, evaluate=True)
            if env.k.simulation.check_collision():
                num_crashes += 1

            avg_reward += reward
        avg_speed.append(np.mean(speed))
        avg_emissions.append(np.mean(emission))

    avg_reward /= (eval_episodes * 10)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}. Number of crashes: {num_crashes}")
    print("---------------------------------------")

    # np.save(f'pareto_speed_continuous_{nn_architecture}.npy', avg_speed)
    # np.save(f'pareto_emission_continuous_{nn_architecture}.npy', avg_emissions)

    return avg_reward, num_crashes


def eval_policy_pareto_discrete(aim, env, eval_episodes=10, nn_architecture='base'):

    avg_reward = 0.0
    num_crashes = 0
    avg_speed = [[], [], [], [], [], [], [], [], [], [], []]
    avg_emissions = [[], [], [], [], [], [], [], [], [], [], []]
    for i in range(eval_episodes * 10):

        state = env.reset()
        env.omega = (i % 11) / 10
        if nn_architecture == 'base':
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
                speed = 0
                emission = 0
                for idx in env.k.vehicle.get_ids():
                    speed += env.k.vehicle.get_speed(idx)
                    emission += env.k.vehicle.kernel_api.vehicle.getCO2Emission(idx) / 50000
                avg_speed[i % 11].append(speed / len(env.k.vehicle.get_ids()))
                avg_emissions[i % 11].append(emission / len(env.k.vehicle.get_ids()))

                if nn_architecture == 'base':
                    actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type)
                else:
                    actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type,
                                                torch.tensor([[env.omega, 1 - env.omega]], dtype=torch.float, device=device).repeat(state.x.shape[0], 1))
                state, reward, done, _ = env.step(rl_actions=actions, evaluate=True)
            if env.k.simulation.check_collision():
                num_crashes += 1

            avg_reward += reward

    avg_reward /= (eval_episodes * 10)
    pareto_speed = []
    pareto_emission = []
    for i in range(11):
        pareto_speed.append(np.mean(avg_speed[i]))
        pareto_emission.append(np.mean(avg_emissions[i]))

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}. Number of crashes: {num_crashes}")
    print("---------------------------------------")

    np.save(f'pareto_speed_discrete_{nn_architecture}.npy', pareto_speed)
    np.save(f'pareto_emission_discrete_{nn_architecture}.npy', pareto_emission)

    return avg_reward, num_crashes


def get_inflows(rate=100):
    inflow = InFlows()
    inflow.add(veh_type="rl",
               edge="b_c",
               # vehs_per_hour=rate,
               probability=rate,
               depart_speed=0,
              )
    inflow.add(veh_type="rl",
               edge="t_c",
               # vehs_per_hour=rate,
               probability=rate,
               depart_speed=0,
              )
    inflow.add(veh_type="rl",
               edge="l_c",
               # vehs_per_hour=rate,
               probability=rate,
               depart_speed=0,
              )
    inflow.add(veh_type="rl",
               edge="r_c",
               # vehs_per_hour=rate,
               probability=rate,
               depart_speed=0,
              )

    return inflow


def eval_policy_inflows(aim, env, eval_episodes=10):

    avg_reward = 0.0
    num_crashes = 0
    tot_veh_num = 0
    for _ in range(eval_episodes):
        state = env.reset()
        while state.x is None:
            state, _, _, _ = env.step([])
        done = False
        ep_steps = 0
        while not done:
            ep_steps += 1

            if state.x is None:
                state, _, done, _ = env.step([])
            else:
                actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type)
                state, reward, done, _ = env.step(rl_actions=actions)
            if env.k.simulation.check_collision():
                num_crashes += 1

            avg_reward += reward
        tot_veh_num += np.sum(env.k.vehicle._num_departed)

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}. Number of crashes: {num_crashes}")
    print(tot_veh_num)
    return avg_reward, num_crashes
