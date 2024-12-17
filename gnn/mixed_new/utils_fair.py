import numpy as np
import torch
from numpy.linalg import inv
from torch_geometric.utils import from_networkx
from itertools import product

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

RED = (255, 0, 0)


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

                    # if env.k.vehicle.get_route(i)[0] in ('t_c', 'b_c'):
                    if env.k.vehicle.get_emission_class(i) == "HBEFA3/PC_G_EU4":
                        emissions_i = 'fuel'
                    else:
                        emissions_i = 'electric'
                    # if env.k.vehicle.get_route(j)[0] in ('t_c', 'b_c'):
                    if env.k.vehicle.get_emission_class(j) == "HBEFA3/PC_G_EU4":
                        emissions_j = 'fuel'
                    else:
                        emissions_j = 'electric'

                    if emissions_i == emissions_j == 'fuel':
                        edges_type[(i, j)] = 'crossing_ff'
                    if emissions_i == emissions_j == 'electric':
                        edges_type[(i, j)] = 'crossing_ee'
                    if emissions_i != emissions_j and emissions_i == 'fuel':
                        edges_type[(i, j)] = 'crossing_fe'
                    if emissions_i != emissions_j and emissions_i == 'electric':
                        edges_type[(i, j)] = 'crossing_ef'
                    
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

                        # if env.k.vehicle.get_route(i)[0] in ('t_c', 'b_c'):
                        if env.k.vehicle.get_emission_class(i) == "HBEFA3/PC_G_EU4":
                            emissions_i = 'fuel'
                        else:
                            emissions_i = 'electric'
                        # if env.k.vehicle.get_route(j)[0] in ('t_c', 'b_c'):
                        if env.k.vehicle.get_emission_class(j) == "HBEFA3/PC_G_EU4":
                            emissions_j = 'fuel'
                        else:
                            emissions_j = 'electric'

                        if emissions_i == emissions_j == 'fuel':
                            edges_type[(i, j)] = 'same_lane_ff'
                        if emissions_i == emissions_j == 'electric':
                            edges_type[(i, j)] = 'same_lane_ee'
                        if emissions_i != emissions_j and emissions_i == 'fuel':
                            edges_type[(i, j)] = 'same_lane_fe'
                        if emissions_i != emissions_j and emissions_i == 'electric':
                            edges_type[(i, j)] = 'same_lane_ef'
                        
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

                    # if env.k.vehicle.get_route(i)[0] in ('t_c', 'b_c'):
                    if env.k.vehicle.get_emission_class(i) == "HBEFA3/PC_G_EU4":
                        emissions_i = 'fuel'
                    else:
                        emissions_i = 'electric'
                    # if env.k.vehicle.get_route(j)[0] in ('t_c', 'b_c'):
                    if env.k.vehicle.get_emission_class(j) == "HBEFA3/PC_G_EU4":
                        emissions_j = 'fuel'
                    else:
                        emissions_j = 'electric'

                    if emissions_i == emissions_j == 'fuel':
                        edges_type[(i, j)] = 'same_lane_ff'
                    if emissions_i == emissions_j == 'electric':
                        edges_type[(i, j)] = 'same_lane_ee'
                    if emissions_i != emissions_j and emissions_i == 'fuel':
                        edges_type[(i, j)] = 'same_lane_fe'
                    if emissions_i != emissions_j and emissions_i == 'electric':
                        edges_type[(i, j)] = 'same_lane_ef'
                    
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

                    # if env.k.vehicle.get_route(i)[0] in ('t_c', 'b_c'):
                    if env.k.vehicle.get_emission_class(i) == "HBEFA3/PC_G_EU4":
                        emissions_i = 'fuel'
                    else:
                        emissions_i = 'electric'
                    # if env.k.vehicle.get_route(j)[0] in ('t_c', 'b_c'):
                    if env.k.vehicle.get_emission_class(j) == "HBEFA3/PC_G_EU4":
                        emissions_j = 'fuel'
                    else:
                        emissions_j = 'electric'

                    if emissions_i == emissions_j == 'fuel':
                        edges_type[(i, j)] = 'same_lane_ff'
                    if emissions_i == emissions_j == 'electric':
                        edges_type[(i, j)] = 'same_lane_ee'
                    if emissions_i != emissions_j and emissions_i == 'fuel':
                        edges_type[(i, j)] = 'same_lane_fe'
                    if emissions_i != emissions_j and emissions_i == 'electric':
                        edges_type[(i, j)] = 'same_lane_ef'
                    
                    edges[(i, j)] = (d_ij, chi_ij)
                    
    return edges, edges_type


def from_networkx_multigraph(g, nn_architecture):
    data = from_networkx(g)
    if data.pos is None:
        return data

    # Extract node attributes
    if nn_architecture == "base":
        node_attrs = ['pos', 'vel', 'acc', 'emission', 'omegas']
    else:
        node_attrs = ['pos', 'vel', 'acc', 'emission']
    data.x = torch.cat([torch.stack([g.nodes[n][attr] for n in g.nodes()]) for attr in node_attrs], dim=-1)

    # Extract edge types
    edge_types = []
    all_edge_types = ['crossing_ff', 'crossing_ee', 'crossing_fe', 'crossing_ef',
                      'same_lane_ff', 'same_lane_ee', 'same_lane_fe', 'same_lane_ef']
    edge_attrs = ['dist', 'bearing']
    edge_attr_list = {attr: [] for attr in edge_attrs}
    for u, v, k, attr in g.edges(data=True, keys=True):
        edge_types.append(k)
        for edge_attr in edge_attrs:
            edge_attr_list[edge_attr].append(attr[edge_attr])

    # Map edge types to integers
    edge_type_mapping = {etype: i for i, etype in enumerate(all_edge_types)}
    edge_type_indices = torch.tensor([edge_type_mapping[etype] for etype in edge_types],
                                     dtype=torch.long, device=device)

    # Add edge types and attributes to the data object
    data.edge_type = edge_type_indices
    data.edge_attr = torch.cat([torch.tensor(edge_attr_list[attr],
                                             dtype=torch.float).view(-1, 1) for attr in edge_attrs],
                               dim=-1).to(device)
    data.edge_index = data.edge_index.to(device)

    return data


def eval_policy_pareto_continuous(aim, env, eval_episodes=66, nn_architecture='base', test=False):

    avg_reward = 0.0
    tot_num_crashes = 0
    avg_speed = []
    avg_emissions = []
    # avg_time_delta = []
    avg_space_delta = []
    omegas = generate_combinations_with_sum()  # CHANGE ALSO NUM EVAL EPISODES (66)
    # omegas = np.random.dirichlet([1, 1, 1], size=eval_episodes*10)

    for i in range(eval_episodes):
        speed = []
        emission = []
        # time_delta = []
        space_delta = []
        for _ in range(10):
            state = env.reset()
            env.omegas = omegas[i]
            if nn_architecture == 'base':
                state.x[:, -3] = env.omegas[0]
                state.x[:, -2] = env.omegas[1]
                state.x[:, -1] = env.omegas[2]
            while state.x is None:
                state, _, _, _ = env.step([], evaluate=True)

            ids = env.k.vehicle.get_ids()
            elec_vehs = list(np.random.choice(ids, 2, replace=False))
            env.k.vehicle.set_emission_class(elec_vehs)
            for v in elec_vehs:
                env.k.vehicle.set_color(v, RED)

            done = False
            ep_steps = 0
            while not done:
                ep_steps += 1

                if state.x is None:
                    state, _, done, _ = env.step([], evaluate=True)
                else:
                    speed_per_veh = 0
                    emission_per_veh = 0
                    # time_per_veh_fuel = 0
                    # time_per_veh_elec = 0
                    space_per_veh_fuel = 0
                    space_per_veh_elec = 0
                    f_vehs = 0
                    e_vehs = 0
                    for index, idx in enumerate(env.k.vehicle.get_ids()):
                        speed_per_veh += env.k.vehicle.get_speed(idx)
                        emission_per_veh += env.k.vehicle.kernel_api.vehicle.getCO2Emission(idx) / 50000
                        # if env.k.vehicle.get_emission_class(idx) == "HBEFA3/PC_G_EU4":
                            # time_per_veh_fuel += 0.1
                        # else:
                            # time_per_veh_elec += 0.1
                        if state.x[index, 3].item() == 1:
                            space_per_veh_fuel += state.x[index, 0].item() + 70
                            f_vehs += 1
                        else:
                            space_per_veh_elec += state.x[index, 0].item() + 70
                            e_vehs += 1

                    if f_vehs < e_vehs:
                        space_per_veh_fuel += 120 * (f_vehs - e_vehs)
                    elif f_vehs > e_vehs:
                        space_per_veh_elec += 120 * (e_vehs - f_vehs)

                    speed.append(speed_per_veh / len(env.k.vehicle.get_ids()))
                    emission.append(emission_per_veh / len(env.k.vehicle.get_ids()))
                    space_delta.append(abs(space_per_veh_elec - space_per_veh_fuel) / 240)

                    if nn_architecture == 'base':
                        actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type)
                    else:
                        om = [env.omegas[0].item(), env.omegas[1].item(), env.omegas[2].item()]
                        actions = aim.select_action(state.x, state.edge_index, state.edge_attr, state.edge_type,
                                                    torch.tensor([om], dtype=torch.float,
                                                                 device=device).repeat(state.x.shape[0], 1))
                    state, reward, done, _ = env.step(rl_actions=actions, evaluate=True)
                if env.k.simulation.check_collision():
                    tot_num_crashes += 1

                avg_reward += reward

        avg_speed.append(np.mean(speed))
        avg_emissions.append(np.mean(emission))
        avg_space_delta.append(np.mean(space_delta))

    avg_reward /= (eval_episodes * 10)

    pareto_front = []
    for i in range(len(avg_speed)):
        pareto_front.append((avg_speed[i], -avg_emissions[i], -avg_space_delta[i]))
    front = compute_pareto_front(pareto_front)

    hv = compute_hypervolume(front)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}."
          f"Number of crashes: {tot_num_crashes}. Hypervolume: {hv}")
    print("---------------------------------------")

    if test:
        np.save(f'pareto_front_continuous_{nn_architecture}.npy', front)

    return tot_num_crashes, hv, front


def compute_pareto_front(solutions):
    """
    Compute the Pareto frontier for a three-objective maximization problem.

    Parameters:
    solutions (list of tuples): A list of solutions. Each solution is a tuple (f1, f2, f3),
                                 where f1, f2, and f3 are the three objectives.

    Returns:
    list of tuples: The Pareto frontier as a list of non-dominated solutions.
    """
    # Sort solutions by the first objective in descending order, breaking ties by second and third in descending order
    sorted_solutions = sorted(solutions, key=lambda x: (-x[0], -x[1], -x[2]))

    pareto_front = []
    max_f2 = float('-inf')
    max_f3 = float('-inf')

    for f1, f2, f3 in sorted_solutions:
        # Check if the current solution is dominated
        if f2 > max_f2 or (f2 == max_f2 and f3 > max_f3):
            pareto_front.append((f1, f2, f3))
            max_f2 = max(max_f2, f2)
            max_f3 = max(max_f3, f3)

    return pareto_front


def compute_hypervolume(pareto_front, reference_point=(0, -1.03, -1)):
    """
    Compute the hypervolume for a three-objective maximization problem.

    Parameters:
    pareto_front (list of tuples): A list of points on the Pareto front. Each point is a tuple (f1, f2, f3),
                                   where f1, f2, and f3 are the three objectives.
    reference_point (tuple): The reference point for the hypervolume computation (f1_ref, f2_ref, f3_ref).

    Returns:
    float: The computed hypervolume.
    """
    # Sort Pareto front by the first objective in descending order, breaking ties with the second and third objectives
    sorted_front = sorted(pareto_front, key=lambda x: (-x[0], -x[1], -x[2]))

    # Compute hypervolume
    hypervolume = 0.0

    # Previous max values for second and third objectives
    previous_f2 = reference_point[1]
    previous_f3 = reference_point[2]

    for f1, f2, f3 in sorted_front:
        if f2 > previous_f2 or f3 > previous_f3:
            # Compute volume of the cuboid defined by the current point and the previous levels
            delta_f2 = f2 - previous_f2
            delta_f3 = f3 - previous_f3
            base_area = (f1 - reference_point[0])
            cuboid_volume = base_area * delta_f2 * delta_f3
            hypervolume += cuboid_volume

            # Update the previous max levels
            previous_f2 = max(previous_f2, f2)
            previous_f3 = max(previous_f3, f3)

    return hypervolume


def generate_combinations_with_sum():
    # Generate all ordered combinations of three integers from 0 to 10
    combinations = [
        combo for combo in product(range(11), repeat=3)
        if sum(combo) == 10
    ]

    real_combinations = np.zeros((66, 3))
    for idx, c in enumerate(combinations):
        comb = []
        for i in range(len(c)):
            comb.append(c[i] / 10 + np.random.randn(1).item() * 0.01)
            if comb[i] < 0:
                comb[i] = 0.01
            elif comb[i] > 1:
                comb[i] = 0.99
        real_combinations[idx] = comb

    return real_combinations
