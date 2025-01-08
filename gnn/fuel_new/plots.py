import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--nn_architecture", default="base")
args = parser.parse_args()

# Load the object from the file
with open(f"results/aim_0_continuous_{args.nn_architecture}_fixom_front.pkl", "rb") as f:
    fronts_list = pickle.load(f)
with open(f"results/aim_0_continuous_{args.nn_architecture}_fixom_hv.pkl", "rb") as f:
    hv_list = pickle.load(f)
with open(f"results/aim_0_continuous_{args.nn_architecture}_fixom_crashes.pkl", "rb") as f:
    crashes_list = pickle.load(f)


def compute_and_plot_ema(data, alpha=0.2):
    """
    Computes the exponential moving average (EMA) of a given array and optionally plots it.

    Parameters:
        data (array-like): The input data to compute EMA on.
        alpha (float): The smoothing factor (0 < alpha ≤ 1). Smaller alpha means more smoothing.
        plot (bool): Whether to plot the original data and EMA (default is True).

    Returns:
        np.ndarray: The computed EMA array.
    """
    if not (0 < alpha <= 1):
        raise ValueError("Alpha must be in the range (0, 1].")

    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]  # Initialize EMA with the first data point

    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]

    return ema


hv_list_ema = compute_and_plot_ema(hv_list)
crashes_list_ema = compute_and_plot_ema(crashes_list)

plt.figure(figsize=(10, 6))
# plt.plot(data, label='Original Data', color='blue', alpha=0.6)
plt.plot(hv_list_ema, color='red', linewidth=2)
plt.title('Hypervolume training curve')
plt.xlabel('Evaluation run')
plt.ylabel('Hypervolume')
plt.grid(True)
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
# plt.plot(data, label='Original Data', color='blue', alpha=0.6)
plt.plot(crashes_list_ema, color='red', linewidth=2)
plt.title('Number of crashes training curve')
plt.xlabel('Evaluation run')
plt.ylabel('Number of crashes')
plt.grid(True)
plt.show()
plt.close()

best_hv_index = np.argmax(hv_list)

pareto_front = np.array(fronts_list[best_hv_index])

pareto_vel, pareto_em = [], []
for i in range(pareto_front.shape[0]):
    pareto_vel.append(pareto_front[i][0])
    pareto_em.append(pareto_front[i][1])

x_staircase = np.concatenate([[pareto_vel[0]], pareto_vel])
y_staircase = np.concatenate([pareto_em, [pareto_em[-1]]])

# Create the figure and axis
plt.figure(figsize=(8, 6))
plt.step(x_staircase, y_staircase, where='post')
plt.scatter(np.array(pareto_vel), np.array(pareto_em))

# Set plot labels and title
plt.xlabel('Velocity [m/s]')
plt.ylabel('-CO2 emissions [mg/s]')
plt.grid(True)
plt.show()
plt.close()

print(crashes_list[best_hv_index])

best_num_sol = 0
best_index = 0
for i in range(len(fronts_list)):
    print(len(fronts_list[i]))
    if len(fronts_list[i]) > best_num_sol:
        best_num_sol = len(fronts_list[i])
        best_index = i

pareto_front = np.array(fronts_list[best_index])

pareto_vel, pareto_em = [], []
for i in range(pareto_front.shape[0]):
    pareto_vel.append(pareto_front[i][0])
    pareto_em.append(pareto_front[i][1])

x_staircase = np.concatenate([[pareto_vel[0]], pareto_vel])
y_staircase = np.concatenate([pareto_em, [pareto_em[-1]]])

# Create the figure and axis
plt.figure(figsize=(8, 6))
plt.step(x_staircase, y_staircase, where='post')
plt.scatter(np.array(pareto_vel), np.array(pareto_em))

# Set plot labels and title
plt.xlabel('Velocity [m/s]')
plt.ylabel('-CO2 emissions [mg/s]')
plt.grid(True)
plt.show()
plt.close()

print(crashes_list[best_index])

with open("results/aim_0_continuous_base_fixom_front.pkl", "rb") as f:
    fronts_list_base = pickle.load(f)
with open("results/aim_0_continuous_smart_fixom_front.pkl", "rb") as f:
    fronts_list_smart = pickle.load(f)

best_num_sol_base = 0
best_index_base = 0
for i in range(len(fronts_list_base)):
    if len(fronts_list_base[i]) > best_num_sol_base:
        best_num_sol_base = len(fronts_list_base[i])
        best_index_base = i

best_num_sol_smart = 0
best_index_smart = 0
for i in range(len(fronts_list_smart)):
    if len(fronts_list_smart[i]) > best_num_sol_smart:
        best_num_sol_smart = len(fronts_list_smart[i])
        best_index_smart = i

front_base = np.array(fronts_list_base[best_index_base])
front_smart = np.array(fronts_list_smart[best_index_smart])

pareto_vel_base, pareto_em_base = [], []
for i in range(front_base.shape[0]):
    pareto_vel_base.append(front_base[i][0])
    pareto_em_base.append(front_base[i][1])

pareto_vel_smart, pareto_em_smart = [], []
for i in range(front_smart.shape[0]):
    pareto_vel_smart.append(front_smart[i][0])
    pareto_em_smart.append(front_smart[i][1])

# Create the staircase points
x_staircase_base = np.concatenate([[pareto_vel_base[0]], pareto_vel_base])
y_staircase_base = np.concatenate([pareto_em_base, [pareto_em_base[-1]]])
x_staircase_smart = np.concatenate([[pareto_vel_smart[0]], pareto_vel_smart])
y_staircase_smart = np.concatenate([pareto_em_smart, [pareto_em_smart[-1]]])

# Create the figure and axis
plt.figure(figsize=(8, 6))
plt.step(x_staircase_base, y_staircase_base, where='post', label='Base')
plt.scatter(np.array(pareto_vel_base), np.array(pareto_em_base))
plt.step(x_staircase_smart, y_staircase_smart, where='post', label='Smart', color='#ff7f0e')
plt.scatter(np.array(pareto_vel_smart), np.array(pareto_em_smart), color='#ff7f0e')

# Set plot labels and title
plt.xlabel('Velocity [m/s]')
plt.ylabel('-CO2 emissions [mg/s]')
plt.legend(fontsize=16)
plt.grid(True)
plt.show()
plt.close()
