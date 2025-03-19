import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches


parser = argparse.ArgumentParser()
parser.add_argument("--nn_architecture", default="smart")
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

fronts_list, hv_list, crashes_list = [], [], []
for seed in range(1, 11):
    # Load the object from the file
    with open(f"results/aim_{seed}_continuous_{args.nn_architecture}_fixom_front.pkl", "rb") as f:
        fronts_list.append(pickle.load(f))
    with open(f"results/aim_{seed}_continuous_{args.nn_architecture}_fixom_hv.pkl", "rb") as f:
        hv_list.append(pickle.load(f))
    with open(f"results/aim_{seed}_continuous_{args.nn_architecture}_fixom_crashes.pkl", "rb") as f:
        crashes_list.append(pickle.load(f))

for i in range(10):
    # assert len(fronts_list[0]) == len(fronts_list[i+1])
    # assert len(hv_list[0]) == len(hv_list[i + 1])
    # assert len(crashes_list[0]) == len(crashes_list[i + 1])
    print(len(fronts_list[i]))

raise KeyboardInterrupt


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

# HYPERVOLUME TRAINING CURVE
plt.figure(figsize=(10, 6))
plt.plot(hv_list_ema, color='blue', linewidth=2)
plt.title('Hypervolume training curve')
plt.xlabel('Evaluation run')
plt.ylabel('Hypervolume')
plt.grid(True)
if args.save:
    plt.savefig(f"plots/hypervolume_{args.nn_architecture}_{args.seed}.pdf", format='pdf')
plt.show()
plt.close()

# NUMBER OF CRASHES TRAINING CURVE
plt.figure(figsize=(10, 6))
plt.plot(crashes_list_ema, color='red', linewidth=2)
plt.title('Number of crashes training curve')
plt.xlabel('Evaluation run')
plt.ylabel('Number of crashes')
plt.grid(True)
if args.save:
    plt.savefig(f"plots/crashes_{args.nn_architecture}_{args.seed}.pdf", format='pdf')
plt.show()
plt.close()

# PARETO FRONT PLUS K-MEANS VELOCITY VS EMISSIONS
best_num_sol = 0
best_index = 0
for i in range(len(fronts_list)):
    # print(len(fronts_list[i]))
    if len(fronts_list[i]) > best_num_sol and crashes_list[i] <= 50:
        best_num_sol = len(fronts_list[i])
        best_index = i

pareto_front = np.array(fronts_list[best_index])
print(len(pareto_front))

n_clusters = 3  # Number of categories
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(pareto_front)

pareto_vel, pareto_em = [], []
for i in range(pareto_front.shape[0]):
    pareto_vel.append(pareto_front[i][0])
    pareto_em.append(pareto_front[i][1])

x_staircase = np.concatenate([[pareto_vel[0]], pareto_vel])
y_staircase = np.concatenate([pareto_em, [pareto_em[-1]]])

# Create the figure and axis
plt.figure(figsize=(8, 6))
plt.step(x_staircase[-18:], y_staircase[-18:], where='post', color='#ff7f0e')
plt.step(x_staircase[9:-17], y_staircase[9:-17], where='post', color='#1f77b4')
plt.step(x_staircase[:10], y_staircase[:10], where='post', color='#2ca02c')
for i in range(n_clusters):
    plt.scatter(
        pareto_front[clusters == i, 0], pareto_front[clusters == i, 1], label=f"Cluster {i+1}"
    )

blue_patch = mpatches.Patch(color='#ff7f0e', label='Emission saving solutions')
red_patch = mpatches.Patch(color='#1f77b4', label=r'Balanced solutions')
green_patch = mpatches.Patch(color='#2ca02c', label=r'Performance based solutions')
plt.legend(handles=[blue_patch, red_patch, green_patch])

# Set plot labels and title
plt.xlabel('Velocity [m/s]')
plt.ylabel('-CO2 emissions [mg/s]')
plt.title('Pareto front velocity vs emissions')
plt.grid(True)
if args.save:
    plt.savefig(f"plots/pareto_{args.nn_architecture}_{args.seed}.pdf", format='pdf')
plt.show()
plt.close()

print(crashes_list[best_index])

# PARETO FRONTS COMPARISON
with open(f"results/aim_{args.seed}_continuous_base_fixom_front.pkl", "rb") as f:
    fronts_list_base = pickle.load(f)
with open(f"results/aim_{args.seed}_continuous_smart_fixom_front.pkl", "rb") as f:
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
plt.step(x_staircase_base, y_staircase_base, where='post', label='Architecture A', color='red')
plt.scatter(np.array(pareto_vel_base), np.array(pareto_em_base), color='red')
plt.step(x_staircase_smart, y_staircase_smart, where='post', label='Architecture B', color='green')
plt.scatter(np.array(pareto_vel_smart), np.array(pareto_em_smart), color='green')

# Set plot labels and title
plt.xlabel('Velocity [m/s]')
plt.ylabel('-CO2 emissions [mg/s]')
plt.title('Comparison between Pareto fronts')
plt.legend(fontsize=16)
plt.grid(True)
if args.save:
    plt.savefig(f"plots/pareto_comparison_{args.seed}.pdf", format='pdf')
plt.show()
plt.close()

# BOXPLOTS
with open(f"results/boxplot_speeds_{args.nn_architecture}_{args.seed}.pkl", "rb") as f:
    boxplot_speeds = pickle.load(f)
with open(f"results/boxplot_emissions_{args.nn_architecture}_{args.seed}.pkl", "rb") as f:
    boxplot_emissions = pickle.load(f)

indexed_bs = list(enumerate(boxplot_speeds))
sorted_indexed_bs = sorted(indexed_bs, key=lambda x: np.median(x[1]))
original_indexes_bs = [idx for idx, _ in sorted_indexed_bs]

new_boxplot_emissions = []
new_boxplot_speeds = []
for i in original_indexes_bs:
    new_boxplot_emissions.append(boxplot_emissions[i])
    new_boxplot_speeds.append(boxplot_speeds[i])

_, ax = plt.subplots(figsize=(10, 6), dpi=100)
ax.boxplot(new_boxplot_speeds, vert=True, patch_artist=True)
ax.grid(True, which='major', linestyle=':', linewidth=1, color='grey', alpha=0.7)
ax.set_xlabel("Pareto efficient solutions")
ax.set_ylabel("Velocity [m/s]")
ax.set_xticklabels([])
ax.set_title("Velocities distribution at test time")
if args.save:
    plt.savefig(f"plots/boxplot_vel_{args.nn_architecture}_{args.seed}.pdf", format='pdf')
plt.show()
plt.close()

_, ax = plt.subplots(figsize=(10, 6), dpi=100)
ax.boxplot(new_boxplot_emissions, vert=True, patch_artist=True)
ax.grid(True, which='major', linestyle=':', linewidth=1, color='grey', alpha=0.7)
ax.set_xlabel("Pareto efficient solutions")
ax.set_ylabel("CO2 emissions [mg/s]")
ax.set_xticklabels([])
ax.set_title("Emissions distribution at test time")
if args.save:
    plt.savefig(f"plots/boxplot_em_{args.nn_architecture}_{args.seed}.pdf", format='pdf')
plt.show()
plt.close()
