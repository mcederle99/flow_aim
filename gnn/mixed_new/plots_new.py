import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches


parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

num_seeds = 10
num_eval = 55
nn_architecture = "smart"

fronts_list = []
hv_list = np.zeros((num_seeds, num_eval))
crashes_list = np.zeros((num_seeds, num_eval))

for seed in range(1, 11):
    # Load the object from the file
    with open(f"results/aim_fair_{seed}_continuous_{nn_architecture}_front.pkl", "rb") as f:
        fronts_list.append(pickle.load(f))
    with open(f"results/aim_fair_{seed}_continuous_{nn_architecture}_hv.pkl", "rb") as f:
        tmp = pickle.load(f)
        while len(tmp) < 55:
            tmp.append(0)
        hv_list[seed - 1] = tmp
    with open(f"results/aim_fair_{seed}_continuous_{nn_architecture}_crashes.pkl", "rb") as f:
        tmp = pickle.load(f)
        while len(tmp) < 55:
            tmp.append(0)
        crashes_list[seed - 1] = tmp


def compute_and_plot_ema(data, alpha=0.2):
    """
    Computes the exponential moving average (EMA) of a given array and optionally plots it.

    Parameters:
        data (array-like): The input data to compute EMA on.
        alpha (float): The smoothing factor (0 < alpha ≤ 1). Smaller alpha means more smoothing.

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


# HYPERVOLUME TRAINING CURVE
mean_hv = np.mean(hv_list, axis=0)
std_hv = np.std(hv_list, axis=0)
hv_list_ema = compute_and_plot_ema(mean_hv)
# Plot the mean reward
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_eval), hv_list_ema, color="#1f77b4")
# Plot the confidence interval (mean ± std deviation)
plt.fill_between(np.arange(num_eval), hv_list_ema - std_hv*1.96, hv_list_ema + std_hv*1.96,
                 color="#1f77b4", alpha=0.2)
plt.title('Hypervolume training curve')
plt.xlabel('Evaluation run')
plt.ylabel('Hypervolume')
plt.grid(True, which='major', linestyle=':', linewidth=1, color='grey', alpha=0.7)
plt.tight_layout()
if args.save:
    plt.savefig(f"plots/hypervolume_{nn_architecture}.pdf", format='pdf')
else:
    plt.show()
plt.close()

# NUMBER OF CRASHES TRAINING CURVE
mean_crashes = np.mean(crashes_list, axis=0)
std_crashes = np.std(crashes_list, axis=0)
crashes_list_ema = compute_and_plot_ema(mean_crashes)
# Plot the mean reward
plt.figure(figsize=(10, 6))
plt.plot(np.arange(num_eval), crashes_list_ema, color="#1f77b4")
# Plot the confidence interval (mean ± std deviation)
plt.fill_between(np.arange(num_eval), max((crashes_list_ema - std_crashes*1.96).all(), 0), crashes_list_ema +
                 std_crashes*1.96,
                 color="#1f77b4", alpha=0.2)
plt.title('Number of crashes training curve')
plt.xlabel('Evaluation run')
plt.ylabel('Number of crashes')
plt.grid(True, which='major', linestyle=':', linewidth=1, color='grey', alpha=0.7)
plt.tight_layout()
if args.save:
    plt.savefig(f"plots/crashes_{nn_architecture}.pdf", format='pdf')
else:
    plt.show()
plt.close()

# PARETO FRONT PLUS K-MEANS VELOCITY VS EMISSIONS
for seed in range(10):
    best_num_sol = 0
    best_index = 0
    for i in range(len(fronts_list[seed])):
        if len(fronts_list[seed][i]) > best_num_sol and crashes_list[seed, i] <= 50:
            best_num_sol = len(fronts_list[seed][i])
            best_index = i

    pareto_front = np.array(fronts_list[seed][best_index])

    n_clusters = 3  # Number of categories
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(pareto_front)
    ordered_clusters = [clusters[len(clusters)//2], clusters[-1], clusters[0]]
    last, mid = 0, 0
    for i in range(len(clusters)):
        if clusters[i] == clusters[0]:
            last += 1
        elif clusters[i] == clusters[-1]:
            mid += 1

    pareto_vel, pareto_em = [], []
    for i in range(pareto_front.shape[0]):
        pareto_vel.append(pareto_front[i][0])
        pareto_em.append(pareto_front[i][1])

    x_staircase = np.concatenate([[pareto_vel[0]], pareto_vel])
    y_staircase = np.concatenate([pareto_em, [pareto_em[-1]]])

    # Create the figure and axis
    plt.figure(figsize=(8, 6))
    plt.step(x_staircase[-mid-1:], y_staircase[-mid-1:], where='post', color='#ff7f0e')
    plt.step(x_staircase[last:-mid], y_staircase[last:-mid], where='post', color='#1f77b4')
    plt.step(x_staircase[:last+1], y_staircase[:last+1], where='post', color='#2ca02c')
    for i in ordered_clusters:
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
        plt.savefig(f"plots/pareto_{nn_architecture}_{seed+1}.pdf", format='pdf')
    else:
        plt.show()
    plt.close()

    # BOXPLOTS + FAIRNESS DISCUSSION
    with open(f"results/boxplot_speeds_{nn_architecture}_{seed+1}.pkl", "rb") as f:
        boxplot_speeds = pickle.load(f)
    with open(f"results/boxplot_emissions_{nn_architecture}_{seed+1}.pkl", "rb") as f:
        boxplot_emissions = pickle.load(f)
    with open(f"results/delta_fairness_{nn_architecture}_{seed+1}.pkl", "rb") as f:
        delta_fairness = pickle.load(f)

    indexed_bs = list(enumerate(boxplot_speeds))
    sorted_indexed_bs = sorted(indexed_bs, key=lambda x: np.median(x[1]))
    original_indexes_bs = [idx for idx, _ in sorted_indexed_bs]

    new_boxplot_emissions = []
    new_boxplot_speeds = []
    for i in original_indexes_bs:
        new_boxplot_emissions.append(boxplot_emissions[i])
        new_boxplot_speeds.append(boxplot_speeds[i])

    new_delta_fairness_a = []
    for i in original_indexes_bs:
        new_delta_fairness_a.append(delta_fairness[i])
    x = np.arange(len(delta_fairness))
    y = new_delta_fairness_a

    n_clusters = 2  # Number of categories
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(np.array(y).reshape(-1, 1))
    plt.figure(figsize=(8, 6))
    colors = ['red', 'green']
    for i in range(len(y)):
        plt.scatter(x[i], y[i], label=f"Cluster {i + 1}", c=colors[clusters[i]])
    plt.show()
    plt.close()

    # Fit a parabola (degree 2) to the data
    coefficients = np.polyfit(x, y, deg=2)  # Returns coefficients [a, b, c] for ax^2 + bx + c
    parabola = np.poly1d(coefficients)  # Create a polynomial function from the coefficients
    # Generate smooth curve for the parabola
    x_smooth = np.linspace(min(x), max(x), 500)
    y_smooth = parabola(x_smooth)
    y_fitted = parabola(x)  # Fitted y-values at original x points
    residuals = y - y_fitted  # Residuals
    std_error = np.sqrt(np.sum(residuals ** 2) / (len(y) - len(coefficients)))
    # Confidence interval (95%)
    from scipy.stats import t
    alpha = 0.025  # Significance level
    t_value = t.ppf(1 - alpha / 2, df=len(y) - len(coefficients))  # t-value for 95% CI
    # Confidence interval bounds
    confidence_interval = t_value * std_error
    y_upper = y_smooth + confidence_interval
    y_lower = y_smooth - confidence_interval

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    # Plot the original data points
    ax.scatter(x, y, color="blue", label="Data Points", zorder=3)
    # Plot the fitted parabola
    ax.plot(x_smooth, y_smooth, color="orange", label=f"Fitted Parabola: {parabola}")
    # Plot the confidence interval
    ax.fill_between(
        x_smooth, y_lower, y_upper, color="orange", alpha=0.3, label="97.5% Confidence Interval"
    )
    # Customize the plot
    ax.set_title("Average $\Delta$ travel time between fuel and electric vehicles")
    ax.set_xlabel("Pareto efficient solutions")
    ax.set_ylabel("Time [s]")
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([])
    # ax.legend()
    ax.grid(True, which='major', linestyle=':', linewidth=1, color='grey', alpha=0.7)
    if args.save:
        plt.savefig(f"plots/delta_time_{nn_architecture}_{seed+1}.pdf", format='pdf')
    else:
        plt.show()
    plt.close()

    fairest_solution = np.argmin(new_delta_fairness_a) + 1

    _, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.boxplot(new_boxplot_speeds, vert=True, patch_artist=True)
    # plt.axvline(x=fairest_solution, color="red", linestyle="--", linewidth=2)
    ax.grid(True, which='major', linestyle=':', linewidth=1, color='grey', alpha=0.7)
    ax.set_xlabel("Pareto efficient solutions")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_xticklabels([])
    ax.set_title("Velocities distribution at test time")
    if args.save:
        plt.savefig(f"plots/boxplot_vel_{nn_architecture}_{seed+1}.pdf", format='pdf')
    else:
        plt.show()
    plt.close()

    _, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.boxplot(new_boxplot_emissions, vert=True, patch_artist=True)
    # plt.axvline(x=fairest_solution, color="red", linestyle="--", linewidth=2)
    ax.grid(True, which='major', linestyle=':', linewidth=1, color='grey', alpha=0.7)
    ax.set_xlabel("Pareto efficient solutions")
    ax.set_ylabel("CO2 emissions [mg/s]")
    ax.set_xticklabels([])
    ax.set_title("Emissions distribution at test time")
    if args.save:
        plt.savefig(f"plots/boxplot_em_{nn_architecture}_{seed+1}.pdf", format='pdf')
    else:
        plt.show()
    plt.close()
