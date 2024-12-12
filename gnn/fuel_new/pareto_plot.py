import numpy as np
import matplotlib.pyplot as plt


front_base = np.load('pareto_front_continuous_base.npy')
front_smart = np.load('pareto_front_continuous_smart.npy')

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
plt.step(x_staircase_base, y_staircase_base, where='post', label='A')
plt.scatter(np.array(pareto_vel_base), np.array(pareto_em_base))
plt.step(x_staircase_smart, y_staircase_smart, where='post', label='B', color='#ff7f0e')
plt.scatter(np.array(pareto_vel_smart), np.array(pareto_em_smart), color='#ff7f0e')

# Set plot labels and title
plt.xlabel('Velocity [m/s]')
plt.ylabel('-CO2 emissions [mg/s]')
plt.legend(fontsize=16)
plt.grid(True)
plt.show()
