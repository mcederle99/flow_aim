import numpy as np
import matplotlib.pyplot as plt
from utils import compute_pareto_front

velocities = np.load('pareto_speed_discrete_smart.npy')
costs = np.load('pareto_emission_discrete_smart.npy')

plt.scatter(velocities, -costs)
plt.xlabel("-Velocity [m/s]")
plt.ylabel("CO2 emissions [mg/s]")
plt.grid()
plt.show()
input("")
plt.close()

pareto_front = []
for i in range(len(velocities)):
    pareto_front.append([-velocities[i], costs[i]])
front = compute_pareto_front(pareto_front)

pareto_vel, pareto_em = [], []
for i in range(len(front)):
    pareto_vel.append(front[i][0])
    pareto_em.append(front[i][1])

plt.scatter(-np.array(pareto_vel), -np.array(pareto_em))
plt.xlabel("-Velocity [m/s]")
plt.ylabel("CO2 emissions [mg/s]")
plt.grid()
plt.show()
