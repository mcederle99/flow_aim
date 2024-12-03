import numpy as np
import matplotlib.pyplot as plt

velocities = np.load('pareto_speed_cs.npy')
costs = np.load('pareto_emission_cs.npy')

plt.scatter(-velocities, costs)
plt.show()
