import numpy as np
import matplotlib.pyplot as plt

velocities = np.load('pareto_speed_cb.npy')
costs = np.load('pareto_emission_cb.npy')

plt.scatter(-velocities, costs)
plt.show()
