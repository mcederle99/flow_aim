import numpy as np
import matplotlib.pyplot as plt

res = np.load('aim_0_manualflow_noidle_fixrv.npy')


def calculate_ema(values, spann):
    ema = np.zeros_like(values)
    alpha = 2 / (spann + 1)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    return ema


# Calculate the EMA
span = 20
ema_values = calculate_ema(res, span)
plt.plot(ema_values)
plt.show()
