import numpy as np


# Simulate accelerometer readings with bias drift and noise
def sim_accelerometer(acc_true, dt, sigma_acc, sigma_bias, seed=None):
    N = len(acc_true)
    bias = np.zeros((N, 2))
    bias[0] = 0.1
    rng = np.random.default_rng(seed)
    for k in range(1, N):
        bias[k] = bias[k-1] + rng.normal(size=2) * sigma_bias * np.sqrt(dt)
    noise = rng.normal(size=(N, 2)) * sigma_acc
    f_raw = acc_true + bias + noise

    return f_raw, bias
