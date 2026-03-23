"""
IMU accelerometer simulation utilities.

Contains:
- "sim_accelerometer": simulates 2D accelerometer measurements from true acceleration
  by adding a random-walk bias and white measurement noise.
"""

import numpy as np


# # Simulate accelerometer readings with bias drift and noise
# def sim_accelerometer(acc_true, dt, sigma_acc, sigma_bias, seed=None):
#     N = len(acc_true)
#     bias = np.zeros((N, 2))
#     bias[0] = 0.1
#     rng = np.random.default_rng(seed)
#     for k in range(1, N):
#         bias[k] = bias[k-1] + rng.normal(size=2) * sigma_bias * np.sqrt(dt)
#     noise = rng.normal(size=(N, 2)) * sigma_acc
#     f_raw = acc_true + bias + noise

#     return f_raw, bias


def sim_accelerometer(acc_true, dt, T_acc=300.0, sigma_bias=9.6e-4, sigma_acc=9.6e-3, seed=None):
  rng = np.random.default_rng(seed)
  N = len(acc_true) 
  bias = np.zeros((N, 2)) 

  phi = np.exp(-dt / T_acc)                               # bias decay factor per step; if T_acc=inf, then phi=1 and we have a pure RW bias
  Qb = sigma_bias**2 * (1.0 - np.exp(-2.0 * dt / T_acc))  # bias driving noise variance per step; chosen so that the resulting bias process has stationary variance sigma_bias^2

  for k in range(1, N):
    bias[k] = phi * bias[k - 1] + np.sqrt(Qb) * rng.normal(size=2)

  noise = sigma_acc * rng.normal(size=(N, 2))
  f_imu = acc_true + bias + noise

  return f_imu, bias
