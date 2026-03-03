import numpy as np


dt = 0.01                           # sample period [s]
sigma_acc = 0.05                    # accelerometer noise std [m/s^2]
sigma_bias = 0.003                  # accelerometer bias driving noise std [m/s^2/sqrt(s)]
sigma_vel = 10e-3                   # velocity measurement noise std [m/s]
sigma_range = 0.5                   # range measurement noise std [m]
vel_threshold = 0.5                 # dominance threshold for dominant-axis detection
dominant_axis_method = "true"       # "true" (use ground-truth velocity) or "nominal" (use estimated velocity + threshold)
velocity_update_rate_hz = 10.0      # [Hz] dominant-axis zero-velocity update rate
beacon_range_rate_hz = 5.0          # [Hz] robot-to-beacon range rate
range_measurement_stop_time = None  # seconds; None => entire run
standstill_time = 20.0              # [s] initial standstill period for calibration

use_virtual_measurements = True     # If True, use virtual measurements: dominant-axis velocity updates and initial standstill velocity updates
beacon_ranging = True               # robot-to-beacon ranging

use_true_initial_position = True    # True => all robots start at true positions
initial_pos_radius = 5.0            # [m] radius for random initial offset
initial_pos_var_robot = 5**2        # covariance for uncertain initial positions
initial_bias_var = 0.1              # covariance for initial accelerometer bias

num_robots = 1
duration_s = 300.0
trajectory_seed_base = 5001
imu_seed_base = 1002
range_seed = 3000

plot_acc = 0
plot_vel = 0
plot_pos = 1
plot_bias = 1

grid_x_limits = (0.0, 35.0)             # used for initial position guessing and plotting
grid_y_limits = (0.0, 20.0)             # used for initial position guessing and plotting


# Define beacons
beacon_height = 2.0  # [m]
beacons_all = np.array([
    [2.5, 2.5, beacon_height],          # Bottom-left
    [2.5, 18.5, beacon_height],         # Top-left
    [17.5, 10.0, beacon_height],        # Center
    [33.5, 2.5, beacon_height],         # Bottom-right
    [33.5, 18.5, beacon_height],        # Top-right
])
