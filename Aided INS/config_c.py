dt = 0.01                           # sample period [s]

# Noise parameters
sigma_bias = 9.6e-4                 # accelerometer bias standard deviation (stationary bias level) [m/s^2]
sigma_acc = 9.6e-3                  # accelerometer white-noise standard deviation (per sample) [m/s^2]
sigma_vel = 10e-3                   # velocity measurement noise std [m/s]
sigma_range = 0.5                   # range measurement noise std [m]

# Initial state uncertainty/config
initial_pos_radius = 5.0            # [m] radius for random initial offset
initial_pos_var_robot = 5**2        # covariance for uncertain initial positions
initial_bias_var = 0.1              # covariance for initial accelerometer bias

# Random seeds
trajectory_seed_base = 5001         # base seed for trajectory generation; each robot gets a unique seed by adding its ID
imu_seed_base = 1002                # base seed for IMU noise generation; each robot gets a unique seed by adding its ID
initial_bias_seed_base = 4002       # base seed for initial accelerometer bias generation; each robot gets a unique seed by adding its ID
range_seed = 3002                   # seed for beacon range measurement noise

# Environment limits
grid_x_limits = (0.0, 35.0)
grid_y_limits = (0.0, 20.0)
