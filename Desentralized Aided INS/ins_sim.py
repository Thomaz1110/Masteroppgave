import numpy as np
import matplotlib.pyplot as plt

from robot import Robot, initialize_robot_positions
import ins_plot


dt = 0.01                           # sample period [s]
sigma_acc = 0.05                    # accelerometer noise std [m/s^2]
sigma_bias = 0.003                  # accelerometer bias driving noise std [m/s^2/sqrt(s)]
sigma_vel = 10e-3                   # velocity measurement noise std [m/s]
sigma_range = 0.5                   # range measurement noise std [m]
vel_threshold = 0.5                 # dominance threshold for dominant-axis detection
dominant_axis_method = "true"       # "true" (use ground-truth velocity) or "nominal" (use estimated velocity + threshold)
velocity_update_rate_hz = 10.0      # [Hz] dominant-axis zero-velocity update rate
beacon_range_rate_hz = 5.0           # [Hz] robot0-to-beacon range rate
range_measurement_stop_time = None  # seconds; None => entire run
standstill_time = 20.0              # [s] initial standstill period for calibration

use_virtual_measurments = True      # If True, use virtual measurements: dominant-axis velocity updates and initial standstill velocity updates
beacon_ranging = True              # robot0-to-beacon ranging

use_true_initial_position = True        # True => all robots start at true positions
robot0_knows_initial_position = True    # If True, robot 0 starts at true position even when others don't
initial_pos_radius = 5.0                # [m] radius for random initial offset
initial_pos_var_robot = 5**2            # covariance for uncertain initial positions
initial_bias_var = 0.1                  # covariance for initial accelerometer bias

num_robots = 2
duration_s = 300.0
trajectory_seed_base = 5001
imu_seed_base = 1002
range_seed = 3000

plot_acc = 0
plot_vel = 0
plot_pos = 1
plot_bias = 1




robots = []
for idx in range(num_robots):
    robot = Robot(
        robot_id=idx,
        dt=dt,
        sigma_acc=sigma_acc,
        sigma_bias=sigma_bias,
        vel_threshold=vel_threshold,
        dominant_axis_method=dominant_axis_method,
        standstill_time=standstill_time,
        duration_s=duration_s,
        trajectory_seed=trajectory_seed_base + idx,
        imu_seed=imu_seed_base + idx,
    )
    robots.append(robot)

initialize_robot_positions(
    robots,
    use_true_initial_position,
    robot0_knows_initial_position,
    initial_pos_radius,
    grid_x_limits=(0.0, 35.0),
    grid_y_limits=(0.0, 20.0),
)


t = robots[0].t
N = len(t)
if any(robot.N != N for robot in robots):
    raise ValueError("All robots must have trajectories of equal length")


# Initialize covariance P
for idx in range(num_robots):
    use_true = use_true_initial_position or (idx == 0 and robot0_knows_initial_position)
    kf = robots[idx].eskf
    if use_true:
        kf.P[0, 0] = 10e-3
        kf.P[1, 1] = 10e-3
    else:
        kf.P[0, 0] = initial_pos_var_robot
        kf.P[1, 1] = initial_pos_var_robot
    kf.P[4, 4] = initial_bias_var
    kf.P[5, 5] = initial_bias_var

# Determine dominant-axis velocity update intervals in steps
if velocity_update_rate_hz > 0.0:
    velocity_update_interval_steps = max(1, int(round(1.0 / (velocity_update_rate_hz * dt))))
else:
    velocity_update_interval_steps = None

# Determine beacon range measurement interval
if beacon_ranging and beacon_range_rate_hz > 0.0:
    beacon_range_interval_steps = max(1, int(round(1.0 / (beacon_range_rate_hz * dt))))
else:
    beacon_range_interval_steps = None

range_rng = np.random.default_rng(range_seed)
current_beacon_index = 0


# Define beacons
beacon_height = 2.0  # [m]
beacons_all = np.array([
    [2.5, 2.5, beacon_height],          # Bottom-left
    [2.5, 18.5, beacon_height],         # Top-left
    [17.5, 10.0, beacon_height],        # Center
    [33.5, 2.5, beacon_height],         # Bottom-right
    [33.5, 18.5, beacon_height],        # Top-right
])


for k in range(1, N):
    for robot in robots:
        robot.propagate_nominal(k)
        robot.eskf.predict(acc_meas_2d=robot.f_imu[k - 1], dt=dt)

    if use_virtual_measurments:
        for robot in robots:
            updated = False

            # Velocity updates based on dominant axis
            if (
                velocity_update_interval_steps is not None
                and (k % velocity_update_interval_steps == 0)
            ):
                dominant_axis = robot.determine_dominant_axis(k)
                if dominant_axis == "x":
                    z = 0.0 - robot.v_nominal[k, 1]
                    robot.eskf.update_velocity_y(z, sigma_vel**2)
                    updated = True
                elif dominant_axis == "y":
                    z = 0.0 - robot.v_nominal[k, 0]
                    robot.eskf.update_velocity_x(z, sigma_vel**2)
                    updated = True

            # Initial standstill velocity calibration updates (both axes)
            if t[k] <= standstill_time:
                z2 = np.array([0.0 - robot.v_nominal[k, 0], 0.0 - robot.v_nominal[k, 1]])
                R2 = np.diag([sigma_vel**2, sigma_vel**2])
                robot.eskf.update_velocity_calibration(z2, R2)
                updated = True

            if updated:
                robot.apply_filter_correction(k)


   

    # Range updates
    beacon_due = (
        beacon_range_interval_steps is not None
        and (k % beacon_range_interval_steps == 0)
        and (range_measurement_stop_time is None or t[k] <= range_measurement_stop_time)
    )

    # Robot0-to-beacon range updates
    if beacon_due:
        beacon = beacons_all[current_beacon_index]
        robot0 = robots[0]
        initiator_true = np.array([robot0.pos_true[k, 0], robot0.pos_true[k, 1], 0.0])
        initiator_nominal_2d = np.array([robot0.p_nominal[k, 0], robot0.p_nominal[k, 1]])
        reflector_true = beacon
        diff_true = initiator_true - reflector_true
        y_meas = np.linalg.norm(diff_true) + range_rng.normal(scale=sigma_range)
        y_meas = max(y_meas, 0.0)
        robot0.eskf.update_beacon_range(
            z_range=y_meas,
            beacon_pos_2d=beacon[:2],
            nominal_pos_2d=initiator_nominal_2d,
            R=sigma_range**2,
        )
        robot0.apply_filter_correction(k)
        current_beacon_index = (current_beacon_index + 1) % len(beacons_all)


# Plotting
if plot_acc:
    for idx, robot in enumerate(robots):
        ins_plot.plot_acceleration(robot.t, robot.acc_true, robot.f_imu, robot_id=idx)

if plot_vel:
    for idx, robot in enumerate(robots):
        ins_plot.plot_velocity(robot.t, robot.vel_true, robot.v_nominal, robot_id=idx)

if plot_pos:
    for idx, robot in enumerate(robots):
        ins_plot.plot_positions(
            robot.t,
            robot.pos_true,
            robot.p_nominal,
            "random",
            beacons=beacons_all if beacon_ranging else None,
            standstill_time=standstill_time,
            robot_id=idx,
            total_robots=num_robots,
        )

if plot_bias:
    for idx, robot in enumerate(robots):
        ins_plot.plot_bias(
            robot.t,
            robot.bias_true,
            robot.b_nominal,
            None,
            robot_id=idx,
            total_robots=num_robots,
        )

plt.show()
