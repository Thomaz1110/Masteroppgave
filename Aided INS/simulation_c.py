import numpy as np
import matplotlib.pyplot as plt

from robot_c import Robot, generate_robot_pair_groups, initialize_robot_positions
from eskf_c import ESKFMultiRobot
import plotting_c as ins_plot
from config_c import (
    dt,
    sigma_acc,
    sigma_bias,
    sigma_vel,
    sigma_range,
    initial_pos_radius,
    initial_pos_var_robot,
    initial_bias_var,
    trajectory_seed_base,
    imu_seed_base,
    initial_bias_seed_base,
    range_seed,
    grid_x_limits,
    grid_y_limits,
)


vel_threshold = 0.5                 # dominance threshold for dominant-axis detection
dominant_axis_method = "true"       # "true" (use ground-truth velocity) or "nominal" (use estimated velocity + threshold)
velocity_update_rate_hz = 10.0      # [Hz] dominant-axis zero-velocity update rate
beacon_range_rate_hz = 1.0           # [Hz] robot0-to-beacon range rate
robot_range_rate_hz = 1.0           # [Hz] robot-to-robot range rate per pair group
range_measurement_stop_time = None  # seconds; None => entire run
standstill_time = 20.0              # [s] initial standstill period for calibration

use_virtual_measurements = True      # If True, use virtual measurements: dominant-axis velocity updates and initial standstill velocity updates
beacon_ranging = False              # robot0-to-beacon ranging
robot_ranging = True               # robot-to-robot ranging

use_true_initial_position = True        # True => all robots start at true positions
robot0_knows_initial_position = True    # If True, robot 0 starts at true position even when others don't

num_robots = 2
duration_s = 500.0

plot_acc = 0
plot_vel = 0
plot_pos = 1
plot_bias = 1

plot_mean_covariance_comparison = True
average_joint_covariance_num_samples = 5
ci_inflation_omega = 0.5



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
        initial_bias_seed=initial_bias_seed_base + idx,
    )
    robots.append(robot)

initialize_robot_positions(
    robots,
    use_true_initial_position,
    robot0_knows_initial_position,
    initial_pos_radius,
    grid_x_limits=grid_x_limits,
    grid_y_limits=grid_y_limits,
)


t = robots[0].t
N = len(t)
if any(robot.N != N for robot in robots):
    raise ValueError("All robots must have trajectories of equal length")


# Initialize ESKF for multi-robot system
kf = ESKFMultiRobot(
    dt,
    sigma_acc,
    sigma_bias,
    sigma_vel,
    sigma_range,
    num_robots,
)

# Initialize covariance P
for idx in range(num_robots):
    use_true = use_true_initial_position or (idx == 0 and robot0_knows_initial_position)
    offset = idx * 6
    if use_true:
        kf.P[offset + 0, offset + 0] = 10e-3
        kf.P[offset + 1, offset + 1] = 10e-3
    else:
        kf.P[offset + 0, offset + 0] = initial_pos_var_robot
        kf.P[offset + 1, offset + 1] = initial_pos_var_robot
    kf.P[offset + 4, offset + 4] = initial_bias_var
    kf.P[offset + 5, offset + 5] = initial_bias_var

covariance_averager = ins_plot.initialize_joint_covariance_averager(
    plot_mean_covariance_comparison,
    duration_s,
    average_joint_covariance_num_samples,
    kf.P,
    num_robots,
    ci_inflation_omega,
)

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

# Determine robot-to-robot range measurement interval
if robot_ranging and robot_range_rate_hz > 0.0:
    robot_range_interval_steps = max(1, int(round(1.0 / (robot_range_rate_hz * dt))))
else:
    robot_range_interval_steps = None

range_rngs = [
    np.random.default_rng(range_seed + idx) for idx in range(num_robots)
]
current_beacon_index = 0

# Generate robot pair groups for robot-to-robot ranging, returning list of list of (initiator_idx, reflector_idx) tuples
robot_pair_groups = generate_robot_pair_groups(num_robots)
current_robot_pair_group_index = 0


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

    kf.predict()

    if use_virtual_measurements:
        # Velocity updates based on dominant axis
        if (
            velocity_update_interval_steps is not None
            and (k % velocity_update_interval_steps == 0)
        ):
            for idx, robot in enumerate(robots):
                dominant_axis = robot.determine_dominant_axis(k)
                if dominant_axis == "x":
                    y_meas = np.array([0.0])
                    y_ins_hat = np.array([robot.v_nominal[k, 1]])
                    kf.update(idx, "velocity_y", y_meas, y_ins_hat)
                elif dominant_axis == "y":
                    y_meas = np.array([0.0])
                    y_ins_hat = np.array([robot.v_nominal[k, 0]])
                    kf.update(idx, "velocity_x", y_meas, y_ins_hat)

    
        # Initial standstill velocity calibration updates (both axes)
        if t[k] <= standstill_time:
            for idx, robot in enumerate(robots):
                # Velocity calibration updates (both axes)
                y_meas = np.array([0.0, 0.0])
                y_ins_hat = np.array([robot.v_nominal[k, 0], robot.v_nominal[k, 1]])
                kf.update(idx, "velocity_calibration", y_meas, y_ins_hat)

        # applying and resetting corrections after every type of update
        corrections = kf.get_state_correction()         # Get state corrections for all robots
        for idx, robot in enumerate(robots):            
            robot.apply_correction(k, corrections[idx]) # Apply corrections to each robot's nominal state
        kf.reset_error_state()                          # Reset error state after applying corrections!


   

    # Range updates
    beacon_due = (
        beacon_range_interval_steps is not None
        and (k % beacon_range_interval_steps == 0)
        and (range_measurement_stop_time is None or t[k] <= range_measurement_stop_time)
    )

    robot_due = (
        robot_range_interval_steps is not None
        and (k % robot_range_interval_steps == 0)
        and (range_measurement_stop_time is None or t[k] <= range_measurement_stop_time)
    )

    # Robot0-to-beacon range updates
    if beacon_due:
        beacon = beacons_all[current_beacon_index]
        robot0 = robots[0]
        initiator_true = np.array([robot0.pos_true[k, 0], robot0.pos_true[k, 1], 0.0])
        initiator_nominal = np.array([robot0.p_nominal[k, 0], robot0.p_nominal[k, 1], 0.0])
        reflector_true = beacon
        reflector_nominal = beacon
        diff_true = initiator_true - reflector_true
        diff_nominal = initiator_nominal - reflector_nominal
        y_meas = np.linalg.norm(diff_true) + range_rngs[0].normal(scale=sigma_range)
        y_meas = max(y_meas, 0.0)
        y_ins_hat = np.linalg.norm(diff_nominal)
        kf.update(
            0,
            "beacon_range",
            y_meas,
            y_ins_hat,
            initiator_nom=initiator_nominal,
            reflector_nom=reflector_nominal,
        )
        current_beacon_index = (current_beacon_index + 1) % len(beacons_all)

        # applying and resetting corrections after every type of update 
        corrections = kf.get_state_correction()         # Get state corrections for all robots
        for idx, robot in enumerate(robots):            
            robot.apply_correction(k, corrections[idx]) # Apply corrections to each robot's nominal state
        kf.reset_error_state()                          # Reset error state after applying corrections

    ins_plot.update_joint_covariance_averager(covariance_averager, t[k], kf.P)


    # Robot-to-robot range updates
    if robot_due and robot_pair_groups:
        pair_group = robot_pair_groups[current_robot_pair_group_index]
        for initiator_idx, reflector_idx in pair_group:
            if beacon_due and (initiator_idx == 0 or reflector_idx == 0):
                continue                                                        # Skip if beacon range already done for robot 0 for this epoch
            initiator_robot = robots[initiator_idx]
            reflector_robot = robots[reflector_idx]
            initiator_true = np.array([initiator_robot.pos_true[k, 0], initiator_robot.pos_true[k, 1], 0.0])
            initiator_nominal = np.array([initiator_robot.p_nominal[k, 0], initiator_robot.p_nominal[k, 1], 0.0])
            reflector_true = np.array([reflector_robot.pos_true[k, 0], reflector_robot.pos_true[k, 1], 0.0])
            reflector_nominal = np.array([reflector_robot.p_nominal[k, 0], reflector_robot.p_nominal[k, 1], 0.0])
            diff_true = initiator_true - reflector_true
            diff_nominal = initiator_nominal - reflector_nominal
            y_meas = np.linalg.norm(diff_true) + range_rngs[initiator_idx].normal(scale=sigma_range)
            y_meas = max(y_meas, 0.0)
            y_ins_hat = np.linalg.norm(diff_nominal)
            kf.update(
                initiator_idx,
                "robot_range",
                y_meas,
                y_ins_hat,
                initiator_nom=initiator_nominal,
                reflector_nom=reflector_nominal,
                reflector_index=reflector_idx,
            )
        current_robot_pair_group_index = (current_robot_pair_group_index + 1) % len(robot_pair_groups)

        # applying and resetting corrections after every type of update
        corrections = kf.get_state_correction()         # Get state corrections for all robots
        for idx, robot in enumerate(robots):            
            robot.apply_correction(k, corrections[idx]) # Apply corrections to each robot's nominal state
        kf.reset_error_state()                          # Reset error state after applying corrections


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

ins_plot.finalize_joint_covariance_averager(covariance_averager)

plt.show()
