import numpy as np
import matplotlib.pyplot as plt

from robot import Robot, initialize_robot_positions
import plotting as ins_plot
from config import (
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


num_robots = 2
duration_s = 500.0
standstill_time = 20.0                  # [s] initial standstill period for calibration
use_true_initial_position = True        # True => all robots start at true positions
trajectory_mode = "random"              # "random", "parallel_x", or "parallel_y"
parallel_track_spacing_lines = 10       # separation in grid lines between the two parallel tracks


vel_threshold = 0.5                     # [m/s] velocity threshold for dominant-axis detection
dominant_axis_method = "true"           # "true" (use ground-truth velocity) or "nominal" (use estimated velocity + threshold)
velocity_update_rate_hz = 10.0          # [Hz] dominant-axis zero-velocity update rate


use_virtual_measurements = True         # If True, use virtual measurements: dominant-axis velocity updates and initial standstill velocity updates
beacon_ranging = False                  # robot-to-beacon ranging
robot_ranging = True                    # robot-to-robot ranging
coop_type = "mutualistic"               # "mutualistic" or "commensalistic" cooperative range updates for robot-to-robot ranging
force_uncorrelated_robot_range = True   # If True, ignore cooperative-history correlation and treat robot pairs as uncorrelated

beacon_range_rate_hz = 1.0              # [Hz] robot-to-beacon range rate
robot_range_rate_hz = 1.0               # [Hz] robot-to-robot range rate
range_measurement_stop_time = None      # seconds; None => entire run

plot_acc = 0
plot_vel = 0
plot_pos = 1
plot_bias = 1
plot_pos_live = False
plot_pos_live_every_n_steps = 20



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
        trajectory_mode=trajectory_mode,
        parallel_track_spacing_lines=parallel_track_spacing_lines,
    )
    robots.append(robot)

initialize_robot_positions(
    robots,
    use_true_initial_position,
    initial_pos_radius,
    grid_x_limits=grid_x_limits,
    grid_y_limits=grid_y_limits,
)


t = robots[0].t
N = len(t)
if any(robot.N != N for robot in robots):
    raise ValueError("All robots must have trajectories of equal length")


# Initialize covariance P
for idx in range(num_robots):
    use_true = use_true_initial_position
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

# Determine robot range measurement interval
if robot_ranging and robot_range_rate_hz > 0.0:
    robot_range_interval_steps = max(1, int(round(1.0 / (robot_range_rate_hz * dt))))
else:
    robot_range_interval_steps = None

range_rngs = [
    np.random.default_rng(range_seed + idx) for idx in range(num_robots)
]
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


robot0_is_next_initiator = True  # Alternate which robot acts as initiator for robot-to-robot ranging

if plot_pos_live:
    for idx, robot in enumerate(robots):
        ins_plot.init_live_position_plot(
            robot.pos_true,
            robot.p_nominal,
            beacons=beacons_all if beacon_ranging else None,
            robot_id=idx,
            grid_x_limits=(0.0, 35.0),
            grid_y_limits=(0.0, 21.0),
        )

for k in range(1, N):
    for robot in robots:
        robot.propagate_nominal(k)
        robot.eskf.predict()

    if use_virtual_measurements:
        for robot in robots:
            updated = False

            # Velocity updates based on dominant axis
            if (
                velocity_update_interval_steps is not None
                and (k % velocity_update_interval_steps == 0)
            ):
                dominant_axis = robot.determine_dominant_axis(k)
                if dominant_axis in {"x", "y"}:
                    non_dominant_idx = 1 if dominant_axis == "x" else 0
                    y_meas = 0.0
                    y_ins_hat = robot.v_nominal[k, non_dominant_idx]
                    robot.eskf.update_dominant_axis_velocity(
                        dominant_axis, y_meas, y_ins_hat, sigma_vel**2
                    )
                    updated = True

            # Initial standstill velocity calibration updates (both axes)
            if t[k] <= standstill_time:
                y_meas = np.array([0.0, 0.0])
                y_ins_hat = np.array([robot.v_nominal[k, 0], robot.v_nominal[k, 1]])
                R = np.diag([sigma_vel**2, sigma_vel**2])
                robot.eskf.update_velocity_calibration(y_meas, y_ins_hat, R)
                updated = True

            if updated:
                robot.apply_filter_correction(k)


   

    # Range updates
    beacon_due = (
        beacon_range_interval_steps is not None
        and (k % beacon_range_interval_steps == 0)
        and (range_measurement_stop_time is None or t[k] <= range_measurement_stop_time)
    )
    robot_range_due = (
        robot_range_interval_steps is not None
        and (k % robot_range_interval_steps == 0)
        and (range_measurement_stop_time is None or t[k] <= range_measurement_stop_time)
    )

    # Robot-to-beacon range updates
    if beacon_due:
        for idx, robot in enumerate(robots):
            beacon = beacons_all[current_beacon_index]                                          # 3D beacon position
            initiator_true = np.array([robot.pos_true[k, 0], robot.pos_true[k, 1], 0.0])        # 3D robot position (z=0)
            initiator_nominal = np.array([robot.p_nominal[k, 0], robot.p_nominal[k, 1], 0.0])   # 3D nominal robot position
            beacon_true, beacon_nominal = beacon, beacon 

            diff_true = initiator_true - beacon_true                                         # True 3D vector for range measurement calculation            
            y_meas = np.linalg.norm(diff_true) + range_rngs[idx].normal(scale=sigma_range)            # Simulated range measurement with noise
            y_meas = max(y_meas, 0.0)                                                           # Ensure non-negative range measurement         
            
            robot.eskf.update_beacon_range(
                z=y_meas,
                initiator_nom=initiator_nominal,
                reflector_nom=beacon_nominal,
                R=sigma_range**2,
            )
            robot.apply_filter_correction(k)
        current_beacon_index = (current_beacon_index + 1) % len(beacons_all)

    
    if robot_range_due and num_robots == 2:

        if robot0_is_next_initiator:
            initiator_robot = robots[0]
            reflector_robot = robots[1]
            robot0_is_next_initiator = False
        else:
            initiator_robot = robots[1]
            reflector_robot = robots[0]
            robot0_is_next_initiator = True
        
        reflector_packet = reflector_robot.get_coop_packet(k)

        initiator_true = np.array([initiator_robot.pos_true[k, 0], initiator_robot.pos_true[k, 1], 0.0])
        reflector_true = np.array([reflector_robot.pos_true[k, 0], reflector_robot.pos_true[k, 1], 0.0])
        diff_true = initiator_true - reflector_true
        y_meas = np.linalg.norm(diff_true) + range_rngs[0].normal(scale=sigma_range)
        y_meas = max(y_meas, 0.0)

        msg_to_reflector = initiator_robot.inflated_covariance_range_update_as_initiator(
            k, y_meas, sigma_range**2, reflector_packet, coop_type, force_uncorrelated_robot_range
        )
        reflector_robot.apply_coop_update_from_initiator(k, msg_to_reflector)

    if plot_pos_live and (k % plot_pos_live_every_n_steps == 0 or k == N - 1):
        for robot in robots:
            ins_plot.update_live_position_plot(k, robot.pos_true, robot.p_nominal)


 


# Plotting
if plot_acc:
    for idx, robot in enumerate(robots):
        ins_plot.plot_acceleration(robot.t, robot.acc_true, robot.f_imu, robot_id=idx)

if plot_vel:
    for idx, robot in enumerate(robots):
        ins_plot.plot_velocity(robot.t, robot.vel_true, robot.v_nominal, robot_id=idx)

if plot_pos:
    if trajectory_mode in {"parallel_x", "parallel_y"} and num_robots == 2:
        ins_plot.plot_positions_combined(
            robots,
            beacons=beacons_all if beacon_ranging else None,
            title=f"{trajectory_mode}: True vs Estimated",
        )
        for idx, robot in enumerate(robots):
            ins_plot.plot_positions(
                robot.t,
                robot.pos_true,
                robot.p_nominal,
                trajectory_mode,
                beacons=beacons_all if beacon_ranging else None,
                standstill_time=standstill_time,
                robot_id=idx,
                total_robots=num_robots,
                show_trajectory_plot=False,
            )
    else:
        for idx, robot in enumerate(robots):
            ins_plot.plot_positions(
                robot.t,
                robot.pos_true,
                robot.p_nominal,
                trajectory_mode,
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
