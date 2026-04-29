import numpy as np
import matplotlib.pyplot as plt

from robot import Robot, initialize_robot_positions, simulate_range_measurement
import plotting
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
    robot_pairing_seed,
    grid_x_limits,
    grid_y_limits,
)


def get_random_robot_pairs(num_robots, pairing_rng):
    robot_ids = np.arange(num_robots, dtype=int)
    pairing_rng.shuffle(robot_ids)

    pairs = []
    for idx in range(0, num_robots - 1, 2):
        pairs.append((int(robot_ids[idx]), int(robot_ids[idx + 1])))

    return pairs


num_robots = 1
duration_s = 1000.0
standstill_time = 20.0                  # [s] initial standstill period for calibration
use_true_initial_position = True        # True => all robots start at true positions
trajectory_mode = "random"              # "random", "parallel_x", or "parallel_y"
parallel_track_spacing_lines = 10       # separation in grid lines between the two parallel tracks


vel_threshold = 0.5                     # [m/s] velocity threshold for dominant-axis detection
dominant_axis_method = "true"           # "true" (use ground-truth velocity) or "nominal" (use estimated velocity + threshold)
velocity_update_rate_hz = 10.0          # [Hz] dominant-axis zero-velocity update rate


use_virtual_measurements = True         # If True, use virtual measurements: dominant-axis velocity updates and initial standstill velocity updates
beacon_ranging = True                  # robot-to-beacon ranging
robot_ranging = False                    # robot-to-robot ranging

cooperative_range_method = "ic"         # "ic" (inflated covariance) or "ci" (covariance intersection) 
ic_coop_type = "mutualistic"            # "mutualistic" or "commensalistic" cooperative range updates for robot-to-robot ranging
force_uncorrelated_robot_range = True   # If True, ignore cooperative-history correlation and treat robot pairs as uncorrelated

beacon_range_rate_hz = 0.1              # [Hz] robot-to-beacon range rate
robot_range_rate_hz = 0.1               # [Hz] robot-to-robot range rate
range_measurement_stop_time = None      # seconds; None => entire run

plot_acc = 0
plot_vel = 0
plot_pos = 1
plot_bias = 1
plot_pos_live = False
plot_pos_live_every_n_steps = 20
show_progress_bar = True
use_individual_robot_plots = num_robots <= 2

if show_progress_bar:
    plotting.start_initialization_progress(num_robots)

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
    if show_progress_bar:
        plotting.print_initialization_progress(idx + 1, num_robots)

if show_progress_bar:
    plotting.finish_initialization_progress()

initialize_robot_positions(
    robots,
    use_true_initial_position,
    initial_pos_radius,
    initial_pos_var_robot,
    initial_bias_var,
    grid_x_limits=grid_x_limits,
    grid_y_limits=grid_y_limits,
)


t = robots[0].t
N = len(t)
if any(robot.N != N for robot in robots):
    raise ValueError("All robots must have trajectories of equal length")

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
robot_pairing_rng = np.random.default_rng(robot_pairing_seed)
current_beacon_index = 0

# Define beacons
beacon_height = 2.0  # [m]
beacons_all = np.array([
    [2.5, 2.5, beacon_height],      # Bottom left        
    [2.5, 10.0, beacon_height],     # Middle left
    [2.5, 18.5, beacon_height],     # Top left
    [17.5, 2.5, beacon_height],     # Middle bottom
    [17.5, 10.0, beacon_height],    # Middle
    [17.5, 18.5, beacon_height],    # Middle top
    [33.5, 2.5, beacon_height],     # Bottom right
    [33.5, 10.0, beacon_height],    # Middle right
    [33.5, 18.5, beacon_height],    # Top right
])


robot0_is_next_initiator = True  # Alternate which robot acts as initiator for robot-to-robot ranging

if plot_pos_live:
    for idx, robot in enumerate(robots):
        plotting.init_live_position_plot(
            robot.pos_true,
            robot.p_nominal,
            beacons=beacons_all if beacon_ranging else None,
            robot_id=idx,
            grid_x_limits=(0.0, 35.0),
            grid_y_limits=(0.0, 21.0),
        )
if show_progress_bar:
    plotting.start_simulation_progress(N - 1, t[-1])
    plotting.print_simulation_progress(0, N - 1, 0.0, t[-1])
















# ---------- SIMULATION LOOP ----------   

for k in range(1, N):
    if show_progress_bar:
        should_refresh_progress = (k == 1) or (k == N - 1) or (k % max(1, N // 250) == 0)
        if should_refresh_progress:
            plotting.print_simulation_progress(k, N - 1, t[k], t[-1])

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



    # ROBOT-TO-BEACON RANGE UPDATES
    if beacon_due:
        for idx, robot in enumerate(robots):
            beacon = beacons_all[current_beacon_index]                                          # 3D beacon position
            initiator_nominal = np.array([robot.p_nominal[k, 0], robot.p_nominal[k, 1], 0.0])   # 3D nominal robot position
            beacon_nominal = beacon

            y_meas = simulate_range_measurement(robot, beacon, k, range_rngs[idx], sigma_range)                                                           
            
            robot.eskf.update_beacon_range(
                z=y_meas,
                initiator_nom=initiator_nominal,
                reflector_nom=beacon_nominal,
                R=sigma_range**2,
            )
            robot.apply_filter_correction(k)
        current_beacon_index = (current_beacon_index + 1) % len(beacons_all)

    




    # ROBOT-TO-ROBOT RANGE UPDATES
    if robot_range_due and num_robots == 2:
        
        if robot0_is_next_initiator:
            initiator_robot = robots[0]
            reflector_robot = robots[1]
            robot0_is_next_initiator = False
        else:
            initiator_robot = robots[1]
            reflector_robot = robots[0]
            robot0_is_next_initiator = True

        # initiator_robot = robots[0]
        # reflector_robot = robots[1]
        
        y_meas = simulate_range_measurement(initiator_robot, reflector_robot, k, range_rngs[initiator_robot.robot_id], sigma_range)

        
        # Inflated covariance (IC) or covariance intersection (CI) cooperative update requested by initiator robot
        if cooperative_range_method == "ic":
            initiator_robot.request_ic_range_update(
                k,
                y_meas,
                sigma_range**2,
                reflector_robot,
                ic_coop_type,
                force_uncorrelated_robot_range
            )
        elif cooperative_range_method == "ci":
            initiator_robot.request_ci_range_update(
                k,
                y_meas,
                sigma_range**2,
                reflector_robot
            )
        else:
            raise ValueError("Invalid cooperative_range_method. Must be 'ic' or 'ci'.")


    elif robot_range_due and num_robots > 2:

        robot_pairs = get_random_robot_pairs(num_robots, robot_pairing_rng)

        for initiator_idx, reflector_idx in robot_pairs:
            initiator_robot = robots[initiator_idx]
            reflector_robot = robots[reflector_idx]

            y_meas = simulate_range_measurement(
                initiator_robot,
                reflector_robot,
                k,
                range_rngs[initiator_robot.robot_id],
                sigma_range,
            )

            if cooperative_range_method == "ic":
                initiator_robot.request_ic_range_update(
                    k,
                    y_meas,
                    sigma_range**2,
                    reflector_robot,
                    ic_coop_type,
                    force_uncorrelated_robot_range
                )
            elif cooperative_range_method == "ci":
                initiator_robot.request_ci_range_update(
                    k,
                    y_meas,
                    sigma_range**2,
                    reflector_robot
                )
            else:
                raise ValueError("Invalid cooperative_range_method. Must be 'ic' or 'ci'.")






    if plot_pos_live and (k % plot_pos_live_every_n_steps == 0 or k == N - 1):
        for robot in robots:
            plotting.update_live_position_plot(k, robot.pos_true, robot.p_nominal)

































# Plotting
if use_individual_robot_plots and plot_acc:
    for idx, robot in enumerate(robots):
        plotting.plot_acceleration(robot.t, robot.acc_true, robot.f_imu, robot_id=idx)

if use_individual_robot_plots and plot_vel:
    for idx, robot in enumerate(robots):
        plotting.plot_velocity(robot.t, robot.vel_true, robot.v_nominal, robot_id=idx)

if use_individual_robot_plots and plot_pos:
    if trajectory_mode in {"parallel_x", "parallel_y"} and num_robots == 2:
        plotting.plot_positions_combined(
            robots,
            beacons=beacons_all if beacon_ranging else None,
            title=f"{trajectory_mode}: True vs Estimated",
        )
        for idx, robot in enumerate(robots):
            plotting.plot_positions(
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
            plotting.plot_positions(
                robot.t,
                robot.pos_true,
                robot.p_nominal,
                trajectory_mode,
                beacons=beacons_all if beacon_ranging else None,
                standstill_time=standstill_time,
                robot_id=idx,
                total_robots=num_robots,
            )

if use_individual_robot_plots and plot_bias:
    for idx, robot in enumerate(robots):
        plotting.plot_bias(
            robot.t,
            robot.bias_true,
            robot.b_nominal,
            None,
            robot_id=idx,
            total_robots=num_robots,
        )




if (not use_individual_robot_plots) and plot_pos:
    robot_error_series = []
    robot_mean_errors = []

    for robot in robots:
        pos_error = np.linalg.norm(robot.pos_true - robot.p_nominal, axis=1)
        robot_error_series.append(pos_error)
        robot_mean_errors.append(np.mean(pos_error))

    robot_mean_errors = np.asarray(robot_mean_errors)
    sorted_indices = np.argsort(robot_mean_errors)

    min_idx = int(sorted_indices[0])
    median_idx = int(sorted_indices[len(sorted_indices) // 2])
    max_idx = int(sorted_indices[-1])

    representative_indices = [min_idx, median_idx, max_idx]

    aiding_parts = []
    if beacon_ranging:
        aiding_parts.append("beacon ranging")
    if robot_ranging:
        if cooperative_range_method == "ic":
            aiding_parts.append(f"robot ranging ({ic_coop_type} IC)")
        elif cooperative_range_method == "ci":
            aiding_parts.append("robot ranging (CI)")
        else:
            aiding_parts.append("robot ranging")
    if not aiding_parts:
        aiding_parts.append("no external aiding")

    plotting.plot_representative_position_errors(
        t,
        [robot_error_series[idx] for idx in representative_indices],
        representative_indices,
        [float(robot_mean_errors[idx]) for idx in representative_indices],
        float(np.mean(robot_mean_errors)),
        aiding_label=", ".join(aiding_parts),
    )

if show_progress_bar:
    plotting.finish_simulation_progress()

plt.show()
