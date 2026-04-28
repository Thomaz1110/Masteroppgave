import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import FormatStrFormatter

# ------------------ GLOBAL STYLE SETTINGS ------------------
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
ARROW_COUNT = 14
ARROW_LINEWIDTH = 1.5


def _style_ax(ax):
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_FONTSIZE)


_pos_comp_fig = None
_pos_comp_axes = None
_pos_comp_total = None
_bias_fig = None
_bias_axes = None
_bias_total = None
_live_pos_fig = None
_live_pos_ax = None
_live_true_line = None
_live_est_line = None
_live_true_marker = None
_live_est_marker = None


def start_initialization_progress(total_robots):
    sys.stdout.write("\n")
    sys.stdout.write("Robot Initialization\n")
    sys.stdout.write(f"Total robots: {total_robots}\n\n")
    sys.stdout.flush()


def print_initialization_progress(robot_idx, total_robots):
    fraction = 0.0 if total_robots <= 0 else min(max(robot_idx / total_robots, 0.0), 1.0)
    msg = (
        f"  {100.0 * fraction:6.2f}%"
        f"   Robot {robot_idx:>6}/{total_robots:<6}"
    )
    sys.stdout.write("\033[2K\r" + msg[:120])
    sys.stdout.flush()


def finish_initialization_progress():
    sys.stdout.write("\n\n")
    sys.stdout.flush()


def start_simulation_progress(total_steps, total_time_s):
    sys.stdout.write("\n")
    sys.stdout.write("Simulation Progress\n")
    sys.stdout.write(f"Total steps: {total_steps}  |  Simulated time: {total_time_s:.1f} s\n\n")
    sys.stdout.flush()


def print_simulation_progress(step_idx, total_steps, current_time_s, total_time_s):
    fraction = 0.0 if total_steps <= 0 else min(max(step_idx / total_steps, 0.0), 1.0)
    msg = (
        f"  {100.0 * fraction:6.2f}%"
        f"   Step {step_idx:>6}/{total_steps:<6}"
        f"   Time {current_time_s:>8.1f}/{total_time_s:<8.1f} s"
    )
    sys.stdout.write("\033[2K\r" + msg[:120])
    sys.stdout.flush()


def finish_simulation_progress():
    sys.stdout.write("\n\n")
    sys.stdout.flush()


def _set_bias_axis_scale(ax, true_min, true_max, step=0.005):
    true_span = true_max - true_min
    margin = max(0.001, 0.1 * true_span)
    lower = step * np.floor((true_min - margin) / step)
    upper = step * np.ceil((true_max + margin) / step)
    if np.isclose(lower, upper):
        lower -= step
        upper += step
    ax.set_ylim(lower, upper)
    ax.set_yticks(np.arange(lower, upper + 0.5 * step, step))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))


def _apply_ddg_ticks_and_grid(ax, pos_true, grid_x_limits=None, grid_y_limits=None):
    x_vals = pos_true[:, 0]
    y_vals = pos_true[:, 1]
    if grid_x_limits is None:
        x_grid_start = np.floor(x_vals.min() / 0.705) * 0.705
        x_grid_end = np.ceil(x_vals.max() / 0.705) * 0.705
    else:
        x_grid_start, x_grid_end = grid_x_limits
    if grid_y_limits is None:
        y_grid_start = np.floor(y_vals.min() / 0.505) * 0.505
        y_grid_end = np.ceil(y_vals.max() / 0.505) * 0.505
    else:
        y_grid_start, y_grid_end = grid_y_limits
    grid_xticks = np.arange(x_grid_start, x_grid_end + 0.705, 0.705)
    grid_yticks = np.arange(y_grid_start, y_grid_end + 0.505, 0.505)
    ax.set_xticks(grid_xticks)
    ax.set_yticks(grid_yticks)

    # label the tick at 0 as "0", keep all other grid tick labels blank
    xlabels = ["" for _ in grid_xticks]
    ylabels = ["" for _ in grid_yticks]
    x0 = np.where(np.isclose(grid_xticks, 0.0))[0]
    y0 = np.where(np.isclose(grid_yticks, 0.0))[0]
    if x0.size:
        xlabels[int(x0[0])] = "0"
    if y0.size:
        ylabels[int(y0[0])] = "0"
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    ax.grid(True)

    if grid_x_limits is None:
        x_major_end = min(35.0, max(5.0, np.ceil(x_vals.max() / 5.0) * 5.0))
    else:
        x_major_end = grid_x_limits[1]
    if grid_y_limits is None:
        y_major_end = min(20.0, max(5.0, np.ceil(y_vals.max() / 5.0) * 5.0))
    else:
        y_major_end = grid_y_limits[1]
    major_xticks = np.arange(0.0, x_major_end + 0.1, 5.0)
    major_yticks = np.arange(0.0, y_major_end + 0.1, 5.0)
    ax.set_xticks(major_xticks, minor=True)
    ax.set_yticks(major_yticks, minor=True)

    # UPDATED: match global tick fontsize
    ax.set_xticklabels([f"{x:.0f}" for x in major_xticks], minor=True, fontsize=TICK_FONTSIZE)
    ax.set_yticklabels([f"{y:.0f}" for y in major_yticks], minor=True, fontsize=TICK_FONTSIZE)


def _ensure_shared_position_axes(total_robots):
    global _pos_comp_fig, _pos_comp_axes, _pos_comp_total
    if (_pos_comp_fig is None or _pos_comp_axes is None or _pos_comp_total != total_robots):
        cols = 2 if total_robots > 1 else 1
        rows = int(np.ceil(total_robots / cols))
        _pos_comp_fig = plt.figure(figsize=(6 * cols, 4 * rows))
        _pos_comp_fig.suptitle("Position components", fontsize=TITLE_FONTSIZE)
        subfigs = _pos_comp_fig.subfigures(rows, cols, squeeze=False)
        axes_per_robot = []
        robot_idx = 0
        for r in range(rows):
            for c in range(cols):
                if robot_idx < total_robots:
                    subfig = subfigs[r][c]
                    subfig.suptitle(f"Robot {robot_idx}", fontsize=TITLE_FONTSIZE-4)
                    ax_row = subfig.subplots(3, 1, sharex=True)
                    for a in ax_row:
                        _style_ax(a)
                    axes_per_robot.append(ax_row)
                    robot_idx += 1
                else:
                    subfigs[r][c].set_visible(False)
        _pos_comp_axes = axes_per_robot
        _pos_comp_total = total_robots
    return _pos_comp_fig, _pos_comp_axes


def _ensure_shared_bias_axes(total_robots):
    global _bias_fig, _bias_axes, _bias_total
    if (_bias_fig is None or _bias_axes is None or _bias_total != total_robots):
        cols = 2 if total_robots > 1 else 1
        rows = int(np.ceil(total_robots / cols))
        _bias_fig = plt.figure(figsize=(6 * cols, 4 * rows))
        _bias_fig.suptitle("Bias estimates", fontsize=TITLE_FONTSIZE, y=0.98)
        _bias_fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.86, wspace=0.18, hspace=0.18)
        subfigs = _bias_fig.subfigures(rows, cols, squeeze=False)
        axes_per_robot = []
        robot_idx = 0
        for r in range(rows):
            for c in range(cols):
                if robot_idx < total_robots:
                    subfig = subfigs[r][c]
                    subfig.suptitle(f"Robot {robot_idx}", fontsize=TITLE_FONTSIZE-4)
                    ax_row = subfig.subplots(2, 1, sharex=True)
                    subfig.subplots_adjust(left=0.16, bottom=0.20, right=0.98, top=0.88, hspace=0.24)
                    for a in ax_row:
                        _style_ax(a)
                    axes_per_robot.append(ax_row)
                    robot_idx += 1
                else:
                    subfigs[r][c].set_visible(False)
        _bias_axes = axes_per_robot
        _bias_total = total_robots
    return _bias_fig, _bias_axes


def init_live_position_plot(
    pos_true,
    pos_est,
    beacons=None,
    robot_id=None,
    grid_x_limits=(0.0, 35.0),
    grid_y_limits=(0.0, 21.0),
):
    global _live_pos_fig, _live_pos_ax
    global _live_true_line, _live_est_line
    global _live_true_marker, _live_est_marker

    plt.ion()
    _live_pos_fig, _live_pos_ax = plt.subplots(figsize=(12, 7))
    _live_true_line, = _live_pos_ax.plot([], [], label="True trajectory", color="tab:blue")
    _live_est_line, = _live_pos_ax.plot([], [], label="Estimated trajectory", color="tab:orange", alpha=0.85)
    _live_true_marker, = _live_pos_ax.plot([], [], marker="o", color="tab:blue", linestyle="None")
    _live_est_marker, = _live_pos_ax.plot([], [], marker="o", color="tab:orange", linestyle="None")

    if beacons is not None and len(beacons) > 0:
        beacons_xy = beacons[:, :2]
        _live_pos_ax.scatter(
            beacons_xy[:, 0],
            beacons_xy[:, 1],
            marker="o",
            color="tab:green",
            label="Known beacons",
            s=80,
        )

    title = "Live True vs Estimated Trajectory"
    if robot_id is not None:
        title += f" (Robot {robot_id})"
    _live_pos_ax.set_title(title, fontsize=TITLE_FONTSIZE)
    _live_pos_ax.set_xlabel("X [m]", fontsize=LABEL_FONTSIZE)
    _live_pos_ax.set_ylabel("Y [m]", fontsize=LABEL_FONTSIZE)
    _live_pos_ax.set_aspect("equal", adjustable="box")
    _live_pos_ax.legend()
    _style_ax(_live_pos_ax)

    _apply_ddg_ticks_and_grid(
        _live_pos_ax,
        pos_true,
        grid_x_limits=grid_x_limits,
        grid_y_limits=grid_y_limits,
    )

    update_live_position_plot(0, pos_true, pos_est)


def update_live_position_plot(k, pos_true, pos_est, pause_s=0.001):
    if _live_pos_ax is None:
        return

    true_xy = pos_true[: k + 1]
    est_xy = pos_est[: k + 1]

    _live_true_line.set_data(true_xy[:, 0], true_xy[:, 1])
    _live_est_line.set_data(est_xy[:, 0], est_xy[:, 1])
    _live_true_marker.set_data([true_xy[-1, 0]], [true_xy[-1, 1]])
    _live_est_marker.set_data([est_xy[-1, 0]], [est_xy[-1, 1]])

    _live_pos_ax.figure.canvas.draw_idle()
    plt.pause(pause_s)


def plot_true_trajectory(pos_true, trajectory_pattern, title="True Trajectory"):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(pos_true[:, 0], pos_true[:, 1], label="True trajectory")
    if len(pos_true) > 0:
        ax.scatter(pos_true[0, 0], pos_true[0, 1], marker="o", color="tab:red", s=80, label="Start")
        num_arrows = ARROW_COUNT
        if len(pos_true) > 1:
            arrow_bases = np.linspace(0, len(pos_true) - 1, num_arrows + 2)[1:-1]
            arrow_indices = np.clip(arrow_bases.astype(int), 0, len(pos_true) - 2)
            used_indices = set()
            for idx in arrow_indices:
                if idx in used_indices:
                    idx = min(idx + 1, len(pos_true) - 2)
                start = pos_true[idx]
                end = pos_true[idx + 1]
                if np.allclose(start, end):
                    continue
                ax.annotate(
                    "",
                    xy=(end[0], end[1]),
                    xytext=(start[0], start[1]),
                    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=ARROW_LINEWIDTH, mutation_scale=15),
                )
                used_indices.add(idx)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Y [m]", fontsize=LABEL_FONTSIZE)
    ax.grid(True)
    ax.legend()
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    _style_ax(ax)

    if trajectory_pattern in {"trajectory1", "trajectory2", "trajectory3", "trajectory4"}:
        _apply_ddg_ticks_and_grid(ax, pos_true)


def plot_trajectory_with_grid(
    pos,
    title="Trajectory",
    grid_x_limits=(0.0, 35.0),
    grid_y_limits=(0.0, 21.0),
    cell_size_x=0.705,
    cell_size_y=0.505,
    margin=1.0,
):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(pos[:, 0], pos[:, 1], label="Trajectory")
    if len(pos) > 0:
        ax.scatter(pos[0, 0], pos[0, 1], marker="o", color="tab:red", s=80, label="Start")
        num_arrows = ARROW_COUNT
        if len(pos) > 1:
            arrow_bases = np.linspace(0, len(pos) - 1, num_arrows + 2)[1:-1]
            arrow_indices = np.clip(arrow_bases.astype(int), 0, len(pos) - 2)
            used_indices = set()
            for idx in arrow_indices:
                if idx in used_indices:
                    idx = min(idx + 1, len(pos) - 2)
                start = pos[idx]
                end = pos[idx + 1]
                if np.allclose(start, end):
                    continue
                ax.annotate(
                    "",
                    xy=(end[0], end[1]),
                    xytext=(start[0], start[1]),
                    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=ARROW_LINEWIDTH, mutation_scale=15),
                )
                used_indices.add(idx)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Y [m]", fontsize=LABEL_FONTSIZE)
    ax.grid(True)
    ax.legend()
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    _style_ax(ax)

    ax.set_xlim(grid_x_limits[0] - margin, grid_x_limits[1] + margin)
    ax.set_ylim(grid_y_limits[0] - margin, grid_y_limits[1] + margin)

    grid_xticks = np.arange(grid_x_limits[0], grid_x_limits[1] + cell_size_x, cell_size_x)
    grid_yticks = np.arange(grid_y_limits[0], grid_y_limits[1] + cell_size_y, cell_size_y)
    ax.set_xticks(grid_xticks)
    ax.set_yticks(grid_yticks)

    xlabels = ["" for _ in grid_xticks]
    ylabels = ["" for _ in grid_yticks]
    x0 = np.where(np.isclose(grid_xticks, 0.0))[0]
    y0 = np.where(np.isclose(grid_yticks, 0.0))[0]
    if x0.size:
        xlabels[int(x0[0])] = "0"
    if y0.size:
        ylabels[int(y0[0])] = "0"
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    major_xticks = np.arange(grid_x_limits[0], grid_x_limits[1] + 0.1, 5.0)
    major_yticks = np.arange(grid_y_limits[0], grid_y_limits[1] + 0.1, 5.0)
    ax.set_xticks(major_xticks, minor=True)
    ax.set_yticks(major_yticks, minor=True)
    ax.set_xticklabels([f"{x:.0f}" for x in major_xticks], minor=True, fontsize=TICK_FONTSIZE)
    ax.set_yticklabels([f"{y:.0f}" for y in major_yticks], minor=True, fontsize=TICK_FONTSIZE)


def plot_acceleration(t, acc_true, f_meas, robot_id=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].plot(t, acc_true[:, 0], label="True a_x")
    axes[0].plot(t, f_meas[:, 0], label="IMU a_x (measured)", alpha=0.7)
    axes[0].set_ylabel("Acceleration X [m/s²]", fontsize=LABEL_FONTSIZE)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, acc_true[:, 1], label="True a_y")
    axes[1].plot(t, f_meas[:, 1], label="IMU a_y (measured)", alpha=0.7)
    axes[1].set_ylabel("Acceleration Y [m/s²]", fontsize=LABEL_FONTSIZE)
    axes[1].set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE)
    axes[1].legend()
    axes[1].grid(True)

    for a in axes:
        _style_ax(a)

    if robot_id is not None:
        fig.suptitle(f"Robot {robot_id} acceleration", fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def plot_velocity(t, vel_true, vel_est, robot_id=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].plot(t, vel_true[:, 0], label="True velocity $v_x$")
    axes[0].plot(t, vel_est[:, 0], label="Nominal velocity $\\hat{v}_{ins,x}$ ", alpha=0.7)
    axes[0].set_ylabel("Velocity X [m/s]", fontsize=LABEL_FONTSIZE)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, vel_true[:, 1], label="True velocity $v_y$")
    axes[1].plot(t, vel_est[:, 1], label="Nominal velocty $\\hat{v}_{ins,y}$", alpha=0.7)
    axes[1].set_ylabel("Velocity Y [m/s]", fontsize=LABEL_FONTSIZE)
    axes[1].set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE)
    axes[1].legend()
    axes[1].grid(True)

    for a in axes:
        _style_ax(a)

    if robot_id is not None:
        fig.suptitle(f"True vs Estimated Velocity (Robot {robot_id})", fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def plot_positions(
    t,
    pos_true,
    pos_est,
    trajectory_pattern,
    beacons=None,
    standstill_time=0.0,
    initial_beacon_guess=None,
    unknown_beacon_true=None,
    robot_id=None,
    total_robots=None,
    show_trajectory_plot=True,
):
    pos_error = np.linalg.norm(pos_true - pos_est, axis=1)
    dt = t[1] - t[0] if len(t) > 1 else 0.0
    standstill_samples = int(np.round(standstill_time / dt)) if standstill_time > 0.0 and dt > 0.0 else 0
    standstill_samples = min(standstill_samples, len(pos_error))
    pos_error_for_mean = pos_error[standstill_samples:]
    if pos_error_for_mean.size == 0:
        mean_error = np.nan
        pos_error_mean = np.zeros_like(pos_error)
    else:
        mean_error = pos_error_for_mean.mean()
        pos_error_mean = np.full_like(pos_error, mean_error)

    if show_trajectory_plot:
        fig_traj, ax_traj = plt.subplots(figsize=(12, 7))
        ax_traj.plot(pos_true[:, 0], pos_true[:, 1], label="True trajectory")
        ax_traj.plot(pos_est[:, 0], pos_est[:, 1], label="Estimated trajectory", alpha=0.7)
        if len(pos_true) > 0:
            ax_traj.scatter(pos_true[0, 0], pos_true[0, 1], marker="o", color="tab:red", s=80, label="Start")
            num_arrows = ARROW_COUNT
            if len(pos_true) > 1:
                arrow_bases = np.linspace(0, len(pos_true) - 1, num_arrows + 2)[1:-1]
                arrow_indices = np.clip(arrow_bases.astype(int), 0, len(pos_true) - 2)
                used_indices = set()
                for idx in arrow_indices:
                    if idx in used_indices:
                        idx = min(idx + 1, len(pos_true) - 2)
                    start = pos_true[idx]
                    end = pos_true[idx + 1]
                    if np.allclose(start, end):
                        continue
                    ax_traj.annotate(
                        "",
                        xy=(end[0], end[1]),
                        xytext=(start[0], start[1]),
                        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=ARROW_LINEWIDTH, mutation_scale=15),
                    )
                    used_indices.add(idx)

        if beacons is not None and len(beacons) > 0:
            beacons_xy = beacons[:, :2]
            ax_traj.scatter(beacons_xy[:, 0], beacons_xy[:, 1], marker="o", color="tab:green", label="Known beacons", s=80)
        if unknown_beacon_true is not None and (robot_id is None or robot_id == 0):
            ax_traj.scatter(
                unknown_beacon_true[0],
                unknown_beacon_true[1],
                marker="o",
                color="tab:red",
                label="Unknown beacon (true)",
            )
        if initial_beacon_guess is not None:
            ax_traj.scatter(
                initial_beacon_guess[0],
                initial_beacon_guess[1],
                marker="x",
                color="tab:purple",
                s=80,
                label="Initial beacon guess",
            )

        ax_traj.set_aspect("equal", adjustable="box")
        ax_traj.set_xlabel("X [m]", fontsize=LABEL_FONTSIZE)
        ax_traj.set_ylabel("Y [m]", fontsize=LABEL_FONTSIZE)
        ax_traj.legend()
        ax_traj.grid(True)

        title = "True vs Estimated Trajectory"
        if robot_id is not None and (total_robots is None or total_robots > 1):
            title += f" (Robot {robot_id})"
        ax_traj.set_title(title, fontsize=TITLE_FONTSIZE)
        _style_ax(ax_traj)

        _apply_ddg_ticks_and_grid(
            ax_traj,
            pos_true,
            grid_x_limits=(0.0, 35.0),
            grid_y_limits=(0.0, 21.0),
        )

    # Time series plots for components and error (shared figure for all robots if desired)
    standalone = False
    if total_robots is not None and robot_id is not None and total_robots > 1:
        _, axes_list = _ensure_shared_position_axes(total_robots)
        row_axes = axes_list[robot_id]
        for a in row_axes:
            a.clear()
    else:
        fig, row_axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle("Position components", fontsize=TITLE_FONTSIZE)
        standalone = True

    row_axes[0].plot(t, pos_true[:, 0], label="True x-position")
    row_axes[0].plot(t, pos_est[:, 0], label="Estimated x-position", alpha=0.7)
    row_axes[0].set_ylabel("X [m]", fontsize=LABEL_FONTSIZE-3)
    row_axes[0].legend()
    row_axes[0].grid(True)

    row_axes[1].plot(t, pos_true[:, 1], label="True y-position")
    row_axes[1].plot(t, pos_est[:, 1], label="Estimated y-position", alpha=0.7)
    row_axes[1].set_ylabel("Y [m]", fontsize=LABEL_FONTSIZE-3)
    row_axes[1].legend()
    row_axes[1].grid(True)

    row_axes[2].plot(t, pos_error, color="tab:orange", label="Instantaneous error")
    if standstill_samples < len(pos_error):
        row_axes[2].plot(
            t[standstill_samples:],
            pos_error_mean[standstill_samples:],
            color="tab:blue",
            linestyle="--",
            label="Mean error",
        )
    row_axes[2].set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE-3)
    row_axes[2].set_ylabel("Position error [m]", fontsize=LABEL_FONTSIZE-3)
    row_axes[2].legend()
    row_axes[2].grid(True)
    row_axes[2].text(t[-1], mean_error, f" {mean_error:.3f} m", color="tab:blue", va="bottom", ha="left")

    for a in row_axes:
        _style_ax(a)

    if standalone:
        fig.tight_layout(rect=[0, 0, 1, 0.96])


def plot_positions_combined(
    robots,
    beacons=None,
    title="True vs Estimated Trajectories",
):
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    all_true = []
    for idx, robot in enumerate(robots):
        color = colors[idx % len(colors)]
        pos_true = robot.pos_true
        pos_est = robot.p_nominal
        all_true.append(pos_true)

        ax.plot(
            pos_true[:, 0],
            pos_true[:, 1],
            color=color,
            label=f"Robot {idx} true",
        )
        ax.plot(
            pos_est[:, 0],
            pos_est[:, 1],
            color=color,
            linestyle="--",
            alpha=0.8,
            label=f"Robot {idx} estimated",
        )
        if len(pos_true) > 0:
            ax.scatter(
                pos_true[0, 0],
                pos_true[0, 1],
                marker="o",
                color=color,
                s=60,
            )

    if beacons is not None and len(beacons) > 0:
        beacons_xy = beacons[:, :2]
        ax.scatter(
            beacons_xy[:, 0],
            beacons_xy[:, 1],
            marker="o",
            color="tab:purple",
            label="Known beacons",
            s=80,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Y [m]", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.legend()
    ax.grid(True)
    _style_ax(ax)

    if all_true:
        all_true = np.vstack(all_true)
        x_margin = 1.0
        y_margin = 1.0
        ax.set_xlim(all_true[:, 0].min() - x_margin, all_true[:, 0].max() + x_margin)
        ax.set_ylim(all_true[:, 1].min() - y_margin, all_true[:, 1].max() + y_margin)


def plot_representative_position_errors(
    t,
    error_series,
    robot_ids,
    mean_errors,
    mean_of_mean_errors,
    aiding_label=None,
):
    fig, ax = plt.subplots(figsize=(11, 6.5))

    labels = ["Min mean", "Median mean", "Max mean"]
    colors = ["tab:green", "tab:blue", "tab:red"]

    for label, color, robot_id, err, mean_err in zip(labels, colors, robot_ids, error_series, mean_errors):
        ax.plot(
            t,
            err,
            color=color,
            linewidth=1.8,
            label=f"{label}: Robot {robot_id} (mean = {mean_err:.3f} m)",
        )

    ax.set_title("Representative Instantaneous Position Errors", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Position error [m]", fontsize=LABEL_FONTSIZE)
    ax.grid(True)
    ax.legend()
    _style_ax(ax)

    fig.subplots_adjust(bottom=0.24)
    fig.text(
        0.5,
        0.09,
        f"Mean of robot mean position errors: {mean_of_mean_errors:.3f} m",
        ha="center",
        va="center",
        fontsize=LABEL_FONTSIZE - 1,
    )
    if aiding_label is not None:
        fig.text(
            0.5,
            0.045,
            f"Aiding: {aiding_label}",
            ha="center",
            va="center",
            fontsize=LABEL_FONTSIZE - 1,
        )
def plot_bias(t, bias_true, b_ins, b_hat=None, robot_id=None, total_robots=None):
    standalone = False
    if total_robots is not None and robot_id is not None and total_robots > 1:
        _, axes_list = _ensure_shared_bias_axes(total_robots)
        axes = axes_list[robot_id]
        for a in axes:
            a.clear()
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        # UPDATED: larger title + closer to plots
        fig.suptitle("True vs Estimated Accelerometer Bias", fontsize=TITLE_FONTSIZE, y=0.94)
        standalone = True

    axes[0].plot(t, bias_true[:, 0], label="True bias $b_{acc,x}$")
    axes[0].plot(t, b_ins[:, 0], label="Nominal bias $\\hat{b}_{acc,ins,x}$", linestyle="--")
    if b_hat is not None:
        axes[0].plot(t, b_hat[:, 0], label="Estimated bias $\\hat{b}_{acc,x}$", linestyle="-")
    axes[0].set_ylabel("Bias X [m/s²]", fontsize=LABEL_FONTSIZE)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, bias_true[:, 1], label="True bias $b_{acc,y}$")
    axes[1].plot(t, b_ins[:, 1], label="Nominal bias $\\hat{b}_{acc,ins,y}$", linestyle="--")
    if b_hat is not None:
        axes[1].plot(t, b_hat[:, 1], label="Estimated bias $\\hat{b}_{acc,y}$", linestyle="-")
    axes[1].set_ylabel("Bias Y [m/s²]", fontsize=LABEL_FONTSIZE)
    axes[1].set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE)
    axes[1].legend()
    axes[1].grid(True)

    for axis_idx, ax in enumerate(axes):
        true_vals = bias_true[:, axis_idx]
        true_min = np.min(true_vals)
        true_max = np.max(true_vals)
        _set_bias_axis_scale(ax, true_min, true_max)

    for a in axes:
        _style_ax(a)

    if standalone:
        # UPDATED: keep space for the suptitle but not too much
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.subplots_adjust(left=0.14, bottom=0.14)
