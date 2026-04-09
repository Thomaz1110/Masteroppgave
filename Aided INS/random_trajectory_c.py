import numpy as np

from trajectory_c import trapezoid_segment

DEFAULT_DT = 0.01
DEFAULT_DURATION_S = 300.0
LIVE_ANIMATION = False


def random_trajectory_generator(
    dt=DEFAULT_DT,
    duration_s=DEFAULT_DURATION_S,
    grid_x_limits=(0.0, 35.0),
    grid_y_limits=(0.0, 21.0),
    cell_size_x=0.705,
    cell_size_y=0.505,
    standstill_time=0.0,
    short_segment_bias=0.3,
    vmax=3.1,
    amax=0.8,
    seed=None,
):
    """
    Generate a 2D random grid-constrained trajectory using trapezoidal segments.

    - Starts at a random grid intersection within the bounds
    - Moves only on the grid (axis-aligned): left/right/up/down
    - Chooses a direction uniformly among valid options
    - Chooses the segment length in cells with a bias toward shorter moves
      (see short_segment_bias), capped by the distance to the boundary
    - Cannot immediately reverse direction (no backtracking)
    - Stays within the grid limits
    - Optional initial standstill period for calibration/testing

    Returns
    -------
    t_total : ndarray
        Time vector [s].
    pos : ndarray
        Position [m] over time, shape (N, 2).
    vel : ndarray
        Velocity [m/s] over time, shape (N, 2).
    acc : ndarray
        Acceleration [m/s²] over time, shape (N, 2).
    """

    total_samples = int(np.round(duration_s / dt))
    if total_samples <= 0:
        return np.array([]), np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))

    rng = np.random.default_rng(seed)

    directions = [
        ("right", np.array([1.0, 0.0]), cell_size_x),
        ("left", np.array([-1.0, 0.0]), cell_size_x),
        ("up", np.array([0.0, 1.0]), cell_size_y),
        ("down", np.array([0.0, -1.0]), cell_size_y),
    ]
    back_dir = {"right": "left", "left": "right", "up": "down", "down": "up"}

    pos = []
    vel = []
    acc = []
    grid_x = np.arange(grid_x_limits[0], grid_x_limits[1] + cell_size_x, cell_size_x)
    grid_y = np.arange(grid_y_limits[0], grid_y_limits[1] + cell_size_y, cell_size_y)
    grid_offset = np.array([0.5 * cell_size_x, 0.5 * cell_size_y])
    start_x = grid_x[rng.integers(len(grid_x))]
    start_y = grid_y[rng.integers(len(grid_y))]
    current_pos = np.array([start_x, start_y]) + grid_offset
    last_dir = None
    standstill_samples = int(np.round(standstill_time / dt))
    if standstill_samples > 0:
        for _ in range(min(standstill_samples, total_samples)):
            pos.append(current_pos.copy())
            vel.append(np.zeros(2))
            acc.append(np.zeros(2))

    while len(pos) < total_samples:
        candidates = []
        for name, vec, cell_size in directions:
            if last_dir is not None and name == back_dir[last_dir]:
                continue
            if name == "right":
                max_cells = int(np.floor((grid_x_limits[1] - current_pos[0]) / cell_size))
            elif name == "left":
                max_cells = int(np.floor((current_pos[0] - grid_x_limits[0]) / cell_size))
            elif name == "up":
                max_cells = int(np.floor((grid_y_limits[1] - current_pos[1]) / cell_size))
            else:  # down
                max_cells = int(np.floor((current_pos[1] - grid_y_limits[0]) / cell_size))
            if max_cells >= 1:
                candidates.append((name, vec, cell_size, max_cells))

        if not candidates:
            # Fall back to any valid direction if all non-back choices are blocked.
            for name, vec, cell_size in directions:
                if name == "right":
                    max_cells = int(np.floor((grid_x_limits[1] - current_pos[0]) / cell_size))
                elif name == "left":
                    max_cells = int(np.floor((current_pos[0] - grid_x_limits[0]) / cell_size))
                elif name == "up":
                    max_cells = int(np.floor((grid_y_limits[1] - current_pos[1]) / cell_size))
                else:  # down
                    max_cells = int(np.floor((current_pos[1] - grid_y_limits[0]) / cell_size))
                if max_cells >= 1:
                    candidates.append((name, vec, cell_size, max_cells))
            if not candidates:
                break

        name, vec, cell_size, max_cells = candidates[rng.integers(len(candidates))]
        if short_segment_bias <= 0.0:
            num_cells = rng.integers(1, max_cells + 1)
        else:
            num_cells = int(np.ceil(rng.geometric(short_segment_bias)))
            num_cells = max(1, min(num_cells, max_cells))
        distance = num_cells * cell_size

        _, p_seg, v_seg, a_seg = trapezoid_segment(distance, vmax, amax, dt)
        if p_seg.size == 0:
            break
        if p_seg[-1] != 0.0:
            scale = distance / p_seg[-1]
            p_seg *= scale
            v_seg *= scale
            a_seg *= scale

        remaining = total_samples - len(pos)
        if len(p_seg) > remaining:
            p_seg = p_seg[:remaining]
            v_seg = v_seg[:remaining]
            a_seg = a_seg[:remaining]

        for k in range(len(p_seg)):
            pos.append(current_pos + vec * p_seg[k])
            vel.append(vec * v_seg[k])
            acc.append(vec * a_seg[k])

        if len(pos) == 0:
            break
        current_pos = pos[-1].copy()
        last_dir = name
    t_total = np.arange(len(pos)) * dt
    return t_total, np.array(pos), np.array(vel), np.array(acc)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    import ins_plot_c as ins_plot

    t, pos, vel, acc = random_trajectory_generator()
    ins_plot.plot_trajectory_with_grid(pos, title="Random Trajectory")
    ins_plot.plot_velocity(t, vel, vel, robot_id=None)
    plt.show()
