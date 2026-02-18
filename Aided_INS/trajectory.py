import numpy as np

def trapezoid_segment(distance, vmax, amax, dt):
    """
    Generate a 1D motion profile with trapezoidal velocity shape.
    Accelerate → cruise → decelerate.

    Parameters
    ----------
    distance : float
        Total travel distance [m].
    vmax : float
        Maximum velocity [m/s].
    amax : float
        Maximum acceleration [m/s²].
    dt : float
        Sampling interval [s].

    Returns
    -------
    t : ndarray
        Time vector [s].
    p : ndarray
        Position over time [m].
    v : ndarray
        Velocity over time [m/s].
    a : ndarray
        Acceleration over time [m/s²].
    """

    # Time and distance required for one acceleration phase
    t_acc = vmax / amax
    d_acc = 0.5 * amax * t_acc**2

    # If total distance too short to reach vmax → triangular profile
    if 2 * d_acc > distance:
        t_acc = np.sqrt(distance / amax)
        t_cruise = 0.0
        vmax = amax * t_acc
    else:
        d_cruise = distance - 2 * d_acc
        t_cruise = d_cruise / vmax

    # Total motion duration
    t_total = 2 * t_acc + t_cruise
    t = np.arange(0, t_total, dt)

    # Build acceleration timeline
    a = np.zeros_like(t)
    for k, tk in enumerate(t):
        if tk < t_acc:                     # accelerating
            a[k] = amax
        elif tk < t_acc + t_cruise:        # constant speed
            a[k] = 0.0
        else:                              # decelerating
            a[k] = -amax

    # Integrate to get velocity and position
    v = np.cumsum(a) * dt
    p = np.cumsum(v) * dt

    return t, p, v, a


def trajectory_generator(dt, path_points=None, segment_length=50, path_lengths=None,
                        vmax=3.1, amax=0.8, standstill_time=0.0, pattern="square"):
    """
    Generate a 2D trajectory following a set of straight path segments.

    Parameters
    ----------
    path_points : ndarray (N×2), optional
        Sequence of direction unit vectors or waypoints that define motion.
        If None, defaults to a square path: +x → +y → -x → -y.
    segment_length : float
        Length of each segment [m].
    vmax : float
        Maximum velocity [m/s].
    amax : float
        Maximum acceleration [m/s²].
    dt : float
        Time step [s].

    Returns
    -------
    t_total : ndarray
        Global time vector [s].
    pos : ndarray
        Position [m] over time, shape (N,2).
    vel : ndarray
        Velocity [m/s] over time, shape (N,2).
    acc : ndarray
        Acceleration [m/s²] over time, shape (N,2).
    """

    # Default path definitions
    current_pos_offset = np.zeros(2)
    if path_points is None:
        if pattern == "square":
            path_points = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
            path_lengths = [segment_length] * len(path_points)
        elif pattern == "trajectory1":
            cell_size_x = 0.705*10  # [m] DDG spec (wide direction)
            cell_size_y = 0.505*10  # [m] DDG spec (narrow direction)
            directions = [
                "right", "right", "up", "right", "up", "left", "up", "right",
                "right", "down", "down", "down", "right", "up", "up", "up", "up", "left", "left", "left",
                "left", "down", "left", "down", "right", "down", "left", "down",
            ]
            dir_map = {
                "right": (np.array([1, 0]), cell_size_x),
                "left": (np.array([-1, 0]), cell_size_x),
                "up": (np.array([0, 1]), cell_size_y),
                "down": (np.array([0, -1]), cell_size_y),
            }
            aggregated_points = []
            aggregated_lengths = []
            prev_vec, prev_len = None, 0.0
            for d in directions:
                vec, length = dir_map[d]
                if prev_vec is not None and np.all(vec == prev_vec):
                    prev_len += length
                else:
                    if prev_vec is not None:
                        aggregated_points.append(prev_vec)
                        aggregated_lengths.append(prev_len)
                    prev_vec = vec
                    prev_len = length
            if prev_vec is not None:
                aggregated_points.append(prev_vec)
                aggregated_lengths.append(prev_len)

            path_points = np.array(aggregated_points)
            path_lengths = aggregated_lengths
            current_pos_offset = np.zeros(2)
        elif pattern == "trajectory2":
            cell_size_x = 0.705*10
            cell_size_y = 0.505*10
            directions = [
                "left", "left", "up", "left", "up", "right", "up", "left",
                "left", "down", "down", "down", "left", "up", "up", "up", "up", "right", "right", "right",
                "right", "down", "right", "down", "left", "down", "right", "down",
            ]
            dir_map = {
                "right": (np.array([1, 0]), cell_size_x),
                "left": (np.array([-1, 0]), cell_size_x),
                "up": (np.array([0, 1]), cell_size_y),
                "down": (np.array([0, -1]), cell_size_y),
            }
            aggregated_points = []
            aggregated_lengths = []
            prev_vec, prev_len = None, 0.0
            for d in directions:
                vec, length = dir_map[d]
                if prev_vec is not None and np.all(vec == prev_vec):
                    prev_len += length
                else:
                    if prev_vec is not None:
                        aggregated_points.append(prev_vec)
                        aggregated_lengths.append(prev_len)
                    prev_vec = vec
                    prev_len = length
            if prev_vec is not None:
                aggregated_points.append(prev_vec)
                aggregated_lengths.append(prev_len)

            path_points = np.array(aggregated_points)
            path_lengths = aggregated_lengths
            current_pos_offset = np.array([35.25, 0.0])
        elif pattern == "trajectory3":
            cell_size_x = 0.705*10  # [m] DDG spec (wide direction)
            cell_size_y = 0.505*10  # [m] DDG spec (narrow direction)
            trajectory1_dirs = [
                "right", "right", "up", "right", "up", "left", "up", "right",
                "right", "down", "down", "down", "right", "up", "up", "up", "up", "left", "left", "left",
                "left", "down", "left", "down", "right", "down", "left", "down",
            ]
            directions = [
                ("down" if d == "up" else "up" if d == "down" else d)
                for d in trajectory1_dirs
            ]
            dir_map = {
                "right": (np.array([1, 0]), cell_size_x),
                "left": (np.array([-1, 0]), cell_size_x),
                "up": (np.array([0, 1]), cell_size_y),
                "down": (np.array([0, -1]), cell_size_y),
            }
            aggregated_points = []
            aggregated_lengths = []
            prev_vec, prev_len = None, 0.0
            for d in directions:
                vec, length = dir_map[d]
                if prev_vec is not None and np.all(vec == prev_vec):
                    prev_len += length
                else:
                    if prev_vec is not None:
                        aggregated_points.append(prev_vec)
                        aggregated_lengths.append(prev_len)
                    prev_vec = vec
                    prev_len = length
            if prev_vec is not None:
                aggregated_points.append(prev_vec)
                aggregated_lengths.append(prev_len)

            path_points = np.array(aggregated_points)
            path_lengths = aggregated_lengths
            current_pos_offset = np.array([0.0, 20.2])
        elif pattern == "trajectory4":
            cell_size_x = 0.705*10
            cell_size_y = 0.505*10
            trajectory2_dirs = [
                "left", "left", "up", "left", "up", "right", "up", "left",
                "left", "down", "down", "down", "left", "up", "up", "up", "up", "right", "right", "right",
                "right", "down", "right", "down", "left", "down", "right", "down",
            ]
            directions = [
                ("down" if d == "up" else "up" if d == "down" else d)
                for d in trajectory2_dirs
            ]
            dir_map = {
                "right": (np.array([1, 0]), cell_size_x),
                "left": (np.array([-1, 0]), cell_size_x),
                "up": (np.array([0, 1]), cell_size_y),
                "down": (np.array([0, -1]), cell_size_y),
            }
            aggregated_points = []
            aggregated_lengths = []
            prev_vec, prev_len = None, 0.0
            for d in directions:
                vec, length = dir_map[d]
                if prev_vec is not None and np.all(vec == prev_vec):
                    prev_len += length
                else:
                    if prev_vec is not None:
                        aggregated_points.append(prev_vec)
                        aggregated_lengths.append(prev_len)
                    prev_vec = vec
                    prev_len = length
            if prev_vec is not None:
                aggregated_points.append(prev_vec)
                aggregated_lengths.append(prev_len)

            path_points = np.array(aggregated_points)
            path_lengths = aggregated_lengths
            current_pos_offset = np.array([35.25, 20.2])
        else:
            raise ValueError(f"Unknown trajectory pattern '{pattern}'")
    else:
        path_points = np.asarray(path_points)
        if path_lengths is None:
            path_lengths = [segment_length] * len(path_points)

    if path_lengths is None:
        path_lengths = [segment_length] * len(path_points)

    pos, vel, acc = [], [], []
    current_pos = current_pos_offset.copy()

    # Optional initial standstill period for calibration/testing
    standstill_samples = int(np.round(standstill_time / dt))
    if standstill_samples > 0:
        for _ in range(standstill_samples):
            pos.append(current_pos.copy())
            vel.append(np.zeros(2))
            acc.append(np.zeros(2))

    # Generate one motion segment per direction
    for direction, seg_length in zip(path_points, path_lengths):
        _, p_seg, v_seg, a_seg = trapezoid_segment(seg_length, vmax, amax, dt)
        if p_seg[-1] != 0.0:
            scale = seg_length / p_seg[-1]
            p_seg *= scale
            v_seg *= scale
            a_seg *= scale
        for k in range(len(p_seg)):
            pos.append(current_pos + direction * p_seg[k])
            vel.append(direction * v_seg[k])
            acc.append(direction * a_seg[k])
        current_pos += direction * p_seg[-1]  # update end point

    pos, vel, acc = np.array(pos), np.array(vel), np.array(acc)
    if pattern in {"trajectory1", "trajectory2", "trajectory3", "trajectory4"}:
        shift = np.array([0.3525, 0.2525])
        pos = pos + shift
    t_total = np.arange(len(pos)) * dt

    return t_total, pos, vel, acc
