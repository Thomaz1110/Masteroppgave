import numpy as np


def make_geometry_beacon_order(num_beacons):
    if num_beacons == 9:
        # 3x3 layout: cross first, then diagonally opposed corners, center last.
        return np.array([1, 5, 7, 3, 0, 8, 2, 6, 4], dtype=int)
    return np.arange(num_beacons, dtype=int)


def get_tdma_beacon_assignments(slot_index, num_robots, beacon_order):
    """
    Return (robot_idx, beacon_idx) pairs for one TDMA beacon-ranging slot.

    Each slot uses as many simultaneous ranges as possible while keeping both
    robot indices and beacon indices unique. Over a full cycle, every robot
    ranges to every beacon once before repeating a beacon.
    """
    if num_robots <= 0:
        return []

    beacon_order = np.asarray(beacon_order, dtype=int).reshape(-1)
    num_beacons = len(beacon_order)
    if num_beacons == 0:
        return []

    if len(np.unique(beacon_order)) != num_beacons:
        raise ValueError("beacon_order must contain unique beacon indices")

    if num_robots >= num_beacons:
        return [
            (int((slot_index - order_pos) % num_robots), int(beacon_idx))
            for order_pos, beacon_idx in enumerate(beacon_order)
        ]

    return [
        (int(robot_idx), int(beacon_order[(slot_index + robot_idx) % num_beacons]))
        for robot_idx in range(num_robots)
    ]
