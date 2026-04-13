import numpy as np
import random_trajectory_c as random_trajectory
import imu_acc_c as imu_acc


class Robot:
    def __init__(
        self,
        robot_id,
        dt,
        sigma_acc,
        sigma_bias,
        vel_threshold,
        dominant_axis_method,
        standstill_time,
        duration_s,
        trajectory_seed,
        imu_seed=None,
        initial_bias_seed=None,
    ):
        self.robot_id = robot_id
        self.dt = dt
        self.vel_threshold = vel_threshold
        if dominant_axis_method not in {"true", "nominal"}:
            raise ValueError(
                "dominant_axis_method must be 'true' or 'nominal'"
            )
        self.dominant_axis_method = dominant_axis_method

        self.t, self.pos_true, self.vel_true, self.acc_true = random_trajectory.random_trajectory_generator(
            dt=dt,
            duration_s=duration_s,
            standstill_time=standstill_time,
            seed=trajectory_seed,
        )
        self.N = len(self.t)

        self.f_imu, self.bias_true = imu_acc.sim_accelerometer(
            self.acc_true, dt, seed=imu_seed, initial_bias_seed=initial_bias_seed
        )

        self.p_nominal = np.zeros_like(self.pos_true)
        self.v_nominal = np.zeros_like(self.vel_true)
        self.b_nominal = np.zeros_like(self.f_imu)

       

    def propagate_nominal(self, k):
        acc_sample = self.f_imu[k - 1]
        acc_ins = acc_sample - self.b_nominal[k - 1]
        self.v_nominal[k] = self.v_nominal[k - 1] + acc_ins * self.dt
        self.p_nominal[k] = self.p_nominal[k - 1] + self.v_nominal[k - 1] * self.dt + 0.5 * acc_ins * self.dt**2
        self.b_nominal[k] = self.b_nominal[k - 1]

    def determine_dominant_axis(self, k):
        if self.dominant_axis_method == "nominal":
            prev_v = self.v_nominal[k - 1]
            if abs(prev_v[0]) > self.vel_threshold and abs(prev_v[0]) > abs(prev_v[1]):
                return "x"
            if abs(prev_v[1]) > self.vel_threshold and abs(prev_v[1]) > abs(prev_v[0]):
                return "y"
            return None

        true_vel = self.vel_true[k]
        # In "true" mode, select the dominant non-zero axis robustly.
        # A tolerance avoids axis flips from tiny numerical leftovers at turns.
        eps = 1e-9
        vx_abs = abs(true_vel[0])
        vy_abs = abs(true_vel[1])
        if vx_abs <= eps and vy_abs <= eps:
            return None
        if vx_abs > vy_abs + eps:
            return "x"
        if vy_abs > vx_abs + eps:
            return "y"
        # Tie-breaker for near-equality: keep deterministic behavior.
        if vx_abs >= vy_abs:
            return "x"
        return None

    def apply_correction(self, k, delta):
        self.p_nominal[k] += delta[0:2]
        self.v_nominal[k] += delta[2:4]
        self.b_nominal[k] += delta[4:6]


    def get_position_measurement(self, k, rng, sigma_pos):
        return self.pos_true[k] + rng.normal(scale=sigma_pos, size=2)


def generate_robot_pair_groups(num):
    if num < 2:
        return []
    indices = list(range(num))
    dummy = -1
    if num % 2 == 1:
        indices.append(dummy)
    rounds = len(indices) - 1
    groups = []
    for _ in range(rounds):
        pairs = []
        for i in range(len(indices) // 2):
            a = indices[i]
            b = indices[-(i + 1)]
            if a != dummy and b != dummy:
                pairs.append((a, b))
        groups.append(pairs)
        indices = [indices[0]] + [indices[-1]] + indices[1:-1]
    return groups


def initialize_robot_positions(
    robots,
    use_true_initial_position,
    robot0_true_initial_position,
    initial_pos_radius,
    grid_x_limits=(0.0, 35.0),
    grid_y_limits=(0.0, 20.0),
):
    """
    Initialize the nominal positions for a list of robots.

    Parameters
    ----------
    robots : list[Robot]
    List of Robot instances.
    use_true_initial_position : bool
    If True, start each robot's nominal position at its true initial position.
    If False, place it randomly on a circle of radius `initial_pos_radius`
    around the true position, subject to grid constraints.
    initial_pos_radius : float
    Radius for the random initial offsets.
    grid_x_limits, grid_y_limits : tuple
    Allowed bounds for x and y coordinates.
    """
    for idx, robot in enumerate(robots):
        use_true = use_true_initial_position
        if idx == 0 and robot0_true_initial_position:
            use_true = True
        if use_true:
            robot.p_nominal[0] = robot.pos_true[0]
        else:
            candidate = robot.pos_true[0].copy()
            for _ in range(100):
                angle = np.random.uniform(0.0, 2.0 * np.pi)
                offset = initial_pos_radius * np.array([np.cos(angle), np.sin(angle)])
                candidate = robot.pos_true[0] + offset
                if (
                    grid_x_limits[0] <= candidate[0] <= grid_x_limits[1]
                    and grid_y_limits[0] <= candidate[1] <= grid_y_limits[1]
                ):
                    robot.p_nominal[0] = candidate
                    break
            else:
                candidate[0] = np.clip(candidate[0], grid_x_limits[0], grid_x_limits[1])
                candidate[1] = np.clip(candidate[1], grid_y_limits[0], grid_y_limits[1])
                robot.p_nominal[0] = candidate
