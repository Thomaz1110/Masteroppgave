"""
Robot model for decentralized aided INS simulation.

Contains:
- "Robot": generates true trajectory + IMU data, propagates nominal INS states,
  performs dominant-axis logic, and applies local ESKF corrections.
- "initialize_robot_positions": sets initial nominal positions (true or random offset).
"""

import numpy as np
import random_trajectory
import imu_acc
from eskf_single import ESKFSingleRobot


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

        self.f_imu, self.bias_true = imu_acc.sim_accelerometer(self.acc_true, dt, seed=imu_seed)

        self.p_nominal = np.zeros_like(self.pos_true)
        self.v_nominal = np.zeros_like(self.vel_true)
        self.b_nominal = np.zeros_like(self.f_imu)
        self.eskf = ESKFSingleRobot(dt=dt, sigma_acc=sigma_acc, sigma_bias=sigma_bias)
        self.V_coop = 1 << int(self.robot_id)  # cooperative correlation-tracking bitmask

       

    def propagate_nominal(self, k):
        acc_sample = self.f_imu[k - 1]
        acc_ins = acc_sample - self.b_nominal[k - 1]
        self.v_nominal[k] = self.v_nominal[k - 1] + acc_ins * self.dt
        self.p_nominal[k] = self.p_nominal[k - 1] + self.v_nominal[k - 1] * self.dt #+ 0.5 * acc_ins * self.dt**2, removed for consistency with centralized case
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

    def apply_filter_correction(self, k):
        delta = self.eskf.get_state_correction().reshape(6)
        self.p_nominal[k] += delta[0:2]
        self.v_nominal[k] += delta[2:4]
        self.b_nominal[k] += delta[4:6]
        self.eskf.reset_error_state()

    def get_position_measurement(self, k, rng, sigma_pos):
        return self.pos_true[k] + rng.normal(scale=sigma_pos, size=2)

    def get_coop_packet(self, k):
        return {
            "id": int(self.robot_id),
            "p_hat": self.p_nominal[k].copy(),
            "P": self.eskf.P.copy(),
            "V": int(self.V_coop),
        }

    def apply_coop_update_from_initiator(self, k, msg):
        self.eskf.deltax = np.asarray(msg["delta"], dtype=float).reshape(6, 1)
        self.eskf.P = np.asarray(msg["P_new"], dtype=float).reshape(6, 6)
        self.apply_filter_correction(k)
        self.V_coop = int(msg["V_new"])

    def do_robot_range_as_initiator(self, k, y_meas, R, reflector_packet, coop_type, force_uncorrelated=False):
        p_i = self.p_nominal[k].copy()
        Pi = self.eskf.P.copy()
        Vi = int(self.V_coop)

        p_j = np.asarray(reflector_packet["p_hat"], dtype=float).reshape(2)
        Pj = np.asarray(reflector_packet["P"], dtype=float).reshape(6, 6)
        Vj = int(reflector_packet["V"])

        di, Pi_new, dj, Pj_new, V_new = ESKFSingleRobot.coop_robot_range(
            Pi=Pi,
            Pj=Pj,
            p_i=p_i,
            p_j=p_j,
            y_meas=y_meas,
            R=R,
            Vi=Vi,
            Vj=Vj,
            type = coop_type,
            force_uncorrelated=force_uncorrelated,
        )

        self.eskf.deltax = di
        self.eskf.P = Pi_new
        self.apply_filter_correction(k)
        self.V_coop = int(V_new)

        if coop_type == "commensalistic":
            dj = np.zeros_like(dj)
            Pj_new = Pj.copy()
            V_new = int(Vj)
        
        msg_to_reflector = {
            "delta": dj,
            "P_new": Pj_new,
            "V_new": int(V_new)
        }

        return msg_to_reflector


def initialize_robot_positions(
    robots,
    use_true_initial_position,
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
    for robot in robots:
        if use_true_initial_position:
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
