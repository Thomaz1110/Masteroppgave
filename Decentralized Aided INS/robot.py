"""
Robot model for decentralized aided INS simulation.

Contains:
- "Robot": generates true trajectory + IMU data, propagates nominal INS states,
  performs dominant-axis logic, and applies local ESKF corrections.
- "initialize_robot_positions": sets initial nominal positions (true or random offset).
"""

import numpy as np
import random_trajectory
import trajectory
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
        initial_bias_seed=None,
        trajectory_mode="random",
        parallel_track_spacing_lines=2,
    ):
        self.robot_id = robot_id
        self.dt = dt
        self.vel_threshold = vel_threshold
        if dominant_axis_method not in {"true", "nominal"}:
            raise ValueError(
                "dominant_axis_method must be 'true' or 'nominal'"
            )
        self.dominant_axis_method = dominant_axis_method

        if trajectory_mode == "random":
            self.t, self.pos_true, self.vel_true, self.acc_true = random_trajectory.random_trajectory_generator(
                dt=dt,
                duration_s=duration_s,
                standstill_time=standstill_time,
                seed=trajectory_seed,
            )
        elif trajectory_mode == "parallel_x":
            self.t, self.pos_true, self.vel_true, self.acc_true = trajectory.parallel_x_trajectory_generator(
                dt=dt,
                duration_s=duration_s,
                robot_id=robot_id,
                standstill_time=standstill_time,
                track_spacing_lines=parallel_track_spacing_lines,
            )
        elif trajectory_mode == "parallel_y":
            self.t, self.pos_true, self.vel_true, self.acc_true = trajectory.parallel_y_trajectory_generator(
                dt=dt,
                duration_s=duration_s,
                robot_id=robot_id,
                standstill_time=standstill_time,
                track_spacing_lines=parallel_track_spacing_lines,
            )
        else:
            raise ValueError("trajectory_mode must be 'random', 'parallel_x', or 'parallel_y'")
        self.N = len(self.t)

        self.f_imu, self.bias_true = imu_acc.sim_accelerometer(
            self.acc_true, dt, seed=imu_seed, initial_bias_seed=initial_bias_seed
        )

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






    # ----- INFLATED COVARIANCE -----
    
    def ic_range_update_as_initiator(self, k, y_meas, R, reflector_packet, ic_coop_type, force_uncorrelated=False):
        p_i = self.p_nominal[k].copy()
        Pi = self.eskf.P.copy()
        Vi = int(self.V_coop)

        p_j = np.asarray(reflector_packet["p_hat"], dtype=float).reshape(2)
        Pj = np.asarray(reflector_packet["P"], dtype=float).reshape(6, 6)
        Vj = int(reflector_packet["V"])

        di, Pi_new, dj, Pj_new, V_new = ESKFSingleRobot.ic_range_update(
            Pi=Pi,
            Pj=Pj,
            p_i=p_i,
            p_j=p_j,
            y_meas=y_meas,
            R=R,
            Vi=Vi,
            Vj=Vj,
            ic_coop_type = ic_coop_type,
            force_uncorrelated=force_uncorrelated,
        )


        # initiator applies its correction immediately; reflector will apply upon receiving msg
        self.eskf.deltax = di
        self.eskf.P = Pi_new
        self.apply_filter_correction(k)
        self.V_coop = int(V_new)

   
        msg_to_reflector = {
            "delta": dj,
            "P_new": Pj_new,
            "V_new": int(V_new)
        }

        return msg_to_reflector
    


    def request_ic_range_update(
        self,
        k,
        y_meas,
        R,
        reflector_robot,
        ic_coop_type,
        force_uncorrelated_robot_range,
    ):
        reflector_packet = reflector_robot.get_coop_packet(k)
        msg_to_reflector = self.ic_range_update_as_initiator(
            k,
            y_meas,
            R,
            reflector_packet,
            ic_coop_type,
            force_uncorrelated_robot_range,
        )

        # only apply cooperative update to reflector if mutualistic; in commensalistic case, initiator benefits but reflector does not update from this interaction
        if ic_coop_type == "mutualistic":
            reflector_robot.apply_ic_coop_update(k, msg_to_reflector)









    # ----- COVARIANCE INTERSECTION -----

    def apply_ic_coop_update(self, k, msg):
        self.eskf.deltax = np.asarray(msg["delta"], dtype=float).reshape(6, 1)
        self.eskf.P = np.asarray(msg["P_new"], dtype=float).reshape(6, 6)
        self.apply_filter_correction(k)
        self.V_coop = int(msg["V_new"])


    def build_ci_update_packet(self, k, y_meas, R, initiator_packet):

        p_j = self.p_nominal[k].copy()
        Pj = self.eskf.P.copy()

        p_i = np.asarray(initiator_packet["p_hat"], dtype=float).reshape(2)
        Pi = np.asarray(initiator_packet["P"], dtype=float).reshape(6, 6)

        delta_i_from_j, P_i_from_j = ESKFSingleRobot.build_ci_pseudoposterior_from_range(
            Pi=Pi,
            Pj=Pj,
            p_i=p_i,
            p_j=p_j,
            y_meas=y_meas,
            R=R,
        )

        return {
            "delta_pseudo": delta_i_from_j.reshape(6, 1),
            "P_pseudo": P_i_from_j.reshape(6, 6),
        }

    def apply_ci_update(self, k, msg):
       
        P_prior = self.eskf.P.copy()
        delta_pseudo = np.asarray(msg["delta_pseudo"], dtype=float).reshape(6, 1)
        P_pseudo = np.asarray(msg["P_pseudo"], dtype=float).reshape(6, 6)

        delta_fused, P_fused = ESKFSingleRobot.ci_fuse(
            P_prior=P_prior,
            delta_pseudo=delta_pseudo,
            P_pseudo=P_pseudo
        )

        self.eskf.deltax = delta_fused.reshape(6, 1)
        self.eskf.P = P_fused.reshape(6, 6)
        self.apply_filter_correction(k)

    def request_ci_range_update(
        self,
        k,
        y_meas,
        R,
        reflector_robot,
    ):
        initiator_packet = self.get_coop_packet(k)
        msg = reflector_robot.build_ci_update_packet(
            k=k,
            y_meas=y_meas,
            R=R,
            initiator_packet=initiator_packet,
        )
        self.apply_ci_update(k, msg)


   






def initialize_robot_positions(
    robots,
    use_true_initial_position,
    initial_pos_radius,
    initial_pos_var_robot,
    initial_bias_var,
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
    initial_pos_var_robot : float
    Initial position variance used when the initial position is uncertain.
    initial_bias_var : float
    Initial accelerometer bias variance.
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

        if use_true_initial_position:
            robot.eskf.P[0, 0] = 10e-3
            robot.eskf.P[1, 1] = 10e-3
        else:
            robot.eskf.P[0, 0] = initial_pos_var_robot
            robot.eskf.P[1, 1] = initial_pos_var_robot

        robot.eskf.P[4, 4] = initial_bias_var
        robot.eskf.P[5, 5] = initial_bias_var



def simulate_range_measurement(initiator, reflector, k, rng, sigma_range):
    
#hasattr checks if object has attribute "pos_true", which means it's a Robot instance. If not, it is a beacon position passed directly as a numpy array.
    
    if hasattr(initiator, "pos_true"):
        initiator_true = np.array(
            [initiator.pos_true[k, 0], initiator.pos_true[k, 1], 0.0],
            dtype=float,
        )
    else:
        initiator_true = np.asarray(initiator, dtype=float).reshape(3)

    if hasattr(reflector, "pos_true"):
        reflector_true = np.array(
            [reflector.pos_true[k, 0], reflector.pos_true[k, 1], 0.0],
            dtype=float,
        )
    else:
        reflector_true = np.asarray(reflector, dtype=float).reshape(3)

    diff_true = initiator_true - reflector_true
    y_meas = np.linalg.norm(diff_true) + rng.normal(scale=sigma_range)
    return max(float(y_meas), 0.0)
