import numpy as np


class ESKFMultiRobot:
    """
    Centralized 2D error-state Kalman filter for multiple robots.
    Each robot contributes the 6-element state
    [δp_x, δp_y, δv_x, δv_y, δb_x, δb_y]^T.

    where:
        δp : position error
        δv : velocity error
        δb_acc : accelerometer bias error
   
    
    Noise vector for each robot is 
        [ w_acc_x, w_acc_y, w_bacc_x, w_bacc_y ]^T
    where:
        w_acc : accelerometer measurement noise
        w_bacc : accelerometer bias driving noise = b_acc_dot
    """

    def __init__(self, dt, sigma_acc, sigma_bias, sigma_vel, sigma_range, num_robots):
        self.dt = dt
        self.sigma_vel = sigma_vel
        self.sigma_range = sigma_range
        self.num_robots = num_robots
        self.state_dim = 6 * num_robots
        self.noise_dim = 4 * num_robots
        
        self.T_acc = np.inf  # accelerometer bias time constant
        decay = 0.0 if np.isinf(self.T_acc) else -1.0 / self.T_acc


        q_acc = sigma_acc ** 2
        q_b = (sigma_bias ** 2) / self.dt

        self.Ac = np.zeros((self.state_dim, self.state_dim))
        self.Ec = np.zeros((self.state_dim, self.noise_dim))

        for robot in range(num_robots):
            state_offset = robot * 6
            noise_offset = robot * 4
            
            # Continuous-time system matrix
            self.Ac[state_offset + 0, state_offset + 2] = 1.0
            self.Ac[state_offset + 1, state_offset + 3] = 1.0
            self.Ac[state_offset + 2, state_offset + 4] = -1.0
            self.Ac[state_offset + 3, state_offset + 5] = -1.0
            self.Ac[state_offset + 4, state_offset + 4] = decay
            self.Ac[state_offset + 5, state_offset + 5] = decay

            # Continuous-time process noise matrix
            self.Ec[state_offset + 2, noise_offset + 0] = -1.0
            self.Ec[state_offset + 3, noise_offset + 1] = -1.0
            self.Ec[state_offset + 4, noise_offset + 2] = 1.0
            self.Ec[state_offset + 5, noise_offset + 3] = 1.0

        # Discrete-time system matrix and process noise matrix
        self.Ad = np.eye(self.state_dim) + self.Ac * self.dt
        self.Ed = self.Ec.copy() * self.dt

        # Discrete-time process noise covariance
        Qc_block = np.diag([q_acc, q_acc, q_b, q_b])
        self.Qc = np.kron(np.eye(num_robots), Qc_block)
        self.Qd = self.Ed @ self.Qc @ self.Ed.T

        self.deltax = np.zeros((self.state_dim, 1))
        self.P = np.eye(self.state_dim) * 1e-3

       

 
    def predict(self):
        self.deltax = self.Ad @ self.deltax # not really necessary since deltax is reset after each update, but included for completeness
        self.P = self.Ad @ self.P @ self.Ad.T + self.Qd 

    def update(self, robot_index, update_type, y_meas, y_ins_hat, initiator_nom=None, reflector_nom=None, reflector_index=None):
        state_offset = robot_index * 6                            # 6 states per robot

        if update_type == "velocity_x":
            C = np.zeros((1, self.state_dim))
            C[0, state_offset + 2] = 1.0
            R = np.array([[self.sigma_vel**2]])
            delta_y = (y_meas - y_ins_hat).reshape(1, 1)

        elif update_type == "velocity_y":
            C = np.zeros((1, self.state_dim))
            C[0, state_offset + 3] = 1.0
            R = np.array([[self.sigma_vel**2]])
            delta_y = (y_meas - y_ins_hat).reshape(1, 1)

        elif update_type == "velocity_calibration":
            C = np.zeros((2, self.state_dim))
            C[0, state_offset + 2] = 1.0
            C[1, state_offset + 3] = 1.0
            R = np.diag([self.sigma_vel**2, self.sigma_vel**2])
            delta_y = (y_meas - y_ins_hat).reshape(2, 1)

        elif update_type == "beacon_range" or update_type == "robot_range":
            if initiator_nom is None or reflector_nom is None:
                raise ValueError("Range update requires nominal_pos and beacon_pos")
            if update_type == "robot_range" and reflector_index is None:
                raise ValueError("Robot range update requires reflector_index")
            
            delta_y = np.array([[y_meas - y_ins_hat]])
            
            C = np.zeros((1, self.state_dim))
            diff = initiator_nom - reflector_nom
            dist = max(np.linalg.norm(diff), 1e-9)
            C[0, state_offset + 0] = diff[0] / dist
            C[0, state_offset + 1] = diff[1] / dist

            if update_type == "robot_range":
                reflector_offset = reflector_index * 6
                C[0, reflector_offset + 0] = -diff[0] / dist
                C[0, reflector_offset + 1] = -diff[1] / dist
            R = np.array([[self.sigma_range ** 2]])


        else:
            raise ValueError(f"Unsupported update type '{update_type}'")

        r = delta_y - C @ self.deltax
        S = C @ self.P @ C.T + R
        K = self.P @ C.T @ np.linalg.inv(S)
        self.deltax = K @ r
        I = np.eye(self.state_dim)
        self.P = (I - K @ C) @ self.P @ (I - K @ C).T + K @ R @ K.T

    def reset_error_state(self):
        self.deltax = np.zeros_like(self.deltax)

    def get_state_correction(self):
        return self.deltax.reshape(self.num_robots, 6)
