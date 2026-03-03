import numpy as np


class ESKFSingleRobot:
    """
    Decentralized 2D error-state Kalman filter for a single robot.
    State: [δp_x, δp_y, δv_x, δv_y, δb_x, δb_y]^T
    """

    def __init__(self, dt, sigma_acc, sigma_bias):
        self.dt = dt
        self.state_dim = 6
        self.noise_dim = 4

        self.T_acc = np.inf
        decay = 0.0 if np.isinf(self.T_acc) else -1.0 / self.T_acc

        q_acc = sigma_acc ** 2
        q_b = sigma_bias ** 2

        self.Ac = np.zeros((self.state_dim, self.state_dim))
        self.Ec = np.zeros((self.state_dim, self.noise_dim))

        self.Ac[0, 2] = 1.0
        self.Ac[1, 3] = 1.0
        self.Ac[2, 4] = -1.0
        self.Ac[3, 5] = -1.0
        self.Ac[4, 4] = decay
        self.Ac[5, 5] = decay

        self.Ec[2, 0] = -1.0
        self.Ec[3, 1] = -1.0
        self.Ec[4, 2] = 1.0
        self.Ec[5, 3] = 1.0

        self.Ad = np.eye(self.state_dim) + self.Ac * self.dt
        self.Ed = self.Ec.copy() * self.dt

        self.Qd = np.diag([q_acc, q_acc, q_b, q_b])

        self.deltax = np.zeros((self.state_dim, 1))
        self.P = np.eye(self.state_dim) * 1e-3

    def predict(self, acc_meas_2d=None, dt=None):
        # Keep signature flexible; current linear model uses fixed dt from init.
        self.deltax = self.Ad @ self.deltax
        self.P = self.Ad @ self.P @ self.Ad.T + self.Ed @ self.Qd @ self.Ed.T

    def _update_linear(self, C, delta_y, R):
        r = delta_y - C @ self.deltax
        S = C @ self.P @ C.T + R
        K = self.P @ C.T @ np.linalg.inv(S)
        self.deltax = K @ r
        I = np.eye(self.state_dim)
        self.P = (I - K @ C) @ self.P @ (I - K @ C).T + K @ R @ K.T

    def update_velocity_x(self, z, R):
        C = np.zeros((1, self.state_dim))
        C[0, 2] = 1.0
        delta_y = np.array([[float(z)]])
        Rm = np.array([[float(R)]]) if np.isscalar(R) else np.asarray(R, dtype=float).reshape(1, 1)
        self._update_linear(C, delta_y, Rm)

    def update_velocity_y(self, z, R):
        C = np.zeros((1, self.state_dim))
        C[0, 3] = 1.0
        delta_y = np.array([[float(z)]])
        Rm = np.array([[float(R)]]) if np.isscalar(R) else np.asarray(R, dtype=float).reshape(1, 1)
        self._update_linear(C, delta_y, Rm)

    def update_velocity_calibration(self, z2, R2x2):
        C = np.zeros((2, self.state_dim))
        C[0, 2] = 1.0
        C[1, 3] = 1.0
        delta_y = np.asarray(z2, dtype=float).reshape(2, 1)
        Rm = np.asarray(R2x2, dtype=float).reshape(2, 2)
        self._update_linear(C, delta_y, Rm)

    def update_beacon_range(self, z_range, beacon_pos_2d, nominal_pos_2d, R):
        beacon_pos_2d = np.asarray(beacon_pos_2d, dtype=float).reshape(2)
        nominal_pos_2d = np.asarray(nominal_pos_2d, dtype=float).reshape(2)
        diff = nominal_pos_2d - beacon_pos_2d
        dist = max(np.linalg.norm(diff), 1e-9)

        C = np.zeros((1, self.state_dim))
        C[0, 0] = diff[0] / dist
        C[0, 1] = diff[1] / dist

        y_ins_hat = dist
        delta_y = np.array([[float(z_range) - y_ins_hat]])
        Rm = np.array([[float(R)]]) if np.isscalar(R) else np.asarray(R, dtype=float).reshape(1, 1)
        self._update_linear(C, delta_y, Rm)

    def get_state_correction(self):
        return self.deltax.copy()

    def reset_error_state(self):
        self.deltax = np.zeros_like(self.deltax)
