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

    def update(self, H, delta_y, R):
        r = delta_y - H @ self.deltax
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.deltax = K @ r
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    def update_dominant_axis_velocity(self, dominant_axis, y_meas, y_ins_hat, R):
        if dominant_axis == "x":
            vel_error_idx = 3               # y-velocity error index in state vector H
        elif dominant_axis == "y":
            vel_error_idx = 2               # x-velocity error index in state vector H
        else:
            raise ValueError("dominant_axis must be 'x' or 'y'")

        H = np.zeros((1, self.state_dim))
        H[0, vel_error_idx] = 1.0
        delta_y = np.array([[float(y_meas) - float(y_ins_hat)]])
        R = np.array([[float(R)]]) if np.isscalar(R) else np.asarray(R, dtype=float).reshape(1, 1)
        self.update(H, delta_y, R)

    def update_velocity_calibration(self, y_meas, y_ins_hat, R):
        H = np.zeros((2, self.state_dim))
        H[0, 2] = 1.0
        H[1, 3] = 1.0
        delta_y = np.asarray(y_meas, dtype=float).reshape(2, 1) - np.asarray(y_ins_hat, dtype=float).reshape(2, 1)
        R = np.asarray(R, dtype=float).reshape(2, 2)
        self.update(H, delta_y, R)

    def update_beacon_range(self, z, initiator_nom, reflector_nom, R):
        diff_nom = initiator_nom - reflector_nom
        dist_nom = max(np.linalg.norm(diff_nom), 1e-9)

        H = np.zeros((1, self.state_dim))
        H[0, 0] = diff_nom[0] / dist_nom
        H[0, 1] = diff_nom[1] / dist_nom

        y_ins_hat = dist_nom
        delta_y = np.array([[float(z) - y_ins_hat]])
        R = np.array([[float(R)]]) if np.isscalar(R) else np.asarray(R, dtype=float).reshape(1, 1)
        self.update(H, delta_y, R)

    def get_state_correction(self):
        return self.deltax.copy()

    def reset_error_state(self):
        self.deltax = np.zeros_like(self.deltax)
