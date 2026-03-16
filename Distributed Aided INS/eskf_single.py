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
        q_b = (sigma_bias ** 2) / self.dt                           # explained down below

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

        self.Qc = np.diag([q_acc, q_acc, q_b, q_b])

        self.Ad = np.eye(self.state_dim) + self.Ac * self.dt
        self.Ed = self.Ec * self.dt
        self.Qd = self.Ed @ self.Qc @ self.Ed.T

        # We model accel bias as b[k+1] = b[k] + sigma_bias*sqrt(dt)*n, so Var(Δb) = sigma_bias^2 * dt.
        # Discretization uses Ed = Ec*dt and Qd = Ed @ Qc @ Ed.T, which would give Var(Δb) = dt^2 * q_b.
        # Therefore we set q_b = sigma_bias^2 / dt so that dt^2*q_b = sigma_bias^2*dt (correct per-step bias RW variance).

        self.deltax = np.zeros((self.state_dim, 1))
        self.P = np.eye(self.state_dim) * 1e-3

    def predict(self, acc_meas_2d=None, dt=None):
        # Keep signature flexible; current linear model uses fixed dt from init.
        self.deltax = self.Ad @ self.deltax
        self.P = self.Ad @ self.P @ self.Ad.T + self.Qd

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

    @staticmethod
    def coop_robot_range(
        Pi, Pj, p_i, p_j, y_meas, R, Vi, Vj, type, omega_grid=None
    ):
        """
        Mutualistic or commensalistic cooperative range update for two robots (i initiator, j reflector).

        Returns
        -------
        delta_i : (6,1) ndarray
        Pi_new : (6,6) ndarray
        delta_j : (6,1) ndarray
        Pj_new : (6,6) ndarray
        V_new : int
            Bitmask union of cooperative histories.
        omega_used : float | None
            Used covariance-intersection weight when correlated, otherwise None.
        """
        if type == "mutualistic":
            c_i, c_j = 1.0, 1.0
        elif type == "commensalistic":
            c_i, c_j = 1.0, 0.0
        else:
            raise ValueError("type must be 'mutualistic' or 'commensalistic'")
        
        Pi = np.asarray(Pi, dtype=float).reshape(6, 6)
        Pj = np.asarray(Pj, dtype=float).reshape(6, 6)
        p_i = np.asarray(p_i, dtype=float).reshape(2)
        p_j = np.asarray(p_j, dtype=float).reshape(2)
        R = np.array([[float(R)]]) if np.isscalar(R) else np.asarray(R, dtype=float).reshape(1, 1)

        if omega_grid is None:
            omega_grid = np.linspace(0.05, 0.95, 19)

        diff_nom = p_i - p_j
        dist_nom = max(np.linalg.norm(diff_nom), 1e-9)

        y_ins_hat = dist_nom
        delta_y = np.array([[float(y_meas - y_ins_hat)]])

        u = diff_nom / dist_nom
        Hi = np.zeros((1, 6))
        Hj = np.zeros((1, 6))
        Hi[0, 0:2] = u
        Hj[0, 0:2] = -u

        correlated = False #(int(Vi) & int(Vj)) != 0

        def logdet_spd(P):
            L = np.linalg.cholesky(P)
            return 2.0 * np.sum(np.log(np.diag(L)))

        def joint_update(omega=None):
            Pbar = np.zeros((12, 12))
            if omega is None:                           # No covariance intersection, use full covariances
                Pbar[:6, :6] = Pi                       
                Pbar[6:, 6:] = Pj
            else:                                       # Covariance intersection with weight omega for initiator, (1-omega) for reflector
                Pbar[:6, :6] = Pi / omega
                Pbar[6:, 6:] = Pj / (1.0 - omega)

            H = np.zeros((1, 12))                       
            H[0, :6] = Hi
            H[0, 6:] = Hj

            S = H @ Pbar @ H.T + R                      # Innovation covariance
            K = Pbar @ H.T @ np.linalg.inv(S)           # Kalman gain
            delta = K @ delta_y                         # State correction for combined state [delta_i; delta_j], with shape (12,1)

            I = np.eye(12)
            Pplus = (I - K @ H) @ Pbar @ (I - K @ H).T + K @ R @ K.T  # Updated covariance for combined state  

            di = delta[:6].reshape(6, 1)
            dj = delta[6:].reshape(6, 1)
            Pi_new = Pplus[:6, :6]
            Pj_new = Pplus[6:, 6:]
            return di, dj, Pi_new, Pj_new

        omega_used = None
        if not correlated:                                      # If not previously cooperated, no covariance intersection needed
            di, dj, Pi_new, Pj_new = joint_update(omega=None)
        
        else:                                                   # If previously cooperated, perform covariance intersection over specified omega grid to find optimal balance between initiator and reflector covariance reductions
            best_w = None                                       
            best_J = np.inf
            for w in omega_grid:
                w = float(w)
                try:
                    _, _, Pi_tmp, Pj_tmp = joint_update(omega=w)
                    J = c_i*logdet_spd(Pi_tmp) + c_j*logdet_spd(Pj_tmp)
                except np.linalg.LinAlgError:
                    continue
                if J < best_J:
                    best_J = J
                    best_w = w

            if best_w is None:
                best_w = 0.5

            #print(best_w)
            di, dj, Pi_new, Pj_new = joint_update(omega=best_w)

        V_new = int(Vi) | int(Vj)
        return di, Pi_new, dj, Pj_new, V_new
