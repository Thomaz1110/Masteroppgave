import numpy as np
from scipy.linalg import expm

class ESKFSingleRobot:
    """
    Decentralized 2D error-state Kalman filter for a single robot.
    State: [δp_x, δp_y, δv_x, δv_y, δb_x, δb_y]^T
    """

    def __init__(self, dt, sigma_acc, sigma_bias):
        self.dt = dt
        self.state_dim = 6
        self.noise_dim = 4

        self.T_acc = 300 #np.inf
        decay = 0.0 if np.isinf(self.T_acc) else -1.0 / self.T_acc

        q_acc = sigma_acc**2 * dt                                   # Accelerometer measurement noise intensity. sigma_acc is per sample, so we use sigma_acc^2*dt for the continuous-time noise intensity to get the correct discrete-time noise variance after discretization.
                                                                    # Remove dt if using datasheet noise density.
        q_b = 2.0 * (sigma_bias ** 2) / self.T_acc                  # Bias driving noise intensity; chosen so that the resulting bias process has stationary variance sigma_bias^2. 

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

        # self.Ad = np.eye(self.state_dim) + self.Ac * self.dt
        # self.Ed = self.Ec * self.dt
        # self.Qd = self.Ed @ self.Qc @ self.Ed.T * self.dt


        n = self.state_dim

        G = self.Ec @ self.Qc @ self.Ec.T

        A_vl = np.block([
            [-self.Ac, G],
            [np.zeros((n, n)), self.Ac.T]
        ])

        M = expm(A_vl * self.dt)

        M12 = M[:n, n:]
        M22 = M[n:, n:]

        self.Ad = M22.T
        self.Qd = self.Ad @ M12

        self.deltax = np.zeros((self.state_dim, 1))
        self.P = np.eye(self.state_dim) * 1e-3

    def predict(self):
        # Keep signature flexible; current linear model uses fixed dt from init.
        self.deltax = self.Ad @ self.deltax
        self.P = self.Ad @ self.P @ self.Ad.T + self.Qd

    def update(self, H, delta_y, R):
        r = delta_y - H @ self.deltax
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.deltax = self.deltax +K @ r
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
    def ic_range_update(
        Pi, Pj, p_i, p_j, y_meas, R, Vi, Vj, ic_coop_type, omega_grid=None, force_uncorrelated=False
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
        if ic_coop_type == "mutualistic":
            c_i, c_j = 1.0, 1.0
        elif ic_coop_type == "commensalistic":
            c_i, c_j = 1.0, 0.0
        else:
            raise ValueError("ic_coop_type must be 'mutualistic' or 'commensalistic'")
        
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

        correlated = False if force_uncorrelated else (int(Vi) & int(Vj)) != 0

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

    @staticmethod
    def build_ci_pseudoposterior_from_range(Pi, Pj, p_i, p_j, y_meas, R):
        """
        Build a one-sided range-based pseudo-posterior for robot i.

        The pseudo-posterior is formed at robot j by marginalizing robot j's
        uncertainty into the effective measurement noise and then performing a
        local EKF-style update for robot i only.
        """
        Pi = np.asarray(Pi, dtype=float).reshape(6, 6)
        Pj = np.asarray(Pj, dtype=float).reshape(6, 6)
        p_i = np.asarray(p_i, dtype=float).reshape(2)
        p_j = np.asarray(p_j, dtype=float).reshape(2)
        R = np.array([[float(R)]]) if np.isscalar(R) else np.asarray(R, dtype=float).reshape(1, 1)

        diff_nom = p_i - p_j
        dist_nom = max(np.linalg.norm(diff_nom), 1e-9)
        residual = np.array([[float(y_meas - dist_nom)]])

        u = diff_nom / dist_nom
        Hi = np.zeros((1, 6))
        Hj = np.zeros((1, 6))
        Hi[0, 0:2] = u
        Hj[0, 0:2] = -u

        R_eff = Hj @ Pj @ Hj.T + R # Effective measurement noise for robot i after marginalizing robot j's uncertainty. This is a key step that accounts for the reflector's uncertainty in the initiator's update, enabling a consistent one-sided update without needing to share full state corrections or covariances. The initiator treats the reflector's uncertainty as part of the measurement noise, which allows it to perform an EKF-style update using only its own state and covariance.
        S_i = Hi @ Pi @ Hi.T + R_eff

        S_i = 0.5 * (S_i + S_i.T)   
        if float(S_i[0, 0]) <= 0.0:
            S_i[0, 0] = 1e-12

        K_i = Pi @ Hi.T @ np.linalg.inv(S_i)
        delta_i_from_j = (K_i @ residual).reshape(6, 1)

        I = np.eye(6)
        A = I - K_i @ Hi
        P_i_from_j = A @ Pi @ A.T + K_i @ R_eff @ K_i.T

        return delta_i_from_j, P_i_from_j.reshape(6, 6)

    @staticmethod
    def ci_fuse(P_prior, delta_pseudo, P_pseudo):
        """
        Fuse a zero-mean prior and a pseudo-posterior using covariance intersection.

        The CI weight is found by grid search on w in [1e-3, 1 - 1e-3],
        minimizing either logdet(P_fused) or trace(P_fused). The prior mean is
        assumed to be zero.
        """
        P_prior = np.asarray(P_prior, dtype=float).reshape(6, 6)
        delta_pseudo = np.asarray(delta_pseudo, dtype=float).reshape(6, 1)
        P_pseudo = np.asarray(P_pseudo, dtype=float).reshape(6, 6)

        ridge = 1e-12 * np.eye(6)
        try:
            P_prior_inv = np.linalg.inv(P_prior)
        except np.linalg.LinAlgError:
            P_prior_inv = np.linalg.inv(P_prior + ridge)

        try:
            P_pseudo_inv = np.linalg.inv(P_pseudo)
        except np.linalg.LinAlgError:
            P_pseudo_inv = np.linalg.inv(P_pseudo + ridge)

        def score_covariance(P):
            objective = "logdet"  # or "trace"
            P = 0.5 * (P + P.T)
            if objective == "logdet":
                try:
                    chol = np.linalg.cholesky(P)
                except np.linalg.LinAlgError:
                    return np.inf
                return 2.0 * np.sum(np.log(np.diag(chol)))
            if objective == "trace":
                return float(np.trace(P))
            raise ValueError("objective must be 'logdet' or 'trace'")

        eps = 1e-3
        w_grid = np.linspace(eps, 1.0 - eps, 199)
        best_w = 0.5
        best_score = np.inf

        for w in w_grid:
            P_candidate = np.linalg.inv(w * P_prior_inv + (1.0 - w) * P_pseudo_inv)
            score = score_covariance(P_candidate)
            if score < best_score:
                best_score = score
                best_w = float(w)


        P_fused = np.linalg.inv(best_w * P_prior_inv + (1.0 - best_w) * P_pseudo_inv)
        delta_fused = P_fused @ ((1.0 - best_w) * P_pseudo_inv @ delta_pseudo)

        return delta_fused.reshape(6, 1), P_fused.reshape(6, 6)
