"""Target motion models for the WAN tracking extension.

The reference paper (Zhao et al., arXiv:2604.02381) places each agent's
patrol disk at a FIXED centre c_i -- i.e. it implicitly assumes a static
target. Real surveillance targets move. We add the three standard target
classes used throughout the tracking literature, each a recognised motion
model so the extension is defensible:

  STATIC            target stays put            -> reproduces the paper
  DYNAMIC (NCV)     nearly-constant-velocity    -> the workhorse tracking
                    x_{k+1} = x_k + T v_k + noise   model (e.g. Li & Jilkov,
                    v_{k+1} = v_k + noise           "Survey of maneuvering
                                                    target tracking")
  TIME-VARYING      maneuvering: the target switches between motion modes
                    (constant velocity / constant turn / accelerate / stop)
                    over the mission -- the coordinated-turn + mode-switch
                    family used for maneuvering targets.

Each agent's patrol centre c_i becomes the (estimated) target position at
the current round, so the existing coverage constraint, mobility energy,
and overlap rho all keep working unchanged -- the disks simply track the
targets. This is the minimal, principled hook: targets drive c_i(t),
everything downstream is the paper's own machinery.
"""

import numpy as np


# ----------------------------------------------------------------------
# discrete-time motion: state s = [x, y, vx, vy]
# ----------------------------------------------------------------------

def _F(T):
    """Nearly-constant-velocity state-transition matrix for step T."""
    return np.array([[1, 0, T, 0],
                     [0, 1, 0, T],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], float)


def _turn(T, omega):
    """Coordinated-turn transition for turn rate omega [rad/s]
    (the standard CT model; reduces to NCV as omega -> 0)."""
    if abs(omega) < 1e-6:
        return _F(T)
    s, c = np.sin(omega * T), np.cos(omega * T)
    return np.array([[1, 0, s / omega,         -(1 - c) / omega],
                     [0, 1, (1 - c) / omega,    s / omega],
                     [0, 0, c,                  -s],
                     [0, 0, s,                  c]], float)


class Target:
    """One moving target with a chosen motion class."""

    def __init__(self, kind, state0, rng, *, area=500.0,
                 q_ncv=0.15, turn_rate=np.deg2rad(18), accel=2.0):
        self.kind = kind                 # 'static' | 'dynamic' | 'time_varying'
        self.s = np.asarray(state0, float)
        self.rng = rng
        self.area = area
        self.q = q_ncv                   # process-noise strength (m/s per sqrt step)
        self.turn_rate = turn_rate
        self.accel = accel
        self.k = 0
        self.history = [self.s[:2].copy()]
        if kind == "static":
            self.s[2:] = 0.0
        # a maneuvering target follows a schedule of modes across rounds
        self._modes = ["cv", "turn", "accel", "cv", "stop", "turn"]

    def pos(self):
        return self.s[:2].copy()

    def vel(self):
        return self.s[2:].copy()

    def step(self, T):
        """Advance one aggregation round of duration T seconds."""
        self.k += 1
        if self.kind == "static":
            pass                                    # no motion
        elif self.kind == "dynamic":
            # nearly-constant-velocity: deterministic drift + small jitter
            self.s = _F(T) @ self.s
            self.s[2:] += self.q * np.sqrt(T) * self.rng.standard_normal(2)
        elif self.kind == "time_varying":
            mode = self._modes[(self.k - 1) % len(self._modes)]
            if mode == "cv":
                self.s = _F(T) @ self.s
            elif mode == "turn":
                self.s = _turn(T, self.turn_rate) @ self.s
            elif mode == "accel":
                spd = np.linalg.norm(self.s[2:])
                if spd > 1e-6:
                    self.s[2:] *= (spd + self.accel * T) / spd
                self.s = _F(T) @ self.s
            elif mode == "stop":
                self.s[2:] = 0.0
                self.s = _F(T) @ self.s
            self.s[2:] += 0.5 * self.q * np.sqrt(T) * self.rng.standard_normal(2)
        # keep targets inside the arena (reflect at the walls)
        for d in (0, 1):
            if self.s[d] < 0:
                self.s[d] = -self.s[d]; self.s[2 + d] = abs(self.s[2 + d])
            elif self.s[d] > self.area:
                self.s[d] = 2 * self.area - self.s[d]; self.s[2 + d] = -abs(self.s[2 + d])
        self.history.append(self.s[:2].copy())
        return self.pos()


def spawn_targets(kind, n, rng, area=500.0, speed=(3.0, 7.0)):
    """Create n targets of one class with random starts and headings."""
    targets = []
    for _ in range(n):
        p = rng.uniform(0.12 * area, 0.88 * area, size=2)
        if kind == "static":
            v = np.zeros(2)
        else:
            ang = rng.uniform(0, 2 * np.pi)
            sp = rng.uniform(*speed)
            v = sp * np.array([np.cos(ang), np.sin(ang)])
        targets.append(Target(kind, np.r_[p, v], rng, area=area))
    return targets


# ----------------------------------------------------------------------
# estimation: agents don't know the true target position, they predict it
# ----------------------------------------------------------------------

class CVPredictor:
    """One-step constant-velocity predictor of a target's next position.

    Mirrors how a tracker would anticipate where to point next: a simple,
    well-understood baseline (a full Kalman filter is the obvious upgrade,
    but CV prediction is the standard first model and keeps the code lean).
    Prediction error grows with target agility, which is exactly what makes
    the time-varying case harder and the learned look-ahead valuable.
    """

    def __init__(self):
        self.prev = None

    def predict(self, meas, T):
        if self.prev is None:
            self.prev = meas.copy()
            return meas.copy()
        v = (meas - self.prev) / max(T, 1e-6)
        self.prev = meas.copy()
        return meas + v * T


def _Q(T, q):
    """Discrete white-noise-acceleration process covariance for the CV model."""
    G = np.array([[T * T / 2, 0], [0, T * T / 2], [T, 0], [0, T]])
    return G @ G.T * q


class KalmanCV:
    """Constant-velocity Kalman filter; smooths noisy measurements before
    predicting one step ahead. The standard upgrade over the finite-difference
    CV predictor: it rejects measurement noise instead of amplifying it."""

    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)

    def __init__(self, q=4.0, r=9.0):
        self.q, self.R = q, r * np.eye(2)
        self.x = None
        self.P = np.eye(4) * 100.0

    def predict(self, meas, T):
        if self.x is None:
            self.x = np.r_[meas, 0.0, 0.0]
            return meas.copy()
        F = _F(T)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + _Q(T, self.q)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (meas - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return (self.H @ (_F(T) @ self.x)).copy()


class IMMPredictor:
    """Interacting-multiple-model predictor over constant-velocity and
    constant-turn models -- the established best practice for maneuvering
    targets. It blends a straight-line and a turning hypothesis by their
    measurement likelihoods, so it stays accurate through turns where a
    single CV model lags."""

    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)

    def __init__(self, q=4.0, r=9.0, turn=np.deg2rad(18)):
        self.q, self.R = q, r * np.eye(2)
        self.turn = turn
        self.mu = np.array([0.5, 0.5])
        self.Pij = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.x = [None, None]
        self.P = [np.eye(4) * 100.0, np.eye(4) * 100.0]

    def _models(self, T):
        return [_F(T), _turn(T, self.turn)]

    def predict(self, meas, T):
        if self.x[0] is None:
            self.x = [np.r_[meas, 0., 0.], np.r_[meas, 0., 0.]]
            return meas.copy()
        cbar = self.Pij.T @ self.mu
        mix = (self.Pij * self.mu[:, None]) / cbar[None, :]
        x0 = [sum(mix[i, j] * self.x[i] for i in range(2)) for j in range(2)]
        P0 = [sum(mix[i, j] * (self.P[i] + np.outer(self.x[i] - x0[j],
              self.x[i] - x0[j])) for i in range(2)) for j in range(2)]
        Fs, like = self._models(T), np.zeros(2)
        for j in range(2):
            xp = Fs[j] @ x0[j]
            Pp = Fs[j] @ P0[j] @ Fs[j].T + _Q(T, self.q)
            S = self.H @ Pp @ self.H.T + self.R
            innov = meas - self.H @ xp
            K = Pp @ self.H.T @ np.linalg.inv(S)
            self.x[j] = xp + K @ innov
            self.P[j] = (np.eye(4) - K @ self.H) @ Pp
            d = float(innov @ np.linalg.inv(S) @ innov)
            like[j] = np.exp(-0.5 * d) / np.sqrt(np.linalg.det(2 * np.pi * S))
        self.mu = cbar * like
        self.mu = self.mu / max(self.mu.sum(), 1e-300)
        nxt = sum(self.mu[j] * (self.H @ (Fs[j] @ self.x[j])) for j in range(2))
        return nxt.copy()
