"""Core system model for the WAN reproduction (Zhao et al., arXiv:2604.02381).

Everything here follows Section II of the paper. Parameters marked [Table I]
come straight from the paper; the rest are not published, so I picked values
that land the pairwise energies in the same ~0.3-0.5 J ballpark as their
Fig. 4 and keep all the qualitative trends of Fig. 5.
"""

import numpy as np

# ----------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------

P = {
    # --- from Table I of the paper ---
    "N": 6,                  # number of agents (paper uses 10; 6 keeps runs quick)
    "area": 500.0,           # square region side [m]
    "R_cov": (80.0, 100.0),  # patrol radius range [m]
    "L_init": (5e6, 10e6),   # initial payload [bits]
    "Tmax": 6.0,             # per-round latency deadline [s]
    "B": 1e6,                # bandwidth [Hz]
    "beta0": 1e-4,           # channel gain at 1 m  (-40 dB)
    "delta": 3.0,            # path loss exponent
    "Pmax": 1.0,             # max transmit power [W]  (30 dBm)
    "vmax": 5.0,             # max speed [m/s]
    "f_cpu": 1e9,            # compute capacity [FLOP/s]
    "tau": 1e-28,            # effective capacitance
    "zeta": 1e-14,           # potential field weight

    # --- not given in the paper, chosen for a sensible reproduction ---
    "N0B": 3.98e-15,         # noise power: -174 dBm/Hz * 1 MHz
    "P_static": 0.2,         # rover baseline power [W]
    "k1": 0.05,              # mobility, linear term [J/m]
    "k2": 0.01,              # mobility, quadratic term [J s/m^2]
    "C_base": 60.0,          # base processing [FLOP/bit]
    "C_gen": 350.0,          # generative processing [FLOP/bit]
    "gamma": 1.0,            # compression complexity factor
    "eta_min": 0.05,         # hardest compression allowed
    "eta_req": 0.60,         # receiver compression cap (paper Eq. 15f)
    "d_floor": 1.0,          # minimum link distance [m], avoids h -> inf

    # --- Direction-1 additions (fidelity model) ---
    "a_D": 1.0,              # distortion units per ln(1/eta) of compression
    "Dmax": 1.2,             # fidelity floor on the root report
    "omega_wz": 0.9,         # Wyner-Ziv efficiency of side information
}


def make_agents(rng, p=P, n=None):
    """Random scenario like Sec. V: centers, radii, payloads, start positions."""
    n = n or p["N"]
    agents = []
    for i in range(n):
        c = rng.uniform(0, p["area"], size=2)
        R = rng.uniform(*p["R_cov"])
        L = rng.uniform(*p["L_init"])
        # start somewhere inside the patrol disk
        ang = rng.uniform(0, 2 * np.pi)
        rad = R * np.sqrt(rng.uniform())
        pos = c + rad * np.array([np.cos(ang), np.sin(ang)])
        agents.append({
            "id": i, "c": c, "R": R, "pos": pos,
            "L": L,                      # current payload [bits]
            "disks": [(c.copy(), R)],    # accumulated coverage (for rho)
            "srcs": {i: [L, 0.0]},       # source id -> [orig bits, sum ln(1/eta)]
            "rho_pred": 1.0,             # correlation w.r.t. predecessor (round 1: 1)
        })
    return agents


# ----------------------------------------------------------------------
# geometry: Jaccard overlap of unions of disks (Monte Carlo)
# ----------------------------------------------------------------------

def jaccard_disks(disks_a, disks_b, rng, n_samp=4000):
    """Area Jaccard index of two unions of disks, estimated by sampling.
    Good enough for the proxy rho in Eq. (5); shapely would be exact but
    this keeps the dependency list short."""
    allc = disks_a + disks_b
    xs = np.array([c[0] for c, _ in allc]); ys = np.array([c[1] for c, _ in allc])
    rs = np.array([r for _, r in allc])
    lo = np.array([np.min(xs - rs), np.min(ys - rs)])
    hi = np.array([np.max(xs + rs), np.max(ys + rs)])
    pts = rng.uniform(lo, hi, size=(n_samp, 2))

    def inside(disks):
        hit = np.zeros(n_samp, dtype=bool)
        for c, r in disks:
            hit |= np.sum((pts - np.asarray(c)) ** 2, axis=1) <= r * r
        return hit

    ina, inb = inside(disks_a), inside(disks_b)
    union = np.count_nonzero(ina | inb)
    if union == 0:
        return 0.0
    return np.count_nonzero(ina & inb) / union


# ----------------------------------------------------------------------
# energy / time pieces  (Sec. II-C, II-D, II-E)
# ----------------------------------------------------------------------

def chan_gain(pi, pj, p=P):
    d = max(np.linalg.norm(np.asarray(pi) - np.asarray(pj)), p["d_floor"])
    return p["beta0"] * d ** (-p["delta"])


def rate(ptx, h, p=P):
    return p["B"] * np.log2(1.0 + ptx * h / p["N0B"])


def mob_energy(dist, t_m, p=P):
    """Eq. (4) rewritten with motion time t_m (so v = dist/t_m)."""
    if dist < 1e-9:
        return 0.0
    return p["P_static"] * t_m + p["k1"] * dist + p["k2"] * dist ** 2 / t_m


def comp_load(L, rho, eta, p=P):
    """Eq. (6): FLOPs to fuse + compress a payload of L bits down by eta."""
    return L * (p["C_base"] + p["C_gen"] * p["gamma"] * (1.0 - rho) * np.log(1.0 / eta))


def comp_energy_time(L, rho, eta, p=P):
    W = comp_load(L, rho, eta, p)
    return p["tau"] * p["f_cpu"] ** 2 * W, W / p["f_cpu"]


def comm_energy(eta_L, h, t1, p=P):
    """Eq. (18): energy to push eta*L bits through gain h in time t1."""
    if eta_L <= 0:
        return 0.0
    z = eta_L / (p["B"] * t1)
    return t1 * (p["N0B"] / h) * (2.0 ** z - 1.0)


def ptx_for(eta_L, h, t1, p=P):
    """Power that delivers eta*L bits in exactly t1 seconds (Eq. 23)."""
    z = eta_L / (p["B"] * t1)
    return (p["N0B"] / h) * (2.0 ** z - 1.0)
