"""CLEAN, DETERMINISTIC mechanism: how target agility moves the optimal
compression ratio and the energy, with NO scenario noise.

Principle (Age-of-Information / remote estimation): the staleness error of
information about a target grows with its speed. We charge that as a freshness
distortion that consumes the fidelity budget Dmax BEFORE compression. The
Direction-1 derived per-hop eta-floor then says: with less budget left, you
may compress less (eta-floor rises), so you must send more bits -> more energy.

This sweeps a single representative aggregation pair (fixed payload, link,
deadline) over target speed and reports eta_floor and the minimum energy.
Because there is no random geometry, the ladder is exact and monotone --
the clean contrast to the inert, noise-dominated mobility channel.

Run:  python3 -m experiments.exp_value_mechanism_clean
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wan.model import P as P0, make_agents, chan_gain
from wan.network import _derived_eta_floor
from wan.solver import solve_pair
from wan.style import use_style; use_style()
os.makedirs("figures", exist_ok=True)

# stress regime (same one the D2 topology experiments use): a harsher channel
# and tighter deadline so that sending more bits actually costs energy. With
# the paper's lazy Table-I radio, communication is nearly free and NOTHING
# moves energy -- documented in the handoff. The mechanism is identical.
P = dict(P0); P.update({"Tmax": 2.5, "L_init": (15e6, 25e6), "beta0": 1e-5,
                        "f_cpu": 2.5e9, "C_gen": 120.0})

A_F = 0.06          # distortion per (m/s): freshness sensitivity
K_REM = 2           # hops left to the root
SPEEDS = np.linspace(0, 9, 19)    # static .. fast maneuvering target

# one fixed, representative pair (same every time -> deterministic)
rng = np.random.default_rng(4)
ag = make_agents(rng, P, n=2)
i, j = ag[0], ag[1]
i["pos"] = np.array([200., 200.]); i["c"] = i["pos"].copy(); i["R"] = 90.; i["L"] = 22e6
j["pos"] = np.array([200., 300.]); j["c"] = j["pos"].copy(); j["R"] = 90.; j["L"] = 22e6
cen = np.array([200., 250.])

print("=" * 64)
print("target speed -> freshness budget -> eta-floor -> energy")
print("=" * 64)
print(f"  {'speed[m/s]':>10s} | {'fresh.D':>8s} | {'eta_floor':>9s} | {'energy[J]':>9s}")
etas, Es = [], []
for v in SPEEDS:
    d0 = A_F * v
    # pre-load the receiver's accumulated distortion with the freshness offset
    jj = dict(j); jj["srcs"] = {1: [j["L"], d0]}
    floor = _derived_eta_floor(jj, K_REM, P)
    r = solve_pair(i, j, P, fidelity=True, lam=0.0, rho_pair=0.0,
                   eta_cap_j=1.0, eta_lo_j=floor, eta_lo_i=floor,
                   centroid=cen, allow_motion=True)
    if r is None:
        print(f"  {v:10.1f} | {d0:8.3f} | {floor:9.3f} |  infeasible")
        etas.append(np.nan); Es.append(np.nan); continue
    etas.append(r["eta_j"]); Es.append(r["E"])
    print(f"  {v:10.1f} | {d0:8.3f} | {floor:9.3f} | {r['E']:9.3f}")

etas, Es = np.array(etas), np.array(Es)
ok = np.isfinite(Es)
mono_eta = np.all(np.diff(etas[ok]) >= -1e-9)
mono_E = np.all(np.diff(Es[ok]) >= -1e-6)
print("-" * 64)
print(f"  eta-floor monotone increasing in speed?  {mono_eta}")
print(f"  energy monotone increasing in speed?      {mono_E}")
print(f"  energy span: {np.nanmin(Es):.3f} -> {np.nanmax(Es):.3f} J "
      f"({100*(np.nanmax(Es)/np.nanmin(Es)-1):.0f}% rise)")

fig, ax1 = plt.subplots(figsize=(6.6, 4.2))
ax1.plot(SPEEDS[ok], etas[ok], "-o", color="#1f5f8b", label="optimal eta (less compression)")
ax1.set_xlabel("target speed [m/s]"); ax1.set_ylabel("optimal compression ratio eta*", color="#1f5f8b")
ax1.tick_params(axis="y", labelcolor="#1f5f8b")
ax2 = ax1.twinx()
ax2.plot(SPEEDS[ok], Es[ok], "-s", color="#c1403d", label="mission-pair energy")
ax2.set_ylabel("pair energy [J]", color="#c1403d"); ax2.tick_params(axis="y", labelcolor="#c1403d")
ax1.set_title("faster target -> stale data -> compress less -> spend more (exact)")
fig.tight_layout(); fig.savefig("figures/W3_value_mechanism_clean.png", dpi=150)
print("\nwrote figures/W3_value_mechanism_clean.png")
