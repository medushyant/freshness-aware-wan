"""Is the coverage constraint actually enforced when the target jumps outside
a tight sensing disk? Construct one pair where the receiver's start position
is 30 m from a NEW centre with R=8 m, and inspect what solve_pair does.

If it returns a solution that leaves the agent ~put (mobility ~0) while the
coverage constraint ||p-c||<=R is violated, then the paper's solver silently
drops coverage when motion is hard -- so a moving target never gets 'chased',
which is exactly why small r_track did not raise energy.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from wan.model import P, make_agents, mob_energy
from wan.solver import solve_pair

rng = np.random.default_rng(0)
agents = make_agents(rng, P, n=2)
i, j = agents[0], agents[1]

# put both agents' start positions at a known spot, then move the receiver's
# patrol centre 30 m away with a tight 8 m sensing radius (a fast target jump)
j["pos"] = np.array([100.0, 100.0]); j["c"] = np.array([130.0, 100.0]); j["R"] = 8.0
i["pos"] = np.array([105.0, 100.0]); i["c"] = np.array([135.0, 100.0]); i["R"] = 8.0
start_gap_j = np.linalg.norm(j["pos"] - j["c"]) - j["R"]
print(f"receiver start is {np.linalg.norm(j['pos']-j['c']):.1f} m from new centre, "
      f"R={j['R']:.0f}  => must move at least {start_gap_j:.1f} m to satisfy coverage")

r = solve_pair(i, j, P, rho_pair=0.0, centroid=np.array([117.,100.]),
               allow_motion=True)
if r is None:
    print("solve_pair returned None (pair declared infeasible)")
else:
    end_gap_j = np.linalg.norm(np.asarray(r["pe_j"]) - j["c"]) - j["R"]
    moved_j = np.linalg.norm(np.asarray(r["pe_j"]) - j["pos"])
    print(f"solve_pair SUCCEEDED: total E = {r['E']:.3f} J")
    print(f"  receiver moved {moved_j:5.1f} m ; end is {end_gap_j:+.1f} m outside its disk")
    print(f"  (coverage satisfied?  {'YES' if end_gap_j <= 1e-3 else 'NO -- constraint silently violated'})")
    print(f"  mobility-only energy for that move ~ {mob_energy(moved_j, P['Tmax']):.3f} J")
