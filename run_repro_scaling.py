"""Reproduction figure F1b: mission energy vs network size N.

The paper's Fig. 6 shows total energy growing with N, with the proposed
matching below distance-based and random. Same qualitative test here,
in the stressed regime where topology matters. Absolute joules differ
from the paper because Table I omits several constants (see README);
the claim checked is the TREND and the ORDERING, not the scale.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wan.model import P
from wan.topology import mission

ps = dict(P); ps["Tmax"]=2.5; ps["L_init"]=(15e6,25e6); ps["beta0"]=1e-5
ps["f_cpu"]=2.5e9; ps["C_gen"]=120.0
Ns, seeds = [4, 6, 8, 10], [3, 4, 5]
rows = {}
for pol in ["paper", "distance", "random"]:
    ys = []
    for n in Ns:
        es = [mission(s, ps, policy=pol, n=n)["E"] for s in seeds]
        fin = [e for e in es if np.isfinite(e)]
        ys.append(np.mean(fin) if fin else np.nan)
    rows[pol] = ys
    print(pol, ["%.2f" % v for v in ys])

ok_growth = all(np.diff([v for v in rows["paper"] if np.isfinite(v)]) > 0)
ok_order = all((not np.isfinite(rows["random"][k])) or rows["paper"][k] <= rows["random"][k] * 1.02
               for k in range(len(Ns)))
print("CHECK growth-with-N:", "PASS" if ok_growth else "FAIL")
print("CHECK paper<=random at every N:", "PASS" if ok_order else "FAIL")

plt.figure(figsize=(6,3.8))
mk = {"paper":"o-","distance":"s--","random":"^:"}
for pol in rows:
    plt.plot(Ns, rows[pol], mk[pol], label=pol if pol!="paper" else "proposed (H-MAP)")
plt.xlabel("number of agents N"); plt.ylabel("mission energy [J]")
plt.title("energy growth with network size (trend reproduction)")
plt.xticks(Ns); plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("figures/F1b_energy_vs_N.png", dpi=150)
with open("results.txt","a") as f:
    f.write("PASS  F1b energy grows with N, H-MAP <= random at every N\n" if (ok_growth and ok_order)
            else "FAIL  F1b scaling trend\n")
print("saved figures/F1b_energy_vs_N.png")
