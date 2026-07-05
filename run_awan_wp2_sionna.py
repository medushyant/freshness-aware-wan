"""WP-2 stretch — C5: rerun the channel headline under the Sionna RT
ray-traced Munich shadowing texture (assets/sionna_gain_grid.npy, exported on
Colab by notebooks/01_sionna_scene.ipynb). Figure F2.5.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from awan import adapters as A
from awan.channel.conformal_cal import certified_run, collect_scores, qhat
from awan.channel.sionna_map import GRID, SionnaMapChannel
from awan.harness import Checks
from awan.registry import record
from awan.runio import FIGS, Run

A.use_style()
C = Checks("WP-2-Sionna")
run = Run("wp2_sionna", {"grid": str(GRID)})

if not GRID.exists():
    C.check("C5", "Sionna ray-traced map present", False, "assets grid missing")
    raise SystemExit(1)

ch = SionnaMapChannel(A.STRESS, {})
# C1-analogue: paper plan under the ray-traced channel
v = t = 0
for seed in range(12):
    out_ = __import__("awan.simcore", fromlist=["run_episode"]).run_episode(
        seed, p=A.STRESS, channel=ch)
    v += out_["violations"]; t += out_["executed"]
viol = v / t
run.log(f"paper plan under sionna_map: {viol*100:.0f}% deadline violations")

# C2-analogue: dB-margin conformal, calibrated ON the ray-traced channel
alpha = 0.2
scores = collect_scores(ch, A.STRESS, seeds=range(100, 116))
q = qhat(scores, alpha)
cr = certified_run(ch, A.STRESS, seeds=range(200, 216), margin_db=q)
run.log(f"alpha={alpha}: qhat={q:.2f} dB -> coverage {cr['coverage']:.3f}")

C.check("C5", "ray-traced (Sionna RT) ground truth confirms H5: paper plan unsafe, conformal certifies",
        viol >= 0.2 and cr["coverage"] >= 1 - alpha,
        f"violations {viol*100:.0f}%; conformal coverage {cr['coverage']:.2f} >= {1-alpha:.2f} "
        f"(q={q:.1f} dB, MEASURED geometry / MODELED radio constants)")

# F2.5: the ray-traced shadowing texture + violation comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ch.new_scenario(7)
g = 100
xs = np.linspace(0, 500, g)
F = np.array([[ch.shadow_db(np.array([x, y])) for x in xs] for y in xs])
im = ax1.imshow(F, origin="lower", extent=[0, 500, 0, 500], cmap="magma")
fig.colorbar(im, ax=ax1, shrink=.8, label="ray-traced shadowing residual [dB]")
ax1.set_title("F2.5a  Sionna RT Munich residual on the arena (seed-7 window)")
ax2.bar(["paper plan\n(sionna_map)", f"conformal α={alpha}\n(sionna_map)"],
        [viol * 100, (1 - cr["coverage"]) * 100], color=["#e63946", "#2a9d8f"])
ax2.axhline(alpha * 100, ls="--", lw=1, color="#555")
ax2.set_ylabel("deadline-violation rate [%]")
ax2.set_title("F2.5b  H5 replicated on ray-traced ground truth")
fig.tight_layout(); fp = FIGS / "F2_5_sionna.png"; fig.savefig(fp, dpi=150); plt.close(fig)
record("F2.5", "wp2_sionna", fp)

run.finish({"viol_rate": viol, "conformal": {"alpha": alpha, "qhat_db": q, **cr}})
C.flush()
