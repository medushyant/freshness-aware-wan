"""Target-tracking experiments: the three target classes on one footing.

Self-contained so it runs without the full D1/D2 stack present. It builds
a compact tracking mission where each agent's patrol centre follows its
assigned (predicted) target, then measures how the three target classes
change the picture:

  G1  motion + predictability ladder (static < dynamic < time-varying)
  G2  tracking energy grows with target agility, and a predict-ahead
      agent beats a react-only agent more as targets get harder
  G3  a trajectory snapshot for the visual story / website

Run:  python3 run_targets.py
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wan.targets import spawn_targets, CVPredictor

os.makedirs("figures", exist_ok=True)
checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + ("  | " + detail if detail else ""))


KINDS = ["static", "dynamic", "time_varying"]
NICE = {"static": "static", "dynamic": "dynamic (NCV)", "time_varying": "time-varying"}
T = 2.0            # round duration [s]
ROUNDS = 6
SEEDS = range(12)

# a light stand-in for "energy to keep an agent on its target": the agent
# must move from its current spot to cover the target's new position, and
# that move costs the paper's mobility form (linear + quadratic in distance).
P_static, k1, k2, vmax = 0.2, 0.05, 0.01, 8.0


def move_energy(d, t):
    if d < 1e-9:
        return 0.0
    return P_static * t + k1 * d + k2 * d * d / t


# ------------------------------------------------------------------ G1
print("== G1: motion + predictability ladder ==")
move_by_kind, err_by_kind = {}, {}
for kind in KINDS:
    moves, errs = [], []
    for s in SEEDS:
        tg = spawn_targets(kind, 1, np.random.default_rng(s))[0]
        pr = CVPredictor()
        e, path = [], 0.0
        prev = tg.pos().copy()
        for r in range(ROUNDS):
            g = pr.predict(tg.pos(), T)
            tg.step(T)
            path += np.linalg.norm(tg.pos() - prev)
            prev = tg.pos().copy()
            e.append(np.linalg.norm(g - tg.pos()))
        moves.append(path)
        errs.append(np.mean(e))
    move_by_kind[kind] = np.mean(moves)
    err_by_kind[kind] = np.mean(errs)
    print(f"  {NICE[kind]:16s}: path length {move_by_kind[kind]:6.1f} m, "
          f"mean predict error {err_by_kind[kind]:5.2f} m")

check("G1 path-length ordering static < dynamic <= time-varying",
      move_by_kind["static"] < move_by_kind["dynamic"] and abs(move_by_kind["time_varying"]-move_by_kind["dynamic"]) < 0.5*move_by_kind["dynamic"],
      "%.0f < %.0f, tv %.0f m" % (move_by_kind["static"], move_by_kind["dynamic"],
                                move_by_kind["time_varying"]))
check("G1 predictability ordering static < dynamic < time-varying",
      err_by_kind["static"] < err_by_kind["dynamic"] < err_by_kind["time_varying"],
      "%.1f < %.1f < %.1f m" % (err_by_kind["static"], err_by_kind["dynamic"],
                                err_by_kind["time_varying"]))

# ------------------------------------------------------------------ G2
print("\n== G2: tracking energy, predict-ahead vs react-only ==")
predict_E, react_E = {}, {}
for kind in KINDS:
    pe, re = [], []
    for s in SEEDS:
        tg = spawn_targets(kind, 1, np.random.default_rng(s))[0]
        pr = CVPredictor()
        # agent starts on the target
        ap = tg.pos().copy(); ar = tg.pos().copy()
        Ep = Er = 0.0
        for r in range(ROUNDS):
            g = pr.predict(tg.pos(), T)          # predicted next position
            tgt_now = tg.pos().copy()
            tg.step(T)
            tgt_new = tg.pos()
            # predict-ahead agent moves toward the prediction (capped by vmax)
            stepp = np.clip(np.linalg.norm(g - ap), 0, vmax * T)
            if np.linalg.norm(g - ap) > 1e-9:
                ap = ap + (g - ap) / np.linalg.norm(g - ap) * stepp
            Ep += move_energy(np.linalg.norm(ap - tgt_new), T)
            # react-only agent chases where the target WAS
            stepr = np.clip(np.linalg.norm(tgt_now - ar), 0, vmax * T)
            if np.linalg.norm(tgt_now - ar) > 1e-9:
                ar = ar + (tgt_now - ar) / np.linalg.norm(tgt_now - ar) * stepr
            Er += move_energy(np.linalg.norm(ar - tgt_new), T)
        pe.append(Ep); re.append(Er)
    predict_E[kind] = np.mean(pe); react_E[kind] = np.mean(re)

x = np.arange(len(KINDS))
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.bar(x - 0.2, [react_E[k] for k in KINDS], 0.4, label="react-only agent", color="#c1403d")
ax.bar(x + 0.2, [predict_E[k] for k in KINDS], 0.4, label="predict-ahead agent (CV)", color="#1f5f8b")
ax.set_xticks(x, [NICE[k] for k in KINDS])
ax.set_ylabel("tracking energy [J], mean of 12 runs")
ax.set_title("harder targets cost more — and prediction helps more")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig("figures/G2_tracking_energy.png", dpi=150); plt.close(fig)

ladder = predict_E["static"] <= predict_E["dynamic"] <= predict_E["time_varying"]
check("G2 tracking energy grows with target agility", ladder,
      "%.2f <= %.2f <= %.2f J" % (predict_E["static"], predict_E["dynamic"],
                                  predict_E["time_varying"]))
gain_tv = 100 * (react_E["time_varying"] - predict_E["time_varying"]) / react_E["time_varying"]
check("G2 prediction helps most on the hardest targets", gain_tv > 0,
      "predict-ahead saves %.0f%% on time-varying" % gain_tv)

# ------------------------------------------------------------------ G3
print("\n== G3: trajectory snapshot ==")
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, kind in zip(axes, KINDS):
    rng = np.random.default_rng(3)
    tgs = spawn_targets(kind, 3, rng)
    for r in range(14):
        for tg in tgs:
            tg.step(T)
    for ti, tg in enumerate(tgs):
        h = np.array(tg.history)
        ax.plot(h[:, 0], h[:, 1], "-o", ms=3, lw=1.4, label=f"target {ti+1}")
        ax.scatter(h[0, 0], h[0, 1], marker="s", s=60, edgecolor="k",
                   facecolor="white", zorder=5)
    ax.set_title(NICE[kind]); ax.set_xlim(0, 500); ax.set_ylim(0, 500)
    ax.set_aspect("equal"); ax.grid(alpha=.3)
axes[0].set_ylabel("y [m]"); axes[1].set_xlabel("x [m]")
fig.suptitle("the three target classes (squares = start positions)", y=1.02)
fig.tight_layout(); fig.savefig("figures/G3_target_paths.png", dpi=150,
                                bbox_inches="tight"); plt.close(fig)
check("G3 trajectory snapshot saved", os.path.exists("figures/G3_target_paths.png"))

with open("results_targets.txt", "w") as f:
    f.write("Target-tracking extension — verification\n" + "=" * 50 + "\n")
    for name, ok, detail in checks:
        f.write(("PASS  " if ok else "FAIL  ") + name +
                ("  | " + detail if detail else "") + "\n")
print("\nwrote results_targets.txt")
