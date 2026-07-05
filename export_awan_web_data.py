"""Dump web/data_awan.json for the A-WAN website tab (playbook §5.4).

Same honesty rule as Phase 1: every number here is read from machine-written
results.json files under runs/ — nothing is typed in.
"""

import json
import pathlib
import re
import shutil

from awan.runio import latest_results

ROOT = pathlib.Path(__file__).resolve().parent
web_figs = ROOT / "web" / "figures_awan"
web_figs.mkdir(exist_ok=True)

wp1 = latest_results("wp1")
wp2 = latest_results("wp2")
wp4 = latest_results("wp4")
try:
    wp3 = latest_results("wp3")
except FileNotFoundError:
    wp3 = None

checks = []
txt = (ROOT / "results_awan.txt").read_text()
for line in txt.splitlines():
    m = re.match(r"^(PASS|FAIL)\s+(\S+)\s+(.*?)(?:\s+\|\s+(.*))?$", line)
    if m:
        checks.append({"status": m.group(1), "id": m.group(2),
                       "claim": m.group(3), "detail": m.group(4) or ""})

data = {
    "checks": checks,
    "n_pass": sum(c["status"] == "PASS" for c in checks),
    "wp1": {
        "energy": wp1["energy"],
        "dropout": wp1["dropout"],
        "breakeven": wp1["breakeven"],
        "llm_stats": wp1["llm_stats"],
        "value_model_r2": wp1["value_model_r2"],
    },
    "wp2": {
        "viol_rates": wp2["viol_rates"],
        "conformal": wp2["conformal"],
        "mobility": wp2["mobility"],
        "predictor_rmse_db": wp2["predictor_rmse_db"],
    },
    "wp3": None if wp3 is None else {
        "g1_validity": wp3["g1_validity"],
        "h7": wp3["h7"],
        "rate_fidelity": wp3["rate_fidelity"],
        "h10_verdict_f1_delta": wp3["h10_verdict_f1_delta"],
        "grounding_gap": wp3["grounding_gap"],
        "corruption": wp3["corruption"],
        "conformal_real": wp3["conformal_real"],
        "meter": {k: v for k, v in wp3["meter"].items() if k != "wall_s"},
    },
    "wp4": {
        "grand": wp4["grand"],
        "i2": wp4["i2"],
        "i3": wp4["i3"],
        "i4": wp4["i4"],
        "runtime_ms": wp4["i5_runtime_ms"],
        "runtime_N": wp4["i5_N"],
        "q_db": wp4["q_db_used"],
    },
}

figs = sorted((ROOT / "figures" / "awan").glob("*.png"))
for f in figs:
    shutil.copy2(f, web_figs / f.name)
data["figures"] = [f.name for f in figs]

out = ROOT / "web" / "data_awan.json"
out.write_text(json.dumps(data))
print(f"wrote {out} ({out.stat().st_size/1e3:.0f} kB), {len(figs)} figures, "
      f"{data['n_pass']}/{len(checks)} checks PASS")
