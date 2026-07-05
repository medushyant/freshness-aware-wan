"""Run the whole Phase-2 suite in order (playbook §0.6).

Usage (under .venv-awan):
    python run_awan_all.py            # assumes VLM cache exists (fast path)
    python run_awan_all.py --full     # also (re)runs VLM perception first

The Phase-1 regression gate (scripts/regress.sh, under .venv) is a separate,
mandatory companion — run it before and after.
"""

import subprocess
import sys

STEPS = [
    ("run_awan_wp0.py", []),
    ("run_awan_wp1.py", []),
    ("run_awan_wp2.py", []),
    ("run_awan_wp3.py", []),
    ("run_awan_wp4.py", []),
    ("export_awan_web_data.py", []),
]

if "--full" in sys.argv:
    STEPS.insert(3, ("scripts/wp3_perceive.py", ["20"]))

fails = []
for script, args in STEPS:
    print(f"\n=== {script} {' '.join(args)} ===")
    r = subprocess.run([sys.executable, script, *args])
    if r.returncode != 0:
        fails.append(script)

txt = open("results_awan.txt").read()
n_pass = txt.count("PASS  ")
n_fail = txt.count("FAIL  ")
print(f"\n================= A-WAN SUITE =================")
print(f"results_awan.txt: {n_pass} PASS / {n_fail} FAIL")
if fails:
    print("script errors:", fails)
sys.exit(1 if (fails or n_fail) else 0)
