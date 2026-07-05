#!/bin/bash
# Phase-1 regression gate (playbook §0.2 / §1.2). Runs all seven frozen
# runners under .venv and diffs the PASS identifier set against baseline/.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
for r in run_direction1 run_direction2 run_repro_scaling run_upgrades \
         run_targets run_freshness run_metaheuristics; do
  echo "=== $r ==="
  python $r.py > /tmp/awan_regress_$r.log 2>&1 || { echo "RUNNER FAILED: $r"; exit 1; }
done
python - <<'EOF'
import re, sys, pathlib
ids = lambda f: sorted(set(re.findall(r"^(?:PASS|FAIL)\s+(\S+)", pathlib.Path(f).read_text(), re.M))) if pathlib.Path(f).exists() else []
fails = lambda f: re.findall(r"^FAIL\s+.*$", pathlib.Path(f).read_text(), re.M) if pathlib.Path(f).exists() else []
ok = True
for f in sorted(pathlib.Path("baseline").glob("results*.txt")):
    cur = pathlib.Path(f.name)
    b, c = ids(f), ids(cur)
    fl = fails(cur)
    if b != c: ok = False; print(f"IDENTIFIER DRIFT {f.name}: baseline={b} current={c}")
    if fl:     ok = False; print(f"FAILURES in {f.name}: {fl}")
print("REGRESSION GATE:", "GREEN" if ok else "RED")
sys.exit(0 if ok else 1)
EOF
