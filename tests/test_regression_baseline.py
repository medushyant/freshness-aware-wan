"""Regression gate as a test (playbook §1.2). Fast mode just re-checks the
current results files against baseline/ identifiers; the slow marker actually
re-executes every frozen runner via scripts/regress.sh under .venv."""

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _ids(path):
    if not path.exists():
        return None
    return sorted(set(re.findall(r"^(?:PASS|FAIL)\s+(\S+)", path.read_text(), re.M)))


def test_identifier_sets_match_baseline():
    for base in sorted((ROOT / "baseline").glob("results*.txt")):
        cur = ROOT / base.name
        assert _ids(cur) == _ids(base), f"identifier drift in {base.name}"


def test_no_fail_lines_in_current_results():
    for base in sorted((ROOT / "baseline").glob("results*.txt")):
        cur = ROOT / base.name
        assert not re.findall(r"^FAIL", cur.read_text(), re.M), f"FAIL in {base.name}"


@pytest.mark.slow
def test_full_rerun_gate():
    r = subprocess.run(["bash", str(ROOT / "scripts" / "regress.sh")],
                       capture_output=True, text=True, timeout=3600)
    assert "REGRESSION GATE: GREEN" in r.stdout, r.stdout[-2000:]
