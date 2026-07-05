# CLAUDE.md — BTP_ILAC_WAN

Phase 2 starts now. Read `01_AWAN_PROJECT_BRIEF.md` then
`02_AWAN_IMPLEMENTATION_PLAYBOOK.md` and execute the playbook top-to-bottom.
All existing Phase-1 code and results are frozen — Prime Directives §0 of the
playbook are binding: never edit `wan/`, `experiments/`, `web/` (except additive
A-WAN files), `scripts/`, `docs/` (existing files), `reference/`, the seven
`run_*.py` Phase-1 runners, `export_web_data.py`, `opv2v_colab.py`,
`results*.txt`, or existing parts of `PROJECT_REPORT.md` (append-only).

All Phase-2 code lives under `awan/` + the `run_awan_*.py` runners, tested with
`.venv-awan` (Python 3.12). Phase-1 runners keep using `.venv` (Python 3.14).
Run `bash scripts/regress.sh` before and after every work package.

Conventions: human-readable code, few comments, every number machine-produced
(no hard-coded results), every energy figure labeled MEASURED or MODELED,
honest negatives reported. Decisions log: `docs/DECISIONS.md`.
