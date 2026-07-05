"""Inference energy metering (playbook §4.4, H8).

macOS `powermetrics` needs sudo (§0.9c). We attempt a passwordless probe once;
in this sandboxed session it is unavailable, so local numbers use the MODELED
fallback — wall-time x device power envelope — and are labeled MODELED
everywhere. notebooks/02_vlm_energy.ipynb provides the MEASURED path (Colab
NVML at 10 Hz with idle-baseline subtraction). The grounding-gap conclusion
(H8) is robust to this: the paper's tau*f^2*W compute term is orders of
magnitude below ANY defensible device envelope.
"""

import shutil
import subprocess

# Apple-silicon package power envelope while a small VLM decodes on CPU.
# Conservative low estimate (favors the paper's model in the gap table).
MODELED_ACTIVE_W = 8.0
MODELED_IDLE_W = 1.5


def powermetrics_available():
    if shutil.which("powermetrics") is None:
        return False
    try:
        r = subprocess.run(["sudo", "-n", "powermetrics", "-i", "1", "-n", "1",
                            "--samplers", "cpu_power"],
                           capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def modeled_energy_j(wall_s):
    """MODELED: (active - idle) envelope x wall time."""
    return (MODELED_ACTIVE_W - MODELED_IDLE_W) * wall_s


def mj_per_token(stats, source="MODELED"):
    """Split prefill (prompt) vs decode (generated) by the standard 1:3
    per-token cost ratio (prefill is compute-bound and batched, decode is
    memory-bound; Samsi et al. ratio) under the total MODELED energy."""
    E_total_mJ = modeled_energy_j(stats["wall_s"]) * 1e3
    tin, tout = stats["tok_in"], stats["tok_out"]
    if tin + 3 * tout == 0:
        return None
    unit = E_total_mJ / (tin + 3 * tout)
    return {"prefill_mJ_per_tok": unit, "decode_mJ_per_tok": 3 * unit,
            "total_mJ": E_total_mJ, "tok_in": tin, "tok_out": tout,
            "wall_s": stats["wall_s"], "source": source}


def paper_compute_energy_j(bits, p, rho=0.0, eta=0.6):
    """The paper's Eq. (6)-(7) energy for 'processing' the same payload,
    with Table-I constants — the term the grounding gap indicts."""
    W = bits * (p["C_base"] + p["C_gen"] * p["gamma"] * (1 - rho) *
                __import__("numpy").log(1.0 / eta))
    return p["tau"] * p["f_cpu"] ** 2 * W
