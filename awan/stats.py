"""Statistics protocol (playbook §7) — implemented once, used by every runner."""

import numpy as np
from scipy import stats as sps


def mean_ci(x, conf=0.95):
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, 0.0
    se = x.std(ddof=1) / np.sqrt(len(x))
    h = se * sps.t.ppf(0.5 + conf / 2, len(x) - 1)
    return float(m), float(h)


def bootstrap_ci(x, conf=0.95, n_boot=10_000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [50 * (1 - conf), 50 * (1 + conf)])
    return float(x.mean()), float(lo), float(hi)


def paired_wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank on paired samples (same seeds)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    return float(sps.wilcoxon(a, b, zero_method="wilcox").pvalue)


def holm(pvals):
    """Holm step-down correction; returns adjusted p-values in input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def cliffs_delta(a, b):
    """Effect size: P(a>b) - P(a<b) over all cross pairs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def summarize_pair(name_a, a, name_b, b):
    ma, ha = mean_ci(a)
    mb, hb = mean_ci(b)
    return {name_a: {"mean": ma, "ci95": ha},
            name_b: {"mean": mb, "ci95": hb},
            "wilcoxon_p": paired_wilcoxon(a, b),
            "cliffs_delta": cliffs_delta(a, b)}
