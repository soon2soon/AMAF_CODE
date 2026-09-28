from __future__ import annotations

import math

import numpy as np


def confidence_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if n == 1:
        return mean, mean
    try:
        from scipy.stats import t
        crit = float(t.ppf((1 + confidence) / 2, n - 1))
    except Exception:
        crit = 1.96
    se = float(np.std(values, ddof=1) / math.sqrt(n))
    return mean - crit * se, mean + crit * se


def paired_effect_size(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    if len(d) < 2 or np.std(d, ddof=1) == 0:
        return float("nan")
    return float(np.mean(d) / np.std(d, ddof=1))


def paired_tests(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    out = {"n_pairs": len(a), "mean_difference": float(np.mean(a - b)) if len(a) else float("nan")}
    out["cohen_dz"] = paired_effect_size(a, b)
    if len(a) >= 2:
        try:
            from scipy.stats import ttest_rel, wilcoxon
            t_res = ttest_rel(a, b)
            out["paired_t_stat"] = float(t_res.statistic)
            out["paired_t_p"] = float(t_res.pvalue)
            try:
                w_res = wilcoxon(a, b)
                out["wilcoxon_stat"] = float(w_res.statistic)
                out["wilcoxon_p"] = float(w_res.pvalue)
            except ValueError:
                out["wilcoxon_stat"] = float("nan")
                out["wilcoxon_p"] = float("nan")
        except Exception:
            pass
    return out
