"""Synthetic minority oversampling (SMOTE-style) for ARP feature rows.

IMPORTANT - these rows are SYNTHETIC, not recorded traffic:
  * every synthetic row carries synthetic=1 and capture = "synthetic<-<source capture>";
  * they are interpolations between real packets of the SAME capture (same attack type),
    so they add class balance, not new information;
  * they are generated only from TRAINING data inside every evaluation and are never used
    for testing - all reported metrics are computed on real packets only;
  * the full set is also written to dataset/arp_dataset_synthetic.csv, separate from the real data.

Method: for a real row x_i pick one of its k nearest neighbours x_j (same capture, standardised
feature space) and create x_i + u * (x_j - x_i), u ~ U(0,1). Discrete columns are then rounded
back to valid values. The synthetic budget is spread over the minority class's captures by
"water-filling": the smallest captures are raised first, so one big capture (e.g. flooding)
does not dominate. Captures with fewer than MIN_REAL real rows are not oversampled.
"""
import numpy as np
import pandas as pd

from features import FEATURES

K_NEIGHBOURS = 5
MIN_REAL = 30                                   # don't oversample captures smaller than this
BINARY = ["is_broadcast", "is_gratuitous", "mac_conflict"]
COUNTS = ["ip_mac_count", "l2_ip_count", "l2_reply_rate_1s", "arp_reply_count"]
SYNTH_COLUMNS = FEATURES + ["label", "capture", "synthetic"]


def plan(counts, target_total, min_real=MIN_REAL):
    """Water-fill extra rows over captures so the class reaches target_total rows.

    counts: {capture: real rows of the minority class}. Returns {capture: n_synthetic}.
    """
    remaining = target_total - sum(counts.values())
    eligible = {c: n for c, n in counts.items() if n >= min_real}
    if remaining <= 0 or not eligible:
        return {}

    def need(level):
        return sum(max(0, level - n) for n in eligible.values())

    lo, hi = min(eligible.values()), min(eligible.values()) + remaining
    while lo < hi:                               # largest level with need(level) <= remaining
        mid = (lo + hi + 1) // 2
        if need(mid) <= remaining:
            lo = mid
        else:
            hi = mid - 1
    alloc = {c: max(0, lo - n) for c, n in eligible.items()}
    leftover = remaining - sum(alloc.values())
    for c in sorted(alloc, key=lambda c: eligible[c])[:max(leftover, 0)]:
        alloc[c] += 1
    return alloc


def _generate(real, n_new, rng, label, capture):
    X = real[FEATURES].to_numpy(dtype=float)
    n = len(X)
    k = min(K_NEIGHBOURS, n - 1)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = X / sd
    sq = (Z ** 2).sum(axis=1)
    dist = sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T
    np.fill_diagonal(dist, np.inf)
    nbrs = np.argsort(dist, axis=1)[:, :k]

    i = rng.integers(0, n, n_new)
    j = nbrs[i, rng.integers(0, k, n_new)]
    u = rng.random((n_new, 1))
    new = X[i] + u * (X[j] - X[i])

    out = pd.DataFrame(new, columns=FEATURES)
    out["op"] = out["op"].round().clip(1, 2)
    for c in BINARY:
        out[c] = (out[c] >= 0.5).astype(float)
    for c in COUNTS:
        out[c] = out[c].round().clip(lower=0)
    out["ip_mac_count"] = out["ip_mac_count"].clip(lower=1)
    out["l2_ip_count"] = out["l2_ip_count"].clip(lower=1)
    for c in ("log_interarrival_ms", "log_reply_delay_ms"):
        out[c] = out[c].clip(lower=0)
    out["label"] = label
    out["capture"] = f"synthetic<-{capture}"
    out["synthetic"] = 1
    return out


def make_synthetic(df, seed=42):
    """Return synthetic rows that balance the classes of REAL dataframe df (may be empty)."""
    counts = df["label"].value_counts()
    empty = pd.DataFrame(columns=SYNTH_COLUMNS)
    if len(counts) < 2 or counts.min() == counts.max():
        return empty
    minority = int(counts.idxmin())
    target = int(counts.max())
    mino = df[df["label"] == minority]
    alloc = plan(mino.groupby("capture").size().to_dict(), target)
    rng = np.random.default_rng(seed)
    parts = [
        _generate(mino[mino["capture"] == cap], n, rng, minority, cap)
        for cap, n in sorted(alloc.items()) if n > 0
    ]
    return pd.concat(parts, ignore_index=True)[SYNTH_COLUMNS] if parts else empty
