"""Choose hyper-parameters by leave-one-capture-out cross-validation on the TRAINING pool only.

    python model_scripts/train.py     # once, to create dataset/arp_dataset_features.csv
    python model_scripts/tune.py      # writes config.json + results/tuning.md / tuning.json
    python model_scripts/train.py     # retrain with the chosen settings

How a candidate is scored (one number, computed per capture then averaged so a big capture cannot dominate):
    score = mean recall over held-out ATTACK captures  -  mean false-alarm rate over held-out NORMAL captures
For each fold one whole capture is held out, the model is trained on the other captures, and the share of the
held-out capture it flags is recorded (recall for an attack capture, false-alarm rate for a normal one).

The unseen hold-out capture in pcap_files/holdout/ is never read here, so it stays an honest final test.

Selection rule: take the best-scoring candidate, but keep the CURRENT setting when it is within TOLERANCE of the
best (a tie is not a reason to change a setting).
"""
import itertools
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from config import load_config, save_config
from features import FEATURES, PROJECT_ROOT
from model_factory import make_iso, make_lr, make_nb, make_rf

TOLERANCE = 0.005
GRIDS = {
    "nb_var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3],
    "if_contamination": [0.005, 0.01, 0.02, 0.05],
    "rf_n_estimators": [100, 200, 400],
    "rf_min_samples_leaf": [1, 5],
    "lr_C": [0.1, 1.0, 10.0],
}


def score_from_flags(flags, labels):
    """flags: {capture: share flagged}; labels: {capture: 0/1} -> (score, mean recall, mean false-alarm rate)."""
    rec = np.mean([flags[c] for c in flags if labels[c] == 1])
    fpr = np.mean([flags[c] for c in flags if labels[c] == 0])
    return float(rec - fpr), float(rec), float(fpr)


def folds(df):
    for cap in df["capture"].unique():
        yield cap, df[df["capture"] != cap], df[df["capture"] == cap]


def main():
    feat_csv = PROJECT_ROOT / "dataset" / "arp_dataset_features.csv"
    if not feat_csv.exists():
        sys.exit("Run model_scripts/train.py first (it writes dataset/arp_dataset_features.csv).")
    df = pd.read_csv(feat_csv)
    labels = df.groupby("capture")["label"].first().to_dict()
    cfg = load_config()
    print(f"Leave-one-capture-out CV on {len(df)} training packets, {len(labels)} captures "
          f"({sum(labels.values())} attack / {len(labels) - sum(labels.values())} normal)")

    # ---- per-fold predictions for each independent model setting
    nb_flags = {v: {} for v in GRIDS["nb_var_smoothing"]}
    if_flags = {v: {} for v in GRIDS["if_contamination"]}
    rf_flags = {v: {} for v in itertools.product(GRIDS["rf_n_estimators"], GRIDS["rf_min_samples_leaf"])}
    lr_flags = {v: {} for v in GRIDS["lr_C"]}
    for cap, train, test in folds(df):
        X_tr, y_tr, X_te = train[FEATURES], train["label"], test[FEATURES]
        normal = train[train["label"] == 0][FEATURES]
        for v in nb_flags:
            nb_flags[v][cap] = make_nb({**cfg, "nb_var_smoothing": v}).fit(X_tr, y_tr).predict(X_te)
        for v in if_flags:
            iso = make_iso({**cfg, "if_contamination": v}).fit(normal)
            if_flags[v][cap] = (iso.predict(X_te) == -1).astype(int)
        for (n, leaf) in rf_flags:
            rf_flags[(n, leaf)][cap] = make_rf({**cfg, "rf_n_estimators": n, "rf_min_samples_leaf": leaf}).fit(X_tr, y_tr).predict(X_te)
        for v in lr_flags:
            lr_flags[v][cap] = make_lr({**cfg, "lr_C": v}).fit(X_tr, y_tr).predict(X_te)
        print(f"  fold done: {cap}")

    share = lambda d: {c: float(p.mean()) for c, p in d.items()}

    # ---- ensemble (shipped): grid over Naive Bayes smoothing x Isolation Forest contamination
    ens_rows = []
    for vs, ct in itertools.product(GRIDS["nb_var_smoothing"], GRIDS["if_contamination"]):
        flags = {c: float(np.maximum(nb_flags[vs][c], if_flags[ct][c]).mean()) for c in labels}
        ens_rows.append(((vs, ct),) + score_from_flags(flags, labels))
    cur_key = (cfg["nb_var_smoothing"], cfg["if_contamination"])
    best_ens = max(ens_rows, key=lambda r: r[1])
    cur_ens = next(r for r in ens_rows if r[0] == cur_key)
    chosen_ens = cur_key if best_ens[1] - cur_ens[1] <= TOLERANCE else best_ens[0]

    # ---- single models for reference
    rf_rows = [(k,) + score_from_flags(share(v), labels) for k, v in rf_flags.items()]
    lr_rows = [(k,) + score_from_flags(share(v), labels) for k, v in lr_flags.items()]
    cur_rf = (cfg["rf_n_estimators"], cfg["rf_min_samples_leaf"])
    best_rf = max(rf_rows, key=lambda r: r[1])
    chosen_rf = cur_rf if best_rf[1] - next(r for r in rf_rows if r[0] == cur_rf)[1] <= TOLERANCE else best_rf[0]
    best_lr = max(lr_rows, key=lambda r: r[1])
    chosen_lr = cfg["lr_C"] if best_lr[1] - next(r for r in lr_rows if r[0] == cfg["lr_C"])[1] <= TOLERANCE else best_lr[0]

    new_cfg = dict(cfg)
    new_cfg.update({"nb_var_smoothing": chosen_ens[0], "if_contamination": chosen_ens[1],
                    "rf_n_estimators": chosen_rf[0], "rf_min_samples_leaf": chosen_rf[1], "lr_C": chosen_lr})
    new_cfg["tuning"] = {
        "method": "leave-one-capture-out CV on the training pool; score = mean attack recall - mean normal false-alarm rate",
        "tolerance": TOLERANCE, "tuned_at": datetime.now().isoformat(timespec="seconds"),
        "ensemble_cv_score": round(next(r for r in ens_rows if r[0] == chosen_ens)[1], 4),
        "rf_cv_score": round(next(r for r in rf_rows if r[0] == chosen_rf)[1], 4),
        "lr_cv_score": round(next(r for r in lr_rows if r[0] == chosen_lr)[1], 4),
    }
    changed = {k: (cfg[k], new_cfg[k]) for k in GRIDS if cfg[k] != new_cfg[k]}
    save_config(new_cfg)

    # ---- report
    def fmt(v):
        return f"{v:g}" if isinstance(v, float) else str(v)
    md = ["# Hyper-parameter tuning (auto-generated by `tune.py`)", "",
          f"Generated: {datetime.now():%Y-%m-%d %H:%M}", "",
          f"Method: leave-one-capture-out cross-validation on the {len(df)}-packet training pool "
          f"({len(labels)} captures). Score = mean recall on held-out attack captures minus mean false-alarm rate on "
          f"held-out normal captures (per-capture, then averaged). The unseen hold-out capture is **not** used. "
          f"A setting is only changed if a candidate beats it by more than {TOLERANCE}.", "",
          "## Ensemble (Naive Bayes OR Isolation Forest) - shipped", "",
          "| nb_var_smoothing | if_contamination | CV score | recall | false-alarm rate | |", "|---|---|---|---|---|---|"]
    for (k, sc, rc, fp) in sorted(ens_rows, key=lambda r: -r[1]):
        tag = ("**chosen**" if k == chosen_ens else "") + (" (previous)" if k == cur_key else "")
        md.append(f"| {fmt(k[0])} | {fmt(k[1])} | {sc:.3f} | {rc:.3f} | {fp:.3f} | {tag} |")
    md += ["", "## Random Forest (compared, not shipped)", "", "| n_estimators | min_samples_leaf | CV score | recall | false-alarm rate | |", "|---|---|---|---|---|---|"]
    for (k, sc, rc, fp) in sorted(rf_rows, key=lambda r: -r[1]):
        md.append(f"| {k[0]} | {k[1]} | {sc:.3f} | {rc:.3f} | {fp:.3f} | {'**chosen**' if k == chosen_rf else ''}{' (previous)' if k == cur_rf else ''} |")
    md += ["", "## Logistic Regression (compared, not shipped)", "", "| C | CV score | recall | false-alarm rate | |", "|---|---|---|---|---|"]
    for (k, sc, rc, fp) in sorted(lr_rows, key=lambda r: -r[1]):
        md.append(f"| {fmt(k)} | {sc:.3f} | {rc:.3f} | {fp:.3f} | {'**chosen**' if k == chosen_lr else ''}{' (previous)' if k == cfg['lr_C'] else ''} |")
    md += ["", "## Result", "",
           ("Changed settings: " + ", ".join(f"`{k}` {fmt(a)} -> {fmt(b)}" for k, (a, b) in changed.items())) if changed
           else "No setting changed: the current values were within the tolerance of the best candidate."]
    (PROJECT_ROOT / "results").mkdir(exist_ok=True)
    (PROJECT_ROOT / "results" / "tuning.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (PROJECT_ROOT / "results" / "tuning.json").write_text(json.dumps({
        "ensemble": [{"nb_var_smoothing": k[0], "if_contamination": k[1], "score": s, "recall": r, "false_alarm": f} for k, s, r, f in ens_rows],
        "random_forest": [{"n_estimators": k[0], "min_samples_leaf": k[1], "score": s, "recall": r, "false_alarm": f} for k, s, r, f in rf_rows],
        "logistic_regression": [{"C": k, "score": s, "recall": r, "false_alarm": f} for k, s, r, f in lr_rows],
        "chosen": {k: new_cfg[k] for k in GRIDS}, "changed": {k: list(v) for k, v in changed.items()}}, indent=2), encoding="utf-8")

    print("\nEnsemble best:", best_ens[0], f"score {best_ens[1]:.3f}  | previous {cur_key} score {cur_ens[1]:.3f}")
    print("Changed:", changed or "nothing (current values within tolerance)")
    print("Wrote config.json, results/tuning.md, results/tuning.json")


if __name__ == "__main__":
    main()
