"""Compare Naive Bayes with Logistic Regression and Random Forest on the REAL data.

Same evaluation as train.py (chronological 70/30 hold-out, same-network hold-out,
leave-one-capture-out, unseen hold-out capture). Nothing here touches the saved models.
Synthetic rows are never used. Reads the dataset CSVs written by `train.py`, so run that first.

Output: results/model_comparison.md and results/model_comparison.json
"""
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn

from config import load_config
from features import FEATURES, PROJECT_ROOT
from model_factory import make_lr, make_nb, make_rf
from train import chronological_split, summarize

DATASET_DIR = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results"


def make_models():
    cfg = load_config()
    return {"Naive Bayes": make_nb(cfg), "Logistic Regression": make_lr(cfg), "Random Forest": make_rf(cfg)}


def fit(model, train):
    return model.fit(train[FEATURES], train["label"])


def scores(model, X):
    return model.predict(X), model.predict_proba(X)[:, 1]


def holdout(df):
    train, test = chronological_split(df)
    y = test["label"].to_numpy()
    out = {}
    for name, m in make_models().items():
        fit(m, train)
        pred, proba = scores(m, test[FEATURES])
        out[name] = summarize(y, pred, proba)
    return out, int((test.label == 0).sum()), int((test.label == 1).sum())


def loco(df):
    rows = []
    for cap in df["capture"].unique():
        test = df[df["capture"] == cap]
        train = df[df["capture"] != cap]
        row = {"capture": cap, "label": int(test["label"].iloc[0]), "packets": len(test)}
        for name, m in make_models().items():
            fit(m, train)
            row[name] = float(m.predict(test[FEATURES]).mean())
        rows.append(row)
    return rows


def unseen(df, hold):
    models = {n: fit(m, df) for n, m in make_models().items()}
    return {n: float(m.predict(hold[FEATURES]).mean()) for n, m in models.items()}, models


def table(res):
    lines = ["| Model | Accuracy | Precision | Recall | F1 | FPR | ROC AUC | MCC |",
             "|---|---|---|---|---|---|---|---|"]
    for n, m in res.items():
        lines.append(f"| {n} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{m['f1']:.3f} | {m['false_positive_rate']:.3f} | {m.get('roc_auc', float('nan')):.3f} | "
                     f"{m['mcc']:.3f} |")
    return lines


def main():
    raw = DATASET_DIR / "arp_dataset_raw.csv"
    hold_csv = DATASET_DIR / "arp_dataset_holdout.csv"
    if not raw.exists():
        sys.exit("Run model_scripts/train.py first (it writes dataset/arp_dataset_raw.csv).")
    df = pd.read_csv(raw)
    hold = pd.read_csv(hold_csv) if hold_csv.exists() else pd.DataFrame()

    res_all, n_norm, n_att = holdout(df)
    envs = {}
    for env, sub in df.groupby("environment"):
        if sub["label"].nunique() < 2:
            continue
        r, tn, ta = holdout(sub)
        envs[env] = {"results": r, "test_normal": tn, "test_attack": ta}
    loco_rows = loco(df)
    unseen_fpr, full_models = (unseen(df, hold) if len(hold) else ({}, {}))

    rf = full_models.get("Random Forest")
    lr = full_models.get("Logistic Regression")
    importances = (dict(sorted(zip(FEATURES, map(float, rf.feature_importances_)),
                               key=lambda kv: -kv[1])) if rf is not None else {})
    coefs = (dict(sorted(zip(FEATURES, map(float, lr[-1].coef_[0])),
                         key=lambda kv: -abs(kv[1]))) if lr is not None else {})

    # ---- console
    print(f"\nAll-data hold-out (test: {n_norm} normal / {n_att} attack)")
    for n, m in res_all.items():
        print(f"  {n:20s} recall {m['recall']:.3f}  FPR {m['false_positive_rate']:.3f}  "
              f"precision {m['precision']:.3f}  MCC {m['mcc']:.3f}")
    for env, e in envs.items():
        print(f"\nSame-network {env}.x.x (test: {e['test_normal']} normal / {e['test_attack']} attack)")
        for n, m in e["results"].items():
            print(f"  {n:20s} recall {m['recall']:.3f}  FPR {m['false_positive_rate']:.3f}  "
                  f"precision {m['precision']:.3f}  MCC {m['mcc']:.3f}")
    print("\nLeave-one-capture-out (share flagged as attack)")
    for r in loco_rows:
        cls = "attack" if r["label"] else "normal"
        print(f"  {r['capture']:38s}{cls:>7s}{r['packets']:6d}  "
              + "  ".join(f"{n.split()[0][:2]} {r[n]:.2f}" for n in make_models()))
    if unseen_fpr:
        print("\nUnseen hold-out capture, false-positive rate: "
              + ", ".join(f"{n} {v:.3f}" for n, v in unseen_fpr.items()))
    print("\nRandom Forest feature importance:", {k: round(v, 3) for k, v in importances.items()})
    print("Logistic Regression coefficients:", {k: round(v, 2) for k, v in coefs.items()})

    # ---- files
    (RESULTS_DIR / "model_comparison.json").write_text(json.dumps(
        {"holdout": res_all, "same_network": envs, "leave_one_capture_out": loco_rows,
         "unseen_holdout_fpr": unseen_fpr, "rf_feature_importance": importances,
         "lr_coefficients": coefs}, indent=2), encoding="utf-8")

    md = ["# Model comparison (auto-generated by `compare_models.py`)", "",
          f"Generated: {datetime.now():%Y-%m-%d %H:%M}  |  scikit-learn {sklearn.__version__}", "",
          f"REAL packets only ({len(df)} training pool + {len(hold)} unseen hold-out); no synthetic rows. "
          "Logistic Regression and Random Forest use `class_weight='balanced'`; Logistic Regression is standardised. "
          "Saved models in `models/` are unchanged.", "",
          f"## All-data hold-out (last 30% of each capture; {n_norm} normal / {n_att} attack)", ""]
    md += table(res_all)
    for env, e in envs.items():
        md += ["", f"## Same-network hold-out ({env}.x.x; {e['test_normal']} normal / {e['test_attack']} attack)", ""]
        md += table(e["results"])
    names = list(make_models())
    md += ["", "## Leave-one-capture-out", "",
           "Share of the held-out capture flagged as attack (normal capture = FPR, attack capture = recall).", "",
           "| Capture | Class | Packets | " + " | ".join(names) + " |",
           "|---|---|---|" + "---|" * len(names)]
    for r in loco_rows:
        md.append(f"| {r['capture']} | {'attack' if r['label'] else 'normal'} | {r['packets']} | "
                  + " | ".join(f"{r[n]:.2f}" for n in names) + " |")
    if unseen_fpr:
        md += ["", f"## Unseen hold-out capture ({len(hold)} packets, normal): false-positive rate", "",
               "| " + " | ".join(names) + " |", "|" + "---|" * len(names),
               "| " + " | ".join(f"{unseen_fpr[n]:.3f}" for n in names) + " |"]
    md += ["", "## What each model relies on (trained on the full training pool)", "",
           "Random Forest importance: " + ", ".join(f"`{k}` {v:.2f}" for k, v in importances.items()), "",
           "Logistic Regression standardised coefficients: "
           + ", ".join(f"`{k}` {v:+.2f}" for k, v in coefs.items())]
    (RESULTS_DIR / "model_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nWrote results/model_comparison.md and model_comparison.json")


if __name__ == "__main__":
    main()
