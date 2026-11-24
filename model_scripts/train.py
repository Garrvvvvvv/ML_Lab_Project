"""Train and evaluate the ARP-spoofing detectors.

    python model_scripts/train.py

Pipeline: pcaps -> per-packet features -> dataset CSVs -> Gaussian Naive Bayes
(supervised) + Isolation Forest (trained on normal traffic only) -> evaluation
of NB, IF and their OR-ensemble -> saved models + results/ report.

Evaluation is deliberately stricter than a random packet split:
  1. Chronological split per capture (first 70% of every capture trains, the
     last 30% tests) so near-identical neighbouring packets don't straddle
     the train/test boundary.
  2. Leave-one-capture-out: train on all other captures, test on the held-out
     one - the honest measure of generalisation to unseen traffic/attack types.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
    log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
    roc_curve,
)

from augment import SYNTH_COLUMNS, make_synthetic
from config import load_config
from features import FEATURES, PROJECT_ROOT, discover_pcaps, extract_arp_features
from model_factory import make_iso, make_nb
from runlog import data_manifest, write_run

CFG = load_config()                      # settings live in config.json (tuned by tune.py)
RANDOM_STATE = CFG["seed"]
TRAIN_FRACTION = CFG["train_fraction"]
IF_CONTAMINATION = CFG["if_contamination"]   # share of normal training packets the IF flags by construction
# SYNTHETIC oversampling is OFF for the shipped models (it never improved real-data results and raised
# the false-positive rate on one unseen capture). `--augment` trains the headline models with it.
# The augmented-vs-real ablation is always computed and reported either way.
AUGMENT = "--augment" in sys.argv

DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def build_dataset(pcap_dir=None, required=True):
    captures = discover_pcaps(pcap_dir) if pcap_dir else discover_pcaps()
    if not captures:
        if required:
            sys.exit("No .pcap/.pcapng files found in pcap_files/")
        return pd.DataFrame()
    print("Captures found (label by name: 'normal' -> 0, otherwise 1):")
    frames = []
    for path, label in captures:
        df = extract_arp_features(path)
        df["label"] = label
        print(f"  {path.name:42s} label={label}  ARP packets={len(df)}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if required and df["label"].nunique() < 2:
        sys.exit("Need at least one normal and one attack capture.")
    return df


def tag_environment(df):
    """Tag each capture with its network (most common first two octets of sender IPs).

    Attack captures and the first normal capture come from one LAN; later normal captures
    come from a different network. Mixing them lets a model learn 'which network' instead of
    'attack', so we also evaluate inside each network separately.
    """
    ips = df["src_ip"].where(df["src_ip"] != "0.0.0.0")
    net = ips.str.split(".").str[:2].str.join(".")
    env = net.groupby(df["capture"]).agg(
        lambda s: s.mode().iat[0] if s.notna().any() else "unknown")
    df["environment"] = df["capture"].map(env)
    return df


def chronological_split(df):
    """First TRAIN_FRACTION of each capture -> train, remainder -> test."""
    train_idx, test_idx = [], []
    for _, grp in df.groupby("capture", sort=False):
        grp = grp.sort_values("pkt_time", kind="stable")
        cut = max(1, int(round(len(grp) * TRAIN_FRACTION)))
        train_idx += list(grp.index[:cut])
        test_idx += list(grp.index[cut:])
    return df.loc[train_idx], df.loc[test_idx]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def fit_models(train_df, augment=True):
    """Fit NB (+ synthetic minority rows when augment) and IF (real normal rows only).

    Synthetic rows are generated from `train_df` only, so they can never leak into a test set.
    """
    fit_df = train_df
    if augment:
        syn = make_synthetic(train_df, RANDOM_STATE)
        if len(syn):
            cols = FEATURES + ["label"]
            fit_df = pd.concat([train_df[cols], syn[cols]], ignore_index=True)
    nb = make_nb(CFG).fit(fit_df[FEATURES], fit_df["label"])
    normal = train_df[train_df["label"] == 0]            # real traffic only
    iso = None
    if len(normal):
        iso = make_iso(CFG).fit(normal[FEATURES])
    return nb, iso


def predict_all(nb, iso, X):
    """Return dict of 0/1 predictions (+ continuous scores) per detector."""
    out = {"nb": nb.predict(X), "nb_score": nb.predict_proba(X)[:, 1]}
    if iso is not None:
        out["if"] = (iso.predict(X) == -1).astype(int)       # -1 = anomaly
        out["if_score"] = -iso.score_samples(X)               # higher = more anomalous
        out["ens"] = np.maximum(out["nb"], out["if"])         # OR logic
    return out


def summarize(y, pred, score=None, proba=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else float("nan"),
        "kappa": cohen_kappa_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if score is not None and len(set(y)) == 2:
        m["roc_auc"] = roc_auc_score(y, score)
    if proba is not None:
        m["log_loss"] = log_loss(y, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1])
    return m


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def holdout_eval(train_df, test_df, nb, iso):
    y = test_df["label"].to_numpy()
    X = test_df[FEATURES]
    p = predict_all(nb, iso, X)
    nb_proba = nb.predict_proba(X)
    results = {
        "Naive Bayes": summarize(y, p["nb"], p["nb_score"], nb_proba),
        "Isolation Forest": summarize(y, p["if"], p["if_score"]),
        "Ensemble (NB OR IF)": summarize(y, p["ens"]),
    }
    return results, y, p


def per_environment_eval(df, augment=True):
    """Chronological hold-out restricted to one network at a time (only networks that
    contain both normal and attack traffic can be scored)."""
    out = {}
    for env, sub in df.groupby("environment"):
        if sub["label"].nunique() < 2:
            continue
        tr, te = chronological_split(sub)
        if te["label"].nunique() < 2 or (tr["label"] == 0).sum() == 0:
            continue
        nb, iso = fit_models(tr, augment)
        res, _, _ = holdout_eval(tr, te, nb, iso)
        out[env] = {
            "captures": sorted(sub["capture"].unique().tolist()),
            "test_normal": int((te.label == 0).sum()),
            "test_attack": int((te.label == 1).sum()),
            "results": res,
        }
    return out


def leave_one_capture_out(df, augment=True):
    """Train on every other capture, test on the held-out one."""
    rows = []
    for cap in df["capture"].unique():
        test = df[df["capture"] == cap]
        train = df[df["capture"] != cap]
        label = int(test["label"].iloc[0])
        row = {"capture": cap, "label": label, "packets": len(test)}
        if train["label"].nunique() < 2:
            row["note"] = "skipped (no other class left to train on)"
            rows.append(row)
            continue
        nb, iso = fit_models(train, augment)
        p = predict_all(nb, iso, test[FEATURES])
        row["nb_flagged"] = float(p["nb"].mean())
        if iso is not None:
            row["if_flagged"] = float(p["if"].mean())
            row["ens_flagged"] = float(p["ens"].mean())
        else:
            row["note"] = "IF skipped (no other normal capture) - record a 2nd normal capture"
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def fmt(x):
    return "  n/a " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:6.4f}"


def print_holdout(results):
    cols = ["accuracy", "precision", "recall", "f1", "false_positive_rate",
            "roc_auc", "kappa", "mcc"]
    print(f"\n{'Detector':22s}" + "".join(f"{c[:9]:>11s}" for c in cols))
    for name, m in results.items():
        print(f"{name:22s}" + "".join(f"{fmt(m.get(c)):>11s}" for c in cols))
    for name, m in results.items():
        (tn, fp), (fn, tp) = m["confusion_matrix"]
        print(f"\nConfusion matrix - {name}  (rows=actual, cols=predicted)")
        print(f"                 Pred Normal  Pred Attack\n"
              f"  Actual Normal  {tn:11d}  {fp:11d}\n"
              f"  Actual Attack  {fn:11d}  {tp:11d}")


def print_environment(envs):
    for env, e in envs.items():
        print(f"\nSame-network hold-out: {env}.x.x  "
              f"(test: {e['test_normal']} normal / {e['test_attack']} attack)")
        print_holdout(e["results"])


def evaluate_unseen(hold, models):
    """Fraction of each never-trained-on capture flagged as attack, per detector."""
    rows = []
    for cap, grp in hold.groupby("capture"):
        row = {"capture": cap, "label": int(grp["label"].iloc[0]), "packets": len(grp)}
        for tag, (nb, iso) in models.items():
            p = predict_all(nb, iso, grp[FEATURES])
            row[f"nb_{tag}"] = float(p["nb"].mean())
            row[f"if_{tag}"] = float(p["if"].mean())
            row[f"ens_{tag}"] = float(p["ens"].mean())
        rows.append(row)
    return rows


def print_unseen(rows):
    print("\nUnseen held-out captures (never used for training; share flagged as attack)")
    print("  normal capture -> false-positive rate; attack capture -> recall")
    for r in rows:
        cls = "attack" if r["label"] else "normal"
        print(f"  {r['capture']:38s}{cls:>8s}{r['packets']:7d}   "
              f"NB {r['nb_real']:.3f}  IF {r['if_real']:.3f}  Ensemble {r['ens_real']:.3f}"
              f"   (with synthetic rows: ensemble {r['ens_aug']:.3f})")


def print_ablation(results, results_real, title):
    print(f"\n{title}: augmented (synthetic) vs real-only training, REAL test packets")
    print(f"  {'Detector':22s}{'recall':>16s}{'false-pos rate':>20s}{'precision':>16s}")
    print(f"  {'':22s}{'aug / real':>16s}{'aug / real':>20s}{'aug / real':>16s}")
    for name in results:
        a, r = results[name], results_real[name]
        print(f"  {name:22s}{a['recall']:8.3f}/{r['recall']:.3f}"
              f"{a['false_positive_rate']:12.3f}/{r['false_positive_rate']:.3f}"
              f"{a['precision']:8.3f}/{r['precision']:.3f}")


def print_loco(rows):
    print("\nLeave-one-capture-out (fraction of the held-out capture flagged as attack)")
    print("  normal capture -> this is the false-positive rate; attack capture -> recall")
    print(f"  {'capture':38s}{'class':>8s}{'pkts':>7s}{'NB':>8s}{'IF':>8s}{'Ens':>8s}")
    for r in rows:
        cls = "attack" if r["label"] else "normal"
        if "nb_flagged" in r:
            print(f"  {r['capture']:38s}{cls:>8s}{r['packets']:7d}"
                  f"{r['nb_flagged']:8.2f}{r.get('if_flagged', float('nan')):8.2f}"
                  f"{r.get('ens_flagged', float('nan')):8.2f}"
                  + (f"   {r['note']}" if "note" in r else ""))
        else:
            print(f"  {r['capture']:38s}{cls:>8s}{r['packets']:7d}   {r['note']}")


def write_markdown(path, df, train_df, test_df, results, loco, envs, aug):
    n_norm = int((df.label == 0).sum())
    lines = [
        "# Latest evaluation (auto-generated by `train.py`)", "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}  |  scikit-learn {sklearn.__version__}", "",
        f"Dataset: **{len(df)} REAL** training-pool ARP packets ({n_norm} normal, "
        f"{len(df) - n_norm} attack) from {df.capture.nunique()} captures"
        f"{' plus ' + str(sum(r['packets'] for r in aug['unseen'])) + ' real held-out packets' if aug.get('unseen') else ''}. "
        f"Chronological split: "
        f"{len(train_df)} train / {len(test_df)} test.", "",
        f"**Shipped models are trained on {'REAL + SYNTHETIC' if aug['enabled'] else 'REAL packets only'}.** "
        f"{aug['n_syn']} SYNTHETIC attack rows (SMOTE-style interpolation, flagged `synthetic=1`, "
        f"stored separately in `dataset/arp_dataset_synthetic.csv`, balanced file "
        f"`dataset/arp_dataset_balanced.csv`) are available to balance the classes; the ablation "
        f"below compares both. Synthetic rows are generated from the training part only; every "
        f"metric in this file is computed on REAL packets only.", "",
        "## Hold-out (last 30% of every capture)", "",
        "| Detector | Accuracy | Precision | Recall | F1 | FPR | ROC AUC | MCC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {fmt(m['accuracy'])} | {fmt(m['precision'])} | {fmt(m['recall'])} | "
            f"{fmt(m['f1'])} | {fmt(m['false_positive_rate'])} | {fmt(m.get('roc_auc'))} | "
            f"{fmt(m['mcc'])} |")
    lines += ["", "Confusion matrices (`[[TN, FP], [FN, TP]]`):", ""]
    for name, m in results.items():
        lines.append(f"- {name}: `{m['confusion_matrix']}`")
    for env, e in envs.items():
        lines += ["", f"## Same-network hold-out ({env}.x.x)", "",
                  "Only captures from this network; chronological 70/30 split "
                  f"(test: {e['test_normal']} normal / {e['test_attack']} attack). "
                  "This is the fairest comparison: normal and attack traffic were recorded "
                  "in the same environment.", "",
                  "| Detector | Accuracy | Precision | Recall | F1 | FPR | ROC AUC | MCC |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, m in e["results"].items():
            lines.append(
                f"| {name} | {fmt(m['accuracy'])} | {fmt(m['precision'])} | {fmt(m['recall'])} | "
                f"{fmt(m['f1'])} | {fmt(m['false_positive_rate'])} | {fmt(m.get('roc_auc'))} | "
                f"{fmt(m['mcc'])} |")
    if aug.get("unseen"):
        lines += ["", "## Unseen held-out captures", "",
                  "Recorded separately, never used for training. Normal capture = false-positive "
                  "rate; attack capture = recall.", "",
                  "| Capture | Class | Packets | NB | IF | Ensemble | Ensemble (trained with synthetic rows) |",
                  "|---|---|---|---|---|---|---|"]
        for r in aug["unseen"]:
            cls = "attack" if r["label"] else "normal"
            lines.append(f"| {r['capture']} | {cls} | {r['packets']} | {r['nb_real']:.3f} | "
                         f"{r['if_real']:.3f} | {r['ens_real']:.3f} | {r['ens_aug']:.3f} |")
    lines += ["", "## Effect of the synthetic augmentation (real test packets only)", "",
              "| View | Detector | Recall (aug / real-only) | FPR (aug / real-only) | Precision (aug / real-only) |",
              "|---|---|---|---|---|"]
    views = [("all data hold-out", aug["results_aug"], aug["results_real"])]
    for env in aug["envs_real"]:
        views.append((f"same-network {env}.x.x", aug["envs_aug"][env]["results"],
                      aug["envs_real"][env]["results"]))
    for view, a_res, r_res in views:
        for name in a_res:
            a, r = a_res[name], r_res[name]
            lines.append(f"| {view} | {name} | {a['recall']:.3f} / {r['recall']:.3f} | "
                         f"{a['false_positive_rate']:.3f} / {r['false_positive_rate']:.3f} | "
                         f"{a['precision']:.3f} / {r['precision']:.3f} |")
    lines += ["", "## Leave-one-capture-out", "",
              "Fraction of the held-out capture flagged as attack "
              "(normal capture = false-positive rate, attack capture = recall). "
              "'Ens + synth' = same ensemble trained with synthetic rows added.", "",
              "| Capture | Network | Class | Packets | NB | IF | Ensemble | Ens + synth | Note |",
              "|---|---|---|---|---|---|---|---|---|"]
    env_of = df.groupby("capture")["environment"].first()
    for r in loco:
        cls = "attack" if r["label"] else "normal"
        g = lambda k: f"{r[k]:.2f}" if k in r else "-"
        lines.append(f"| {r['capture']} | {env_of[r['capture']]} | {cls} | {r['packets']} | {g('nb_flagged')} | "
                     f"{g('if_flagged')} | {g('ens_flagged')} | {g('ens_flagged_aug')} | {r.get('note', '')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path, results, y, p):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib not installed - skipping evaluation.png")
        return
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (name, m) in zip(axes[:3], results.items()):
        cm = np.array(m["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0, 1], ["Normal", "Attack"])
        ax.set_yticks([0, 1], ["Normal", "Attack"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax = axes[3]
    for key, label in (("nb_score", "Naive Bayes"), ("if_score", "Isolation Forest")):
        fpr, tpr, _ = roc_curve(y, p[key])
        ax.plot(fpr, tpr, label=f"{label} (AUC {roc_auc_score(y, p[key]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (hold-out)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    DATASET_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Extracting features from PCAP files...")
    df = build_dataset()
    df = df.sort_values(["capture", "pkt_time"], kind="stable").reset_index(drop=True)
    df = tag_environment(df)
    print("\nNetwork per capture:")
    for cap, env in df.groupby("capture")["environment"].first().items():
        print(f"  {cap:42s} {env}.x.x")

    # Dataset exactly as extracted (human-readable) and the model input table.
    df.to_csv(DATASET_DIR / "arp_dataset_raw.csv", index=False)
    df[["capture"] + FEATURES + ["label"]].to_csv(
        DATASET_DIR / "arp_dataset_features.csv", index=False)
    n_norm = int((df.label == 0).sum())
    print(f"\n[INFO] {len(df)} REAL rows ({n_norm} normal / {len(df) - n_norm} attack) "
          f"saved to dataset/")

    # Synthetic (SMOTE-style) rows that balance the classes - kept in separate files and
    # flagged synthetic=1 so they can never be mistaken for recorded traffic.
    syn_all = make_synthetic(df, RANDOM_STATE)
    real_out = df[["capture"] + FEATURES + ["label"]].assign(synthetic=0)
    syn_all[SYNTH_COLUMNS].to_csv(DATASET_DIR / "arp_dataset_synthetic.csv", index=False)
    balanced = pd.concat([real_out[SYNTH_COLUMNS], syn_all[SYNTH_COLUMNS]], ignore_index=True)
    balanced.to_csv(DATASET_DIR / "arp_dataset_balanced.csv", index=False)
    print(f"[INFO] +{len(syn_all)} SYNTHETIC rows -> balanced file: {len(balanced)} rows "
          f"({int((balanced.label == 0).sum())} normal / {int((balanced.label == 1).sum())} attack)")
    for cap, n in syn_all.groupby("capture").size().items():
        print(f"         {cap}: {n}")

    hold = build_dataset(PROJECT_ROOT / "pcap_files" / "holdout", required=False)
    if len(hold):
        hold.to_csv(DATASET_DIR / "arp_dataset_holdout.csv", index=False)
        print(f"[INFO] {len(hold)} REAL held-out rows (never trained on) saved to "
              f"dataset/arp_dataset_holdout.csv")
    print(f"[INFO] TOTAL REAL records recorded: {len(df) + len(hold)} "
          f"(training pool {len(df)} + held-out {len(hold)})")

    train_df, test_df = chronological_split(df)
    print(f"[INFO] chronological split: {len(train_df)} train / {len(test_df)} test "
          f"(test: {int((test_df.label == 0).sum())} normal, "
          f"{int((test_df.label == 1).sum())} attack)")

    # Both variants are always evaluated (ablation); AUGMENT only picks which one is shipped.
    nb0, iso0 = fit_models(train_df, False)          # real packets only
    nb1, iso1 = fit_models(train_df, True)           # + synthetic minority rows (NB only)
    if iso0 is None:
        sys.exit("No normal packets in the training split.")
    results_real, y_test, preds_real = holdout_eval(train_df, test_df, nb0, iso0)
    results_aug, _, preds_aug = holdout_eval(train_df, test_df, nb1, iso1)
    envs_real = per_environment_eval(df, False)
    envs_aug = per_environment_eval(df, True)
    loco_real = leave_one_capture_out(df, False)
    loco_aug = {r["capture"]: r for r in leave_one_capture_out(df, True)}
    for r in loco_real:
        v = loco_aug[r["capture"]].get("ens_flagged")
        if v is not None:
            r["ens_flagged_aug"] = v

    if AUGMENT:
        nb, iso, results, preds, envs = nb1, iso1, results_aug, preds_aug, envs_aug
        loco = []
        for r in loco_real:
            row = dict(loco_aug[r["capture"]])
            if "ens_flagged" in row:
                row["ens_flagged_aug"] = row["ens_flagged"]
            loco.append(row)
    else:
        nb, iso, results, preds, envs, loco = nb0, iso0, results_real, preds_real, envs_real, loco_real
    print_holdout(results)
    print_environment(envs)
    print_loco(loco)
    print_ablation(results_aug, results_real, "All-data hold-out")
    for env in envs_real:
        print_ablation(envs_aug[env]["results"], envs_real[env]["results"], f"Same-network {env}.x.x")
    unseen = []
    if len(hold):
        unseen = evaluate_unseen(hold, {"aug": (nb1, iso1), "real": (nb0, iso0)})
        print_unseen(unseen)
    aug = {"enabled": AUGMENT, "n_syn": int(len(syn_all)), "results_real": results_real,
           "results_aug": results_aug, "envs_real": envs_real, "envs_aug": envs_aug,
           "unseen": unseen}

    metrics = {"holdout": results, "same_network_holdout": envs,
               "leave_one_capture_out": loco,
               "ablation": {"real_only": {"holdout": results_real, "same_network_holdout": envs_real},
                            "augmented": {"holdout": results_aug, "same_network_holdout": envs_aug}},
               "unseen_holdout": unseen,
               "synthetic_rows": int(len(syn_all))}

    # ---- experiment tracking: data hashes + settings + code version, then a traceable model record
    manifest = data_manifest([PROJECT_ROOT / "pcap_files", PROJECT_ROOT / "pcap_files" / "holdout"])
    ens = results["Ensemble (NB OR IF)"]
    headline = {
        "all_data_holdout": {"nb_recall": results["Naive Bayes"]["recall"],
                             "nb_fpr": results["Naive Bayes"]["false_positive_rate"],
                             "ensemble_recall": ens["recall"], "ensemble_fpr": ens["false_positive_rate"]},
        "unseen_capture_ensemble_fpr": unseen[0]["ens_real" if not AUGMENT else "ens_aug"] if unseen else None,
    }
    run_id, run_dir, run_info = write_run(RESULTS_DIR, CFG, metrics, manifest, headline)

    joblib.dump(nb, MODEL_DIR / "naive_bayes_model.pkl")
    joblib.dump(iso, MODEL_DIR / "isolation_forest_model.pkl")
    (PROJECT_ROOT / "models" / "metadata.json").write_text(json.dumps({
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
        "if_contamination": IF_CONTAMINATION,
        "config": CFG,
        "train_captures": sorted(df.capture.unique().tolist()),
        "synthetic_augmentation": AUGMENT,
        "synthetic_rows_in_dataset_files": int(len(syn_all)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "git_commit": run_info["git"]["commit"],
        "data_hash": run_info["data_hash"],
        "headline_metrics": headline,
    }, indent=2), encoding="utf-8")

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_markdown(RESULTS_DIR / "metrics.md", df, train_df, test_df, results, loco, envs, aug)
    write_plot(RESULTS_DIR / "evaluation.png", results, y_test, preds)
    print(f"\nTraining complete. Models -> models/, report -> results/, run record -> results/runs/{run_id}/")


if __name__ == "__main__":
    main()
