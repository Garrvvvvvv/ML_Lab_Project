"""Run the NB + Isolation Forest ensemble on any capture.

    python model_scripts/predict.py path/to/capture.pcapng
    python model_scripts/predict.py capture.pcapng --label 0     # you know it is normal
    python model_scripts/predict.py capture.pcapng --label 1     # you know it is an attack

For an honest demo use a capture that was NOT used in training - e.g. drop it in
pcap_files/holdout/ (train.py only reads the top level of pcap_files/).
No label encoders are needed any more, so unseen IPs/MACs cannot crash inference.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn

from features import FEATURES, PROJECT_ROOT, extract_arp_features

MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_OUT = PROJECT_ROOT / "results" / "prediction_output.csv"


def load_models():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    if meta["features"] != FEATURES:
        sys.exit("Feature list in models/metadata.json differs from features.py - "
                 "re-run model_scripts/train.py.")
    if meta["sklearn_version"] != sklearn.__version__:
        print(f"[WARN] models trained with scikit-learn {meta['sklearn_version']}, "
              f"running {sklearn.__version__}; re-run train.py if loading fails.")
    nb = joblib.load(MODEL_DIR / "saved_models" / "naive_bayes_model.pkl")
    iso = joblib.load(MODEL_DIR / "saved_models" / "isolation_forest_model.pkl")
    return nb, iso


def predict_with_ensemble(pcap_path, out_path=DEFAULT_OUT, true_label=None):
    df = extract_arp_features(pcap_path)
    if df.empty:
        print("No ARP packets found in the provided file.")
        return None

    nb, iso = load_models()
    X = df[FEATURES]
    df["nb_prediction"] = nb.predict(X)
    df["if_prediction"] = (iso.predict(X) == -1).astype(int)       # -1 = anomaly
    df["final_prediction"] = np.maximum(df["nb_prediction"], df["if_prediction"])  # OR

    n = len(df)
    print(f"\nFile: {Path(pcap_path).name}  ({n} ARP packets)")
    print(f"{'detector':22s}{'flagged':>9s}{'share':>8s}")
    for col, name in (("nb_prediction", "Naive Bayes"),
                      ("if_prediction", "Isolation Forest"),
                      ("final_prediction", "Ensemble (OR)")):
        k = int(df[col].sum())
        print(f"{name:22s}{k:9d}{k / n:8.1%}")

    if true_label is not None:
        what = "false-positive rate" if true_label == 0 else "detection rate (recall)"
        print(f"\nKnown label = {true_label} -> ensemble {what}: "
              f"{df['final_prediction'].mean():.1%}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nDetailed per-packet output saved to {out_path}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pcap", help="capture file (.pcap / .pcapng) to analyse")
    ap.add_argument("--label", type=int, choices=[0, 1],
                    help="ground truth for the whole capture, to print FPR / recall")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output CSV path")
    args = ap.parse_args()
    if not Path(args.pcap).is_file():
        sys.exit(f"File not found: {args.pcap}")
    predict_with_ensemble(args.pcap, args.out, args.label)


if __name__ == "__main__":
    main()
