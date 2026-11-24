"""Single source of truth for training settings (read from config.json at the project root).

`train.py`, `compare_models.py`, `tune.py` and the simulator app all read these values, so a setting is
defined in exactly one place. `tune.py` rewrites the tuned entries and records how they were chosen.
"""
import json

from features import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "config.json"

DEFAULTS = {
    "seed": 42,
    "train_fraction": 0.70,        # chronological split: first 70% of each capture trains
    "if_contamination": 0.02,      # Isolation Forest: expected share of outliers in normal traffic
    "nb_var_smoothing": 1e-9,      # Gaussian Naive Bayes variance floor
    "rf_n_estimators": 200,
    "rf_min_samples_leaf": 1,
    "lr_C": 1.0,                   # Logistic Regression inverse regularisation strength
    "tuning": None,                # filled in by tune.py (method, date, scores)
}


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return cfg


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
