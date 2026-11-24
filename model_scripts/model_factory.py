"""Builds each model from the settings in config.json (one place, used by training, tuning and the app)."""
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_nb(cfg):
    return GaussianNB(var_smoothing=cfg["nb_var_smoothing"])


def make_iso(cfg):
    return IsolationForest(contamination=cfg["if_contamination"], random_state=cfg["seed"])


def make_lr(cfg):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=cfg["lr_C"], max_iter=2000, class_weight="balanced", random_state=cfg["seed"]))


def make_rf(cfg):
    return RandomForestClassifier(
        n_estimators=cfg["rf_n_estimators"], min_samples_leaf=cfg["rf_min_samples_leaf"],
        class_weight="balanced", random_state=cfg["seed"], n_jobs=-1)
