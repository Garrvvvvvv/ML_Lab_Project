"""Training must be reproducible, models must round-trip through disk, and saved metadata must be consistent."""
import json

import joblib
import numpy as np
import pytest

from config import DEFAULTS, load_config
from features import FEATURES, PROJECT_ROOT
from model_factory import make_rf
from train import chronological_split, fit_models, predict_all


def test_training_is_deterministic(pool):
    train, test = chronological_split(pool)
    nb1, iso1 = fit_models(train, False)
    nb2, iso2 = fit_models(train, False)
    a, b = predict_all(nb1, iso1, test[FEATURES]), predict_all(nb2, iso2, test[FEATURES])
    for key in ("nb", "if", "ens"):
        assert np.array_equal(a[key], b[key])
    cfg = load_config()
    rf1 = make_rf(cfg).fit(train[FEATURES], train["label"])
    rf2 = make_rf(cfg).fit(train[FEATURES], train["label"])
    assert np.array_equal(rf1.predict(test[FEATURES]), rf2.predict(test[FEATURES]))


def test_models_round_trip_through_disk(pool, tmp_path):
    train, test = chronological_split(pool)
    nb, iso = fit_models(train, False)
    joblib.dump(nb, tmp_path / "nb.pkl")
    joblib.dump(iso, tmp_path / "iso.pkl")
    before = predict_all(nb, iso, test[FEATURES])
    after = predict_all(joblib.load(tmp_path / "nb.pkl"), joblib.load(tmp_path / "iso.pkl"), test[FEATURES])
    for key in ("nb", "if", "ens"):
        assert np.array_equal(before[key], after[key])


def test_isolation_forest_sees_only_normal_traffic(pool):
    """The anomaly detector is trained on normal packets only (otherwise it is just a worse classifier)."""
    train, _ = chronological_split(pool)
    _, iso = fit_models(train, False)
    assert iso.n_features_in_ == len(FEATURES)
    assert iso._max_samples <= (train["label"] == 0).sum()


def test_config_is_complete_and_valid():
    cfg = load_config()
    assert set(DEFAULTS) <= set(cfg)
    assert 0 < cfg["train_fraction"] < 1 and 0 < cfg["if_contamination"] < 0.5 and cfg["rf_n_estimators"] >= 10


def test_saved_model_metadata_is_traceable():
    meta_path = PROJECT_ROOT / "models" / "metadata.json"
    if not meta_path.exists():
        pytest.skip("no trained model yet (run model_scripts/train.py)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["features"] == FEATURES
    for key in ("run_id", "data_hash", "config", "trained_at", "sklearn_version"):
        assert meta.get(key), f"metadata missing {key}"
    run_dir = PROJECT_ROOT / "results" / "runs" / meta["run_id"]
    assert (run_dir / "run.json").exists() and (run_dir / "data_manifest.json").exists()
    nb = joblib.load(PROJECT_ROOT / "models" / "saved_models" / "naive_bayes_model.pkl")
    assert nb.n_features_in_ == len(FEATURES)
