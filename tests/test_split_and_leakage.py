"""The evaluation split must not leak, and synthetic rows must stay out of every test set."""
import pandas as pd
import pytest

from augment import make_synthetic
from features import FEATURES, PROJECT_ROOT
from train import chronological_split


def test_chronological_split_respects_time(pool):
    train, test = chronological_split(pool)
    assert set(train.index).isdisjoint(test.index)
    assert len(train) + len(test) == len(pool)
    for cap, grp in pool.groupby("capture"):
        tr, te = train[train.capture == cap], test[test.capture == cap]
        assert len(tr) and len(te)
        assert tr["pkt_time"].max() <= te["pkt_time"].min(), f"{cap}: a training packet is later than a test packet"


def test_synthetic_rows_are_labelled_and_built_from_training_only(pool):
    train, test = chronological_split(pool)
    syn = make_synthetic(train, seed=42)
    assert len(syn) > 0
    assert (syn["synthetic"] == 1).all()
    assert syn["capture"].str.startswith("synthetic<-").all()
    sources = {c.replace("synthetic<-", "") for c in syn["capture"].unique()}
    assert sources <= set(train["capture"].unique())          # generated from training captures
    assert "synthetic" not in test.columns                    # nothing synthetic in the test split
    # interpolation can never leave the range spanned by the real training rows it came from
    for src in sources:
        real = train[train.capture == src][FEATURES]
        fake = syn[syn.capture == "synthetic<-" + src][FEATURES]
        assert (fake.min() >= real.min() - 1e-9).all() and (fake.max() <= real.max() + 1e-9).all()


def test_synthetic_rows_are_not_in_the_real_dataset_files():
    real = PROJECT_ROOT / "dataset" / "arp_dataset_features.csv"
    synth = PROJECT_ROOT / "dataset" / "arp_dataset_synthetic.csv"
    if not (real.exists() and synth.exists()):
        pytest.skip("dataset files not generated yet (run model_scripts/train.py)")
    assert "synthetic" not in pd.read_csv(real, nrows=1).columns
    assert (pd.read_csv(synth)["synthetic"] == 1).all()
