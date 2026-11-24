"""Shared fixtures: small deterministic traffic built with the virtual-LAN simulator (no pcap files needed)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model_scripts"))

from simulate import LiveLAN  # noqa: E402


def build_lan(seed, attacks=(), seconds=40, attack_start=15):
    """Run a virtual LAN for `seconds`; start the given attacks at `attack_start`."""
    lan = LiveLAN(normal_rate=3.0, n_hosts=6, seed=seed)
    for t in range(seconds):
        if t == attack_start:
            for a in attacks:
                lan.start(a)
        lan.advance(1)
    return lan


def capture_frame(lan, name, label):
    df = lan.frame().copy()
    df["capture"] = name
    df["label"] = label            # file-level label, as in the real captures
    return df


@pytest.fixture(scope="session")
def pool():
    """Training-pool style dataframe: two normal and two attack captures, unique index."""
    parts = [
        capture_frame(build_lan(1), "normal_a", 0),
        capture_frame(build_lan(2), "normal_b", 0),
        capture_frame(build_lan(3, ["basic", "flooding"], seconds=30, attack_start=10), "attack_a", 1),
        capture_frame(build_lan(4, ["gratuitous", "unicast"], seconds=60, attack_start=10), "attack_b", 1),
    ]
    return pd.concat(parts, ignore_index=True)
