"""The live stream must give the same features as batch processing, cost O(new packets) per update, and
recognise a network that is busier than the training data."""
import time

import numpy as np
import pandas as pd
import pytest

from conftest import build_lan
from features import FeatureExtractor, PROJECT_ROOT, extract_from_packets
from simulate import HybridLAN, LiveLAN, packets_per_second, training_rate_limit


def test_incremental_extraction_equals_batch():
    """Feeding packets in many small chunks gives exactly the rows of one big call (training vs live can't disagree)."""
    packets = build_lan(7, ["basic", "flooding", "race"]).packets
    batch = extract_from_packets(packets, "x")
    extractor = FeatureExtractor("x")
    chunks = [extractor.process(packets[i:i + 37]) for i in range(0, len(packets), 37)]
    pd.testing.assert_frame_equal(batch, pd.concat(chunks, ignore_index=True))


def test_live_frame_grows_incrementally_and_matches_batch():
    lan = LiveLAN(seed=5, keep_packets=True)
    sizes = []
    for t in range(30):
        if t == 10:
            lan.start("basic")
        lan.advance(1)
        sizes.append(len(lan.frame()))            # frame() called every tick, like the app does
    assert sizes == sorted(sizes) and sizes[-1] == lan.n_packets > 0
    batch = extract_from_packets(lan.packets, "live")
    pd.testing.assert_frame_equal(lan.frame()[batch.columns].reset_index(drop=True), batch)


def test_live_session_does_not_hold_raw_packets_by_default():
    lan = LiveLAN(seed=5, keep_packets=False)
    for _ in range(20):
        lan.advance(1)
    assert lan.packets == []                       # memory stays flat on a long, busy session
    assert len(lan.frame()) == lan.n_packets > 0


class _FakeCapture:
    """Stands in for the real sniffer: hands out recorded packets in batches."""

    def __init__(self, packets):
        self.queue, self.batch = list(packets), 200

    def drain(self):
        out, self.queue = self.queue[: self.batch], self.queue[self.batch:]
        return out


def test_hybrid_stream_is_incremental_and_keeps_no_raw_packets():
    source = LiveLAN(seed=3, keep_packets=True)
    for _ in range(40):
        source.advance(1)
    lan = HybridLAN(_FakeCapture([]))               # keep_packets defaults to False
    lan.t0 -= 60                                    # pretend the live session started a minute ago
    packets = source.packets
    for i, p in enumerate(packets):                 # timestamps inside that minute, so every packet is already due
        p.time = lan.t0 + 0.001 * i
    lan.capture = _FakeCapture(packets)
    for _ in range(len(packets) // 200 + 3):
        lan.advance()
    assert lan.real_count == len(packets) and lan.packets == []
    assert len(lan.frame()) == lan.n_packets == len(packets)


def test_packets_per_second_uses_only_the_recent_window():
    times = np.arange(0, 20, 0.1)                   # 10 packets/s for 20 s
    assert packets_per_second(times, now=20, window=5) == pytest.approx(10, abs=0.3)
    assert packets_per_second(times, now=20, window=5) > packets_per_second(times[:100], now=20, window=5)


def test_training_rate_limit_matches_known_counts():
    t = np.concatenate([np.full(5, s) + np.linspace(0, 0.9, 5) for s in range(100)])   # exactly 5 pkts/s
    burst = np.concatenate([np.full(50, 100.0) + np.linspace(0, 0.9, 50)])              # one 50-packet second
    raw = pd.DataFrame({"capture": "normal", "label": 0, "pkt_time": np.concatenate([t, burst])})
    raw = pd.concat([raw, pd.DataFrame({"capture": "attack", "label": 1, "pkt_time": np.arange(0, 100, 0.01)})])
    limit = training_rate_limit(raw, quantile=1.0)
    assert limit == 50                              # attack captures are ignored; the busiest NORMAL second wins


def test_real_training_traffic_defines_a_sensible_busy_threshold():
    raw_path = PROJECT_ROOT / "dataset" / "arp_dataset_raw.csv"
    if not raw_path.exists():
        pytest.skip("dataset not generated yet (run model_scripts/train.py)")
    limit = training_rate_limit(pd.read_csv(raw_path, usecols=["capture", "pkt_time", "label"]))
    assert 5 < limit < 100                          # normal traffic in training is a few packets/s, never hundreds
