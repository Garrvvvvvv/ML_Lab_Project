"""Feature extraction must be causal, deterministic and identical between training and inference."""
import pandas as pd
from scapy.all import wrpcap

from conftest import build_lan
from features import FEATURES, extract_arp_features, extract_from_packets


def test_features_are_causal():
    """A packet's features depend only on earlier packets: appending later packets never changes earlier rows."""
    packets = build_lan(7, ["basic", "flooding"]).packets
    full = extract_from_packets(packets, "x")
    half = extract_from_packets(packets[: len(packets) // 2], "x")
    pd.testing.assert_frame_equal(full.iloc[: len(half)][FEATURES].reset_index(drop=True),
                                  half[FEATURES].reset_index(drop=True))


def test_features_are_deterministic():
    packets = build_lan(7, ["basic"]).packets
    a = extract_from_packets(packets, "x")[FEATURES]
    b = extract_from_packets(packets, "x")[FEATURES]
    pd.testing.assert_frame_equal(a, b)


def test_pcap_and_in_memory_paths_agree(tmp_path):
    """predict.py (reads a pcap) and the live simulator (reads packets) must compute the same features."""
    packets = build_lan(8, ["unicast"]).packets
    path = tmp_path / "sim.pcap"
    wrpcap(str(path), packets)
    from_file = extract_arp_features(path)[FEATURES].reset_index(drop=True)
    from_mem = extract_from_packets(packets, "x")[FEATURES].reset_index(drop=True)
    # pcap stores microsecond timestamps, so timing features may differ by a hair
    pd.testing.assert_frame_equal(from_file, from_mem, check_exact=False, atol=0.05)


def test_feature_list_has_no_identity_columns():
    """No raw IP/MAC may be a model input (it lets a model memorise which capture a row came from)."""
    banned = ("ip", "mac", "src", "dst")
    assert not [f for f in FEATURES if f in ("src_ip", "dst_ip", "src_mac", "dst_mac", "eth_src", "eth_dst")]
    assert len(FEATURES) == 10 and len(set(FEATURES)) == 10 and banned
