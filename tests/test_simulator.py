"""The simulator's ground truth must be exact: only injected packets are labelled attack, and only while the attack runs."""
from simulate import LiveLAN


def test_no_attack_means_no_attack_packets():
    lan = LiveLAN(seed=1)
    for _ in range(20):
        lan.advance(1)
    df = lan.frame()
    assert len(df) == len(lan.packets) > 0
    assert df["is_attack"].sum() == 0


def test_attack_labels_follow_the_buttons():
    lan = LiveLAN(seed=2)
    for _ in range(10):
        lan.advance(1)
    start = lan.t
    lan.start("flooding")
    for _ in range(5):
        lan.advance(1)
    lan.stop("flooding")
    attack_before = int(lan.frame()["is_attack"].sum())
    for _ in range(5):
        lan.advance(1)
    df = lan.frame()
    assert attack_before > 0
    assert int(df["is_attack"].sum()) == attack_before            # nothing injected after Stop
    atk = df[df["is_attack"] == 1]
    assert (atk["attack_name"] == "flooding").all()
    assert atk["time_s"].min() >= start - 1e-6                    # nothing injected before Launch
    assert len(df) == len(lan.packets)                            # ground truth aligned with the packets
