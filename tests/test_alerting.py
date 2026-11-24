"""The alert rule must stay quiet on noise and on busy hosts, and fire for an attacker-like sender."""
import numpy as np
import pandas as pd

from alerting import alert_state


def _window(senders, flagged_idx, in_window=None):
    eth = pd.Series(senders)
    flagged = np.zeros(len(eth), dtype=bool)
    flagged[list(flagged_idx)] = True
    return eth, flagged, np.ones(len(eth), dtype=bool) if in_window is None else np.asarray(in_window)


def test_nothing_flagged_is_quiet():
    assert alert_state(*_window(["a"] * 20, [])) == "quiet"


def test_attacker_whose_packets_are_all_flagged_alerts():
    senders = ["attacker"] * 6 + ["h1", "h2", "h3", "h4"]
    assert alert_state(*_window(senders, range(6))) == "alert"


def test_isolated_flags_on_different_hosts_are_only_stray():
    senders = [f"h{i}" for i in range(50)]
    assert alert_state(*_window(senders, [3, 17, 29, 41])) == "stray"      # 4 flags, but one per host


def test_busy_scanner_with_a_few_flags_is_stray_not_an_alert():
    senders = ["scanner"] * 400 + ["h1"] * 10                              # 6 of 400 packets flagged = 1.5%
    assert alert_state(*_window(senders, range(6))) == "stray"


def test_too_few_flags_from_one_sender_do_not_alert():
    assert alert_state(*_window(["attacker"] * 2 + ["h1"] * 5, [0, 1])) == "stray"


def test_flags_outside_the_window_are_ignored():
    senders = ["attacker"] * 10
    in_window = [False] * 7 + [True] * 3
    assert alert_state(*_window(senders, range(7), in_window)) == "quiet"  # all flags are older than the window


def test_thresholds_are_adjustable():
    senders = ["a"] * 4 + ["b"] * 4
    args = _window(senders, [0, 1])
    assert alert_state(*args) == "stray"
    assert alert_state(*args, min_flags=2, min_share=0.5) == "alert"
