"""When does a detector raise an ALERT? (used by the simulator's status cards)

A per-packet verdict is not an alert. On a busy network even a good detector flags the odd normal packet, and a busy
host (for example one that scans the subnet and sends hundreds of requests) can collect a few flags. An attacker,
however, keeps sending forged packets, so most of its recent packets are flagged. A detector therefore raises an ALERT
only for a SENDER that has at least `min_flags` flagged packets AND more than `min_share` of its packets in the window
flagged. Anything flagged that does not look like that is reported as "stray" (noise).
"""
ALERT_WINDOW_S = 10      # look-back window, seconds of network time
ALERT_MIN_FLAGS = 3      # flagged packets from one sender
ALERT_MIN_SHARE = 0.5    # share of that sender's packets in the window that are flagged


def alert_state(eth_src, flagged, in_window, min_flags=ALERT_MIN_FLAGS, min_share=ALERT_MIN_SHARE):
    """Return "quiet", "stray" or "alert".

    eth_src   - pandas Series: Ethernet sender of every packet
    flagged   - boolean array: the detector flagged this packet
    in_window - boolean array: the packet is inside the look-back window
    """
    hit = flagged & in_window
    if not hit.any():
        return "quiet"
    flags_per_sender = eth_src[hit].value_counts()
    packets_per_sender = eth_src[in_window].value_counts().reindex(flags_per_sender.index)
    share = flags_per_sender / packets_per_sender
    return "alert" if ((flags_per_sender >= min_flags) & (share >= min_share)).any() else "stray"
