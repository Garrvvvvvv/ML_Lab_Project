"""Shared ARP feature extraction used by BOTH train.py and predict.py.

Keeping this in one place guarantees that training and inference build
features identically (previously the logic was copy-pasted into both scripts).

Design notes
------------
* Features are computed *per capture file*: all state (IP->MAC maps, reply
  counters, rate windows, inter-arrival gaps) is reset at the start of every
  pcap, so gaps between captures recorded days apart never leak in as a feature.
* No raw identity features (IP/MAC strings) are given to the models. Identities
  are kept in the raw CSV for readability only. Feeding label-encoded MACs/IPs
  let the model memorise *which capture file* a row came from, and made
  prediction crash on any previously unseen address.
* State is only ever updated from past packets, so a feature never looks at the
  future of the stream (important for the chronological train/test split).
"""
import math
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
from scapy.all import ARP, Ether, rdpcap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PCAP_DIR = PROJECT_ROOT / "pcap_files"

BROADCAST_MAC = "ff:ff:ff:ff:ff:ff"
ZERO_IP = "0.0.0.0"          # source IP used by ARP probes (RFC 5227)
RATE_WINDOW_S = 1.0          # window for the per-sender reply-rate feature

# Columns fed to the models (no identity strings).
FEATURES = [
    "op",                    # 1 = request, 2 = reply
    "is_broadcast",          # Ethernet destination is ff:ff:ff:ff:ff:ff
    "is_gratuitous",         # sender IP == target IP (announcement)
    "mac_conflict",          # this IP was earlier seen with a different MAC
    "ip_mac_count",          # distinct MACs ever claimed for this sender IP
    "l2_ip_count",           # distinct IPs claimed by this Ethernet sender
    "l2_reply_rate_1s",      # replies from this Ethernet sender in last 1 s
    "arp_reply_count",       # running count of replies sent by this IP
    "log_interarrival_ms",   # log1p(gap to previous packet in same capture, ms)
    "log_reply_delay_ms",    # log1p(request->reply delay, ms); 0 if unmatched
]

# Computed and stored in the raw CSV for analysis, but deliberately NOT model inputs:
#  * is_unsolicited_reply  - depends on the capture viewpoint (how many replies addressed to
#    *other* hosts the sniffer happens to see: 0% in one normal capture, 53-64% in others).
#  * eth_arp_mac_mismatch  - in the attack captures it is an artifact of the attack tool
#    (forged frames carry Ethernet src 00:00:00:00:00:00); removing it did not change recall.
DIAGNOSTIC_COLUMNS = ["is_unsolicited_reply", "eth_arp_mac_mismatch"]

# Human-readable columns kept in the raw CSV only.
IDENTITY_COLUMNS = ["src_ip", "dst_ip", "src_mac", "dst_mac", "eth_src", "eth_dst"]


def label_for(filename):
    """Capture label by naming convention: names containing 'normal' -> 0, else 1."""
    return 0 if "normal" in Path(filename).name.lower() else 1


def discover_pcaps(pcap_dir=PCAP_DIR):
    """Return [(path, label)] for every .pcap/.pcapng directly inside pcap_dir.

    Sub-folders (e.g. pcap_files/holdout/) are deliberately NOT scanned, so
    captures kept there stay unseen by training and can be used as a demo for
    predict.py.
    """
    files = sorted(
        p for p in Path(pcap_dir).glob("*")
        if p.suffix.lower() in (".pcap", ".pcapng")
    )
    return [(p, label_for(p)) for p in files]


def extract_arp_features(file_path):
    """Parse one capture and return a DataFrame with one row per ARP packet."""
    packets = sorted(rdpcap(str(file_path)), key=lambda p: float(p.time))  # stable
    return extract_from_packets(packets, Path(file_path).name)


def extract_from_packets(packets, capture_name):
    """Feature rows for already-loaded Scapy packets (sorted by time). Same logic as for pcap files."""
    return FeatureExtractor(capture_name).process(packets)


class FeatureExtractor:
    """Stateful ARP feature extractor.

    Feed packets in time order with `process()`; each call returns the rows for just those packets and keeps
    the per-capture state (IP<->MAC maps, counters, rate windows) for the next call. A whole capture in one
    call gives exactly the same rows as the same packets fed in many small calls, so training (one big call)
    and the live simulator (one small call per second) can never disagree - and a live session costs
    O(new packets) per update instead of re-reading everything.
    """

    def __init__(self, capture_name):
        self.capture_name = capture_name
        self.ip_macs = defaultdict(set)         # sender IP  -> set of ARP sender MACs seen
        self.l2_ips = defaultdict(set)          # Ethernet src -> set of sender IPs claimed
        self.last_mac_for_ip = {}
        self.pending_requests = {}              # (target_ip, requester_ip) -> request time
        self.reply_counter = defaultdict(int)
        self.reply_times = defaultdict(deque)   # Ethernet src -> recent reply timestamps
        self.prev_time = None

    def process(self, packets):
        rows = [r for r in (self._row(pkt) for pkt in packets) if r is not None]
        return pd.DataFrame(rows)

    def _row(self, pkt):
        if ARP not in pkt:
            return None
        arp = pkt[ARP]
        t = float(pkt.time)

        src_ip, dst_ip = arp.psrc, arp.pdst
        src_mac, dst_mac = arp.hwsrc.lower(), arp.hwdst.lower()
        op = int(arp.op)
        if Ether in pkt:
            eth_src, eth_dst = pkt[Ether].src.lower(), pkt[Ether].dst.lower()
        else:  # e.g. Linux "cooked" captures without an Ethernet header
            eth_src, eth_dst = src_mac, dst_mac

        is_probe = src_ip == ZERO_IP
        is_broadcast = int(eth_dst == BROADCAST_MAC)
        is_gratuitous = int((not is_probe) and src_ip == dst_ip)
        mismatch = int(eth_src != src_mac)

        # --- request/reply matching and delay -------------------------------
        delay_ms = None
        unsolicited = 0
        if op == 1:
            self.pending_requests[(dst_ip, src_ip)] = t
        elif op == 2:
            self.reply_counter[src_ip] += 1
            key = (src_ip, dst_ip)
            if key in self.pending_requests:
                delay_ms = (t - self.pending_requests.pop(key)) * 1000.0
            else:
                unsolicited = 1

        # --- IP <-> MAC consistency (probes carry no real sender IP) --------
        mac_conflict = 0
        if not is_probe:
            if src_ip in self.last_mac_for_ip and self.last_mac_for_ip[src_ip] != src_mac:
                mac_conflict = 1
            self.last_mac_for_ip[src_ip] = src_mac
            self.ip_macs[src_ip].add(src_mac)
            self.l2_ips[eth_src].add(src_ip)
        ip_mac_count = len(self.ip_macs[src_ip]) if not is_probe else 1
        l2_ip_count = len(self.l2_ips[eth_src]) if self.l2_ips[eth_src] else 1

        # --- reply rate of this Ethernet sender over the last second --------
        window = self.reply_times[eth_src]
        if op == 2:
            window.append(t)
        while window and t - window[0] > RATE_WINDOW_S:
            window.popleft()

        gap_s = 0.0 if self.prev_time is None else max(t - self.prev_time, 0.0)
        self.prev_time = t

        return {
            "capture": self.capture_name,
            "pkt_time": t,
            "src_ip": src_ip, "dst_ip": dst_ip,
            "src_mac": src_mac, "dst_mac": dst_mac,
            "eth_src": eth_src, "eth_dst": eth_dst,
            "op": op,
            "is_broadcast": is_broadcast,
            "is_gratuitous": is_gratuitous,
            "is_unsolicited_reply": unsolicited,
            "eth_arp_mac_mismatch": mismatch,
            "mac_conflict": mac_conflict,
            "ip_mac_count": ip_mac_count,
            "l2_ip_count": l2_ip_count,
            "l2_reply_rate_1s": len(window),
            "arp_reply_count": self.reply_counter[src_ip],
            "interarrival_s": gap_s,
            "reply_delay_ms": -1.0 if delay_ms is None else delay_ms,
            "log_interarrival_ms": math.log1p(gap_s * 1000.0),
            "log_reply_delay_ms": 0.0 if delay_ms is None else math.log1p(max(delay_ms, 0.0)),
        }
