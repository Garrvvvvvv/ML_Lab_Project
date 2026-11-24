"""Virtual-LAN ARP traffic simulator (for the demo web app).

EVERYTHING HERE IS SIMULATED. Packets are built in memory with Scapy, written to a temporary
.pcap and read back through the SAME feature extractor used for training (`features.py`).
Nothing is ever transmitted on any network interface.

Because we generate the traffic, per-packet ground truth is known exactly (attacker-injected
packet = 1, everything else = 0), unlike the file-level labels of the recorded captures.

Scenarios (the five attack types from the project plus a mix):
  basic      attacker sends forged replies claiming the gateway's IP (man-in-the-middle setup)
  flooding   high-rate forged replies with random IP/MAC pairs
  gratuitous broadcast "announcements" claiming the gateway's IP with the attacker's MAC
  race       attacker answers a victim's request for the gateway before the real gateway does
  unicast    forged replies sent directly to one victim (stealthier, low rate)
  mixed      all of the above
"""
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.all import ARP, Ether, wrpcap

from features import FeatureExtractor, extract_arp_features, extract_from_packets

SCENARIOS = {
    "none": "Normal traffic only (check for false alarms)",
    "basic": "Basic ARP spoofing (gateway impersonation)",
    "flooding": "ARP request/reply flooding",
    "gratuitous": "Gratuitous ARP poisoning",
    "race": "Reply race condition",
    "unicast": "Unicast reply spoofing (stealthy)",
    "mixed": "Mixed: all attacks at once",
}
BROADCAST = "ff:ff:ff:ff:ff:ff"
ZERO = "00:00:00:00:00:00"
SUBNET = "192.168.50."


def _mac(rng):
    return ":".join(f"{b:02x}" for b in [0x02, *rng.integers(0, 256, 5)])


def _hosts(rng, n):
    hosts = [(SUBNET + "1", _mac(rng))]                       # gateway
    hosts += [(SUBNET + str(10 + i), _mac(rng)) for i in range(n)]
    return hosts


def _request(t, src, dst_ip):
    return t, Ether(src=src[1], dst=BROADCAST) / ARP(op=1, hwsrc=src[1], psrc=src[0], hwdst=ZERO, pdst=dst_ip), 0


def _reply(t, src, dst, is_attack=0, claim_ip=None, claim_mac=None, eth_dst=None):
    ip = claim_ip or src[0]
    mac = claim_mac or src[1]
    return t, Ether(src=src[1], dst=eth_dst or dst[1]) / ARP(op=2, hwsrc=mac, psrc=ip, hwdst=dst[1], pdst=dst[0]), is_attack


def simulate(scenario="basic", duration_s=60, normal_rate=3.0, n_hosts=8, intensity=1.0, seed=7):
    """Return (features_df, meta). features_df has FEATURES + per-packet 'is_attack' ground truth,
    'time_s' (seconds from start), 'src_ip', 'src_mac', 'kind' and 'info' columns."""
    rng = np.random.default_rng(seed)
    hosts = _hosts(rng, n_hosts)
    gw, victims = hosts[0], hosts[1:]
    attacker = (SUBNET + "66", _mac(rng))
    t_attack = duration_s * 0.3                                # attack starts 30% in
    t_end = duration_s
    events = []                                                 # (time, packet, is_attack)

    # --- normal background: Poisson request/reply exchanges
    t = 0.0
    while True:
        t += rng.exponential(1.0 / max(normal_rate, 0.1))
        if t >= t_end:
            break
        src = victims[rng.integers(len(victims))]
        dst = hosts[rng.integers(len(hosts))] if rng.random() > 0.35 else gw
        if dst == src:
            continue
        events.append(_request(t, src, dst[0]))
        events.append(_reply(t + rng.uniform(0.0004, 0.004), dst, src))

    active = {"basic", "flooding", "gratuitous", "race", "unicast"} if scenario == "mixed" else (
        {scenario} if scenario != "none" else set())
    k = max(intensity, 0.1)

    if "basic" in active:                                       # forged gw<->victim mappings
        t = t_attack
        while t < t_end:
            v = victims[rng.integers(len(victims))]
            events.append(_reply(t, attacker, v, 1, claim_ip=gw[0]))
            events.append(_reply(t + 0.002, attacker, gw, 1, claim_ip=v[0]))
            t += rng.exponential(1.0 / (1.0 * k))
    if "flooding" in active:                                    # many random IP/MAC pairs, high rate
        t = t_attack
        while t < t_end:
            fake_ip = SUBNET + str(rng.integers(2, 254))
            events.append(_reply(t, attacker, (SUBNET + "255", BROADCAST), 1,
                                 claim_ip=fake_ip, claim_mac=_mac(rng), eth_dst=BROADCAST))
            t += rng.exponential(1.0 / (40.0 * k))
    if "gratuitous" in active:                                  # broadcast announcements for gw IP
        t = t_attack
        while t < t_end:
            events.append(_reply(t, attacker, (gw[0], BROADCAST), 1, claim_ip=gw[0], eth_dst=BROADCAST))
            t += rng.exponential(1.0 / (2.0 * k))
    if "unicast" in active:                                     # direct replies to one victim
        v = victims[0]
        t = t_attack
        while t < t_end:
            events.append(_reply(t, attacker, v, 1, claim_ip=gw[0]))
            t += rng.exponential(1.0 / (0.5 * k))
    if "race" in active:                                        # forged reply before the real one
        for ev_t, pkt, _ in [e for e in events if e[2] == 0 and e[1][ARP].op == 1
                             and e[1][ARP].pdst == gw[0] and e[0] >= t_attack]:
            if rng.random() < min(0.9, 0.5 * k):
                v = next(h for h in hosts if h[1] == pkt[ARP].hwsrc)
                events.append(_reply(ev_t + 0.0002, attacker, v, 1, claim_ip=gw[0]))

    events.sort(key=lambda e: e[0])
    # strictly increasing timestamps keep ground truth aligned with the extractor's stable sort
    base = 1_700_000_000.0
    stamps = np.maximum.accumulate([e[0] for e in events])
    for i in range(1, len(stamps)):
        stamps[i] = max(stamps[i], stamps[i - 1] + 1e-6)
    pkts, truth = [], []
    for ts, (_, pkt, atk) in zip(stamps, events):
        pkt.time = base + float(ts)
        pkts.append(pkt)
        truth.append(atk)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "simulated.pcap"
        wrpcap(str(path), pkts)
        df = extract_arp_features(path)
    if len(df) != len(truth):
        raise RuntimeError(f"feature rows ({len(df)}) != simulated packets ({len(truth)})")
    df = df.reset_index(drop=True)
    df["is_attack"] = truth
    df["time_s"] = df["pkt_time"] - df["pkt_time"].iloc[0] if "pkt_time" in df else np.arange(len(df))
    df["kind"] = np.where(df["op"] == 1, "request", "reply")
    meta = {"scenario": scenario, "duration_s": duration_s, "attack_start_s": t_attack if active else None,
            "packets": len(df), "attack_packets": int(sum(truth)), "attacker_mac": attacker[1],
            "gateway": gw[0]}
    return df, meta


ATTACKS = {k: v for k, v in SCENARIOS.items() if k not in ("none", "mixed")}
BASE_TS = 1_700_000_000.0
MAX_PACKETS = 60_000          # a live session stops here (memory guard; ~8 min on a very busy network)


class LiveLAN:
    """A virtual LAN that runs step by step; attacks are started and stopped by the user.

    Still 100% simulated: packets exist only in memory and are never transmitted. Every packet
    remembers which attack (if any) injected it, so ground truth is exact.
    """

    def __init__(self, normal_rate=3.0, n_hosts=8, seed=7, keep_packets=True):
        self.rng = np.random.default_rng(seed)
        self.hosts = _hosts(self.rng, n_hosts)
        self.gw, self.victims = self.hosts[0], self.hosts[1:]
        self.subnet = SUBNET
        self.attacker = (SUBNET + "66", _mac(self.rng))
        self.normal_rate = normal_rate
        self.intensity = 1.0
        self.t = 0.0
        self._init_stream(keep_packets)
        self.queue = []                                   # events not yet emitted: (time, pkt, name)
        self.next_normal = self.rng.exponential(1.0 / max(normal_rate, 0.1))
        self.active = {}                                  # attack -> next event time
        self.log = []                                     # {"name", "start", "stop"}
        self._last_ts = 0.0

    def _init_stream(self, keep_packets):
        """State for the incremental packet -> feature pipeline (shared by LiveLAN and HybridLAN)."""
        self.keep_packets = keep_packets    # keep raw Scapy packets (tests / offline analysis); the app turns this off
        self.packets = []
        self.n_packets = 0
        self._extractor = FeatureExtractor("live")
        self._new, self._new_names = [], []  # emitted since the last frame()
        self._frame = pd.DataFrame()

    def _record(self, pkt, name):
        self.n_packets += 1
        self._new.append(pkt)
        self._new_names.append(name)
        if self.keep_packets:
            self.packets.append(pkt)

    # ---- user actions
    def start(self, name):
        if name in ATTACKS and name not in self.active:
            self.active[name] = self.t
            self.log.append({"name": name, "start": self.t, "stop": None})

    def stop(self, name):
        if name in self.active:
            del self.active[name]
            for e in reversed(self.log):
                if e["name"] == name and e["stop"] is None:
                    e["stop"] = self.t
                    break

    def stop_all(self):
        for n in list(self.active):
            self.stop(n)

    @property
    def full(self):
        return self.n_packets >= MAX_PACKETS

    # ---- time stepping
    def _emit_attacks(self, end):
        rng, k = self.rng, max(self.intensity, 0.1)
        rates = {"basic": 1.0, "flooding": 40.0, "gratuitous": 2.0, "unicast": 0.5}
        for name in list(self.active):
            if name == "race":
                continue                                   # reacts to requests, see _emit_normal
            t = self.active[name]
            while t < end:
                if name == "basic":
                    v = self.victims[rng.integers(len(self.victims))]
                    self.queue.append((t, _reply(t, self.attacker, v, 1, claim_ip=self.gw[0])[1], name))
                    self.queue.append((t + 0.002, _reply(t, self.attacker, self.gw, 1, claim_ip=v[0])[1], name))
                elif name == "flooding":
                    pkt = _reply(t, self.attacker, (self.subnet + "255", BROADCAST), 1,
                                 claim_ip=self.subnet + str(rng.integers(2, 254)),
                                 claim_mac=_mac(rng), eth_dst=BROADCAST)[1]
                    self.queue.append((t, pkt, name))
                elif name == "gratuitous":
                    pkt = _reply(t, self.attacker, (self.gw[0], BROADCAST), 1, claim_ip=self.gw[0], eth_dst=BROADCAST)[1]
                    self.queue.append((t, pkt, name))
                elif name == "unicast":
                    self.queue.append((t, _reply(t, self.attacker, self.victims[0], 1, claim_ip=self.gw[0])[1], name))
                t += rng.exponential(1.0 / (rates[name] * k))
            self.active[name] = t

    def _emit_normal(self, end):
        rng = self.rng
        while self.next_normal < end:
            t = self.next_normal
            self.next_normal += rng.exponential(1.0 / max(self.normal_rate, 0.1))
            src = self.victims[rng.integers(len(self.victims))]
            dst = self.hosts[rng.integers(len(self.hosts))] if rng.random() > 0.35 else self.gw
            if dst == src:
                continue
            self.queue.append((t, _request(t, src, dst[0])[1], ""))
            self.queue.append((t + rng.uniform(0.0004, 0.004), _reply(t, dst, src)[1], ""))
            if "race" in self.active and dst == self.gw and rng.random() < min(0.9, 0.5 * max(self.intensity, 0.1)):
                self.queue.append((t + 0.0002, _reply(t, self.attacker, src, 1, claim_ip=self.gw[0])[1], "race"))

    def advance(self, dt):
        """Simulate `dt` seconds of LAN time."""
        if self.full:
            return
        end = self.t + dt
        self._emit_normal(end)
        self._emit_attacks(end)
        self._flush(end)

    def _flush(self, end):
        """Move every queued event due before `end` into the packet list, in time order."""
        self.queue.sort(key=lambda e: e[0])
        due = [e for e in self.queue if e[0] < end]
        self.queue = [e for e in self.queue if e[0] >= end]
        for ev_t, pkt, name in due:
            ts = max(ev_t, self._last_ts + 1e-6)
            self._last_ts = ts
            pkt.time = BASE_TS + ts
            self._record(pkt, name)
        self.t = end

    def frame(self):
        """Feature rows (+ ground truth) for every packet so far.

        Incremental: only packets emitted since the previous call are run through the feature extractor, so the
        cost per update does not grow with the length of the session. The returned frame is shared - do not modify it.
        """
        if self._new:
            rows = self._extractor.process(self._new).reset_index(drop=True)
            rows["attack_name"] = self._new_names
            rows["is_attack"] = (rows["attack_name"] != "").astype(int)
            rows["time_s"] = rows["pkt_time"] - BASE_TS
            rows["kind"] = np.where(rows["op"] == 1, "request", "reply")
            self._frame = rows if self._frame.empty else pd.concat([self._frame, rows], ignore_index=True)
            self._new, self._new_names = [], []
        return self._frame


def packets_per_second(times, now, window=5.0):
    """Average packets per second over the last `window` seconds of LAN time."""
    t = np.asarray(times, dtype=float)
    return float(((t > now - window) & (t <= now)).sum() / window)


def training_rate_limit(raw_df, quantile=0.999):
    """Busiest NORMAL traffic seen in training: a high percentile of per-second ARP packet counts across the normal
    captures. A live network far above this is outside what the models were trained on."""
    counts = []
    for _, g in raw_df[raw_df["label"] == 0].groupby("capture"):
        secs = np.floor(g["pkt_time"] - g["pkt_time"].min()).astype(int)
        counts.append(np.bincount(secs))
    return float(np.quantile(np.concatenate(counts), quantile))


class LiveCapture:
    """Passive, read-only ARP capture from a real interface (Scapy + Npcap).

    Packets are kept in memory only - nothing is written to disk and nothing is transmitted.
    """

    def __init__(self, iface_name):
        from scapy.all import AsyncSniffer, conf
        iface = next(i for i in conf.ifaces.values() if i.name == iface_name)
        self._lock = threading.Lock()
        self._buf = []
        self.sniffer = AsyncSniffer(iface=iface, filter="arp", store=False, prn=self._on)
        self.sniffer.start()

    def _on(self, pkt):
        if ARP in pkt:
            with self._lock:
                self._buf.append(pkt)

    def drain(self):
        with self._lock:
            out, self._buf = self._buf, []
        return out

    def stop(self):
        try:
            self.sniffer.stop()
        except Exception:
            pass

    @staticmethod
    def interfaces():
        from scapy.all import conf
        return [i.name for i in conf.ifaces.values()
                if i.name and "loopback" not in (i.description or "").lower() and "Direct" not in (i.description or "")]


class HybridLAN(LiveLAN):
    """REAL live ARP traffic + attack packets injected into the in-memory stream by the user.

    Real packets are captured passively and assumed normal. Attacks are built from real hosts
    the sniffer has observed (so a forged reply genuinely contradicts a mapping the detector has
    already seen) and are added to the stream only - nothing is ever transmitted on the network.
    """

    def __init__(self, capture, seed=7, keep_packets=False):
        self.rng = np.random.default_rng(seed)
        self.capture = capture
        self.t0 = time.time()
        self.hosts, self.victims, self.gw = [], [], None
        self.subnet = SUBNET
        self.attacker = (SUBNET + "66", _mac(self.rng))
        self.intensity = 1.0
        self.t = 0.0
        self.queue = []
        self._init_stream(keep_packets)
        self.active, self.log = {}, []
        self._last_ts = 0.0
        self.sender_mac = {}                              # observed sender IP -> MAC (first seen)
        self.target_hits = {}                             # how often each IP was asked for
        self.real_count = 0

    @property
    def ready(self):
        return len(self.sender_mac) >= 3

    def _refresh_hosts(self):
        known = list(self.sender_mac.items())
        gw_ip = max((ip for ip, _ in known), key=lambda ip: self.target_hits.get(ip, 0))
        self.gw = (gw_ip, self.sender_mac[gw_ip])
        self.victims = [(ip, m) for ip, m in known if ip != gw_ip]
        self.hosts = [self.gw] + self.victims
        self.subnet = ".".join(gw_ip.split(".")[:3]) + "."
        self.attacker = (self.subnet + "66", self.attacker[1])

    def start(self, name):
        if name in ATTACKS and self.ready and name not in self.active:
            self._refresh_hosts()
            self.active[name] = self.t
            self.log.append({"name": name, "start": self.t, "stop": None})

    def advance(self, dt=None):
        if self.full:
            return
        end = time.time() - self.t0
        for pkt in self.capture.drain():
            arp = pkt[ARP]
            ts = max(float(pkt.time) - self.t0, 0.0)
            end = max(end, ts)
            self.real_count += 1
            if arp.psrc != "0.0.0.0":
                self.sender_mac.setdefault(arp.psrc, arp.hwsrc.lower())
            if arp.op == 1:
                self.target_hits[arp.pdst] = self.target_hits.get(arp.pdst, 0) + 1
                if "race" in self.active and self.gw and arp.pdst == self.gw[0]:     # forged reply wins the race
                    req = (arp.psrc, arp.hwsrc.lower())
                    if self.rng.random() < min(0.9, 0.5 * max(self.intensity, 0.1)):
                        self.queue.append((ts + 0.0002, _reply(ts, self.attacker, req, 1, claim_ip=self.gw[0])[1], "race"))
            self.queue.append((ts, pkt, ""))
        self._emit_attacks(end)
        self._flush(end)


if __name__ == "__main__":
    for s in SCENARIOS:
        d, m = simulate(s, 60)
        print(f"{s:11s} packets={m['packets']:5d} attack={m['attack_packets']:5d}")
