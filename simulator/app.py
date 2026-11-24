"""ARP spoofing detection - simulator web app.

Run:  streamlit run simulator/app.py

* "Simulate" builds traffic on a VIRTUAL LAN in memory (nothing is sent to any network),
  injects the chosen attack and scores every packet with all detectors.
* "Replay" scores a recorded capture from pcap_files/ packet by packet.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model_scripts"))
from alerting import ALERT_MIN_FLAGS, ALERT_MIN_SHARE, ALERT_WINDOW_S, alert_state  # noqa: E402
from config import load_config  # noqa: E402
from features import FEATURES, discover_pcaps, extract_arp_features  # noqa: E402
from model_factory import make_lr, make_rf  # noqa: E402
from simulate import (ATTACKS, SCENARIOS, HybridLAN, LiveCapture, LiveLAN, packets_per_second,  # noqa: E402
                      simulate, training_rate_limit)

MODE_HYBRID = "REAL live traffic + injected attacks"
MODE_LIVE = "Virtual LAN: launch attacks manually"
MODE_SCRIPTED = "Scripted simulation"
MODE_REPLAY = "Replay recorded capture"

DETECTORS = ["Naive Bayes", "Isolation Forest", "Ensemble (NB OR IF)",
             "Logistic Regression", "Random Forest", "Majority vote (NB, IF, LR, RF)"]
SHORT = {"Naive Bayes": "NB", "Isolation Forest": "IF", "Ensemble (NB OR IF)": "ENS",
         "Logistic Regression": "LR", "Random Forest": "RF", "Majority vote (NB, IF, LR, RF)": "VOTE"}

# The detectors we ship and stand behind, vs. models trained only for comparison. On a network busier than the
# training data, the comparison models (LR, RF) can flag every packet, so they are shown separately.
SHIPPED = ["Naive Bayes", "Isolation Forest", "Ensemble (NB OR IF)"]
COMPARISON = [d for d in DETECTORS if d not in SHIPPED]

# The alert rule (what counts as an ALERT vs stray flags) is in model_scripts/alerting.py
STATE_LABEL = {"quiet": "🟢 quiet", "stray": "🟡 stray flags", "alert": "🔴 ALERT"}

st.set_page_config(page_title="ARP Spoofing Simulator", page_icon="🛡️", layout="wide")


# ------------------------------------------------------------------ models
@st.cache_resource(show_spinner="Loading models (Logistic Regression / Random Forest are trained on the real pool)...")
def load_models():
    nb = joblib.load(ROOT / "models" / "saved_models" / "naive_bayes_model.pkl")
    iso = joblib.load(ROOT / "models" / "saved_models" / "isolation_forest_model.pkl")
    pool = pd.read_csv(ROOT / "dataset" / "arp_dataset_features.csv")      # REAL packets only
    cfg = load_config()
    lr, rf = make_lr(cfg), make_rf(cfg)
    lr.fit(pool[FEATURES], pool["label"])
    rf.fit(pool[FEATURES], pool["label"])
    raw = pd.read_csv(ROOT / "dataset" / "arp_dataset_raw.csv", usecols=["capture", "pkt_time", "label"])
    return {"nb": nb, "iso": iso, "lr": lr, "rf": rf, "n_train": len(pool),
            "rate_limit": training_rate_limit(raw)}        # busiest normal ARP rate seen in training (packets/s)


def score(df, models):
    X = df[FEATURES]
    nb = models["nb"].predict(X)
    iso = (models["iso"].predict(X) == -1).astype(int)
    lr = models["lr"].predict(X)
    rf = models["rf"].predict(X)
    return {
        "Naive Bayes": nb, "Isolation Forest": iso, "Ensemble (NB OR IF)": np.maximum(nb, iso),
        "Logistic Regression": lr, "Random Forest": rf,
        "Majority vote (NB, IF, LR, RF)": ((nb + iso + lr + rf) >= 3).astype(int),
    }


def metrics_table(df, preds, attack_start=None):
    y = df["is_attack"].to_numpy()
    t = df["time_s"].to_numpy()
    rows = []
    for name in DETECTORS:
        p = preds[name]
        tp = int(((p == 1) & (y == 1)).sum()); fn = int(((p == 0) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum()); tn = int(((p == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        fpr = fp / (fp + tn) if fp + tn else np.nan
        f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else np.nan
        row = {"Detector": name, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
               "Recall": rec, "Precision": prec, "FPR (all normal)": fpr, "F1": f1}
        if attack_start is not None:
            before = (y == 0) & (t < attack_start)
            during = (y == 0) & (t >= attack_start)
            row["FPR clean traffic"] = float(p[before].mean()) if before.any() else np.nan
            row["Legit pkts flagged during attack"] = float(p[during].mean()) if during.any() else np.nan
            hit = np.where((p == 1) & (y == 1))[0]
            row["Time to detect (s)"] = float(t[hit[0]] - attack_start) if len(hit) else np.nan
            row["Attack pkts before 1st alert"] = int(hit[0] - np.where(y == 1)[0][0]) if len(hit) else np.nan
        else:
            row["FPR clean traffic"] = fpr
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ live mode
ATTACK_SHORT = {"basic": "Basic spoof", "flooding": "Flooding", "gratuitous": "Gratuitous",
                "race": "Race condition", "unicast": "Unicast spoof"}


def mask_mac(m):
    return m[:8] + ":xx:xx:xx" if isinstance(m, str) and len(m) >= 8 else m


def mask_ip(ip):
    parts = str(ip).split(".")
    return ".".join(parts[:2] + ["x", "x"]) if len(parts) == 4 else ip


def live_page(models, normal_rate, n_hosts, seed, intensity, tick_s, hybrid=False, iface=None, mask_ids=False):
    ss = st.session_state
    ss.setdefault("lan", None)
    ss.setdefault("running", False)
    ss.setdefault("cap", None)
    if (ss.lan is not None) and (isinstance(ss.lan, HybridLAN) != hybrid):       # switched mode: drop the other kind of run
        ss.lan, ss.running = None, False
    if hybrid:
        st.warning("**REAL traffic, captured live and read-only** from your network interface; real packets are assumed "
                   "normal. **You** decide when an attack is *injected*: forged packets built from real hosts seen so far "
                   "are added to the in-memory stream only - **nothing is ever transmitted on the network**. "
                   "Injected attacks are simulated, so detection scores are illustrations, not accuracy evidence.")
    else:
        st.warning("**SIMULATED live LAN.** Normal traffic runs on its own; **you** decide when an attack starts. Packets "
                   "exist only in memory - nothing is sent to any network. Simulated attacks are cleaner than the lab "
                   "recordings, so scores here illustrate behaviour and are not accuracy evidence.")

    c = st.columns([1.2, 1, 1.3, 3])
    if c[0].button("▶ Start / Reset", type="primary", use_container_width=True):
        if ss.cap is not None:
            ss.cap.stop()
            ss.cap = None
        if hybrid:
            try:
                ss.cap = LiveCapture(iface)
            except Exception as exc:                       # no Npcap / no permission / bad interface
                st.error(f"Could not start the capture on '{iface}': {exc}")
                return
            ss.lan = HybridLAN(ss.cap, int(seed))
        else:
            ss.lan = LiveLAN(normal_rate, n_hosts, int(seed), keep_packets=False)
        ss.live_scored = {"n": 0, "flags": {d: np.zeros(0, dtype=int) for d in DETECTORS}}
        ss.running = True
        st.rerun()
    lan = ss.lan
    if lan is None:
        st.info("Press **▶ Start / Reset** to " + ("start the passive live capture" if hybrid else "bring up the virtual LAN")
                + ", then use the launch buttons to start an attack.")
        return
    lan.intensity = intensity
    if not hybrid:
        lan.normal_rate = normal_rate
    if c[1].button("⏸ Pause" if ss.running else "▶ Resume", use_container_width=True):
        ss.running = not ss.running
        st.rerun()
    if c[2].button("⏹ Stop all attacks", use_container_width=True):
        lan.stop_all()
        st.rerun()

    st.subheader("Launch an attack")
    cols = st.columns(len(ATTACKS))
    for col, key in zip(cols, ATTACKS):
        active = key in lan.active
        if col.button(("⏹ Stop " if active else "🚀 Launch ") + ATTACK_SHORT[key], key="atk_" + key,
                      type="primary" if active else "secondary", use_container_width=True,
                      help=ATTACKS[key]):
            if hybrid and not active and not lan.ready:
                st.toast("Waiting for real ARP traffic: need at least 3 hosts seen on the network first.")
            else:
                (lan.stop if active else lan.start)(key)
                st.rerun()

    def body():
        if lan.full and ss.running:
            ss.running = False
            st.rerun()
        if ss.running:
            lan.advance(tick_s)
        df = lan.frame()
        if df.empty:
            st.info("Waiting for traffic...")
            return
        sc = ss.live_scored
        if len(df) > sc["n"]:
            new = score(df.iloc[sc["n"]:], models)
            for d in DETECTORS:
                sc["flags"][d] = np.concatenate([sc["flags"][d], new[d]])
            sc["n"] = len(df)
        flags = pd.DataFrame({SHORT[d]: sc["flags"][d] for d in DETECTORS})
        y = df["is_attack"].to_numpy()

        if lan.full:
            st.error(f"Packet limit ({len(df)}) reached - press **Start / Reset** for a new run.")
        m = st.columns(5 if hybrid else 4)
        m[0].metric("LAN clock", f"{lan.t:.0f} s")
        m[1].metric("Real packets captured" if hybrid else "Packets", lan.real_count if hybrid else len(df))
        m[2].metric("Injected attack packets" if hybrid else "Attack packets", int(y.sum()))
        m[3].metric("Active attacks", ", ".join(ATTACK_SHORT[a] for a in lan.active) or "none")
        if hybrid:
            m[4].metric("Real hosts seen", len(lan.sender_mac))

        recent = (df["time_s"] > lan.t - ALERT_WINDOW_S).to_numpy()


        def status_card(col, d):
            f = flags[SHORT[d]].to_numpy()
            col.metric(d.split(" (")[0], STATE_LABEL[alert_state(df["eth_src"], f == 1, recent)],
                       f"caught {int(((f == 1) & (y == 1)).sum())} · false {int(((f == 1) & (y == 0)).sum())}",
                       delta_color="off")

        real_rate = packets_per_second(df.loc[df["is_attack"] == 0, "time_s"], lan.t)
        if real_rate > models["rate_limit"]:
            st.warning(f"**Busy network:** about {real_rate:.0f} ARP packets/s right now, but the normal traffic the models "
                       f"were trained on never went above about {models['rate_limit']:.0f}/s. The comparison models "
                       f"(Logistic Regression, Random Forest) are unreliable at this rate and tend to flag everything - "
                       f"read the shipped detectors.")

        st.subheader(f"Shipped detectors (last {ALERT_WINDOW_S} s)")
        st.caption(f"🔴 ALERT = a sender with at least {ALERT_MIN_FLAGS} flagged packets and over {ALERT_MIN_SHARE:.0%} of its "
                   f"packets flagged in {ALERT_WINDOW_S} s · 🟡 stray flags = flagged packets that don't look like an attacker "
                   "(normal noise) · 🟢 quiet = nothing flagged.")
        for col, d in zip(st.columns(len(SHIPPED)), SHIPPED):
            status_card(col, d)
        st.caption("Comparison models, for reference only: trained on quieter traffic, so they can over-alert on a new network.")
        for col, d in zip(st.columns(len(COMPARISON)), COMPARISON):
            status_card(col, d)

        cum = pd.concat([df[["is_attack"]].rename(columns={"is_attack": "ACTUAL ATTACK"}).cumsum(), flags.cumsum()], axis=1)
        st.line_chart(cum.iloc[::max(1, len(cum) // 500)], height=280)

        sub = df.iloc[-10:]
        feed = pd.DataFrame({
            "t (s)": sub["time_s"].round(2),
            "sender IP": sub["src_ip"].map(mask_ip) if mask_ids else sub["src_ip"],
            "sender MAC": sub["src_mac"].map(mask_mac) if mask_ids else sub["src_mac"], "type": sub["kind"],
            "truth": np.where(sub["is_attack"] == 1,
                              ("INJECTED: " if hybrid else "ATTACK: ") + sub["attack_name"],
                              "real (assumed normal)" if hybrid else "normal"),
            **{SHORT[d]: np.where(flags.loc[sub.index, SHORT[d]] == 1, "🔴", "·") for d in DETECTORS}})
        st.dataframe(feed.iloc[::-1], use_container_width=True, hide_index=True)

        if lan.log:
            st.subheader("How fast was each launch detected? (first flagged attack packet)")
            rows = []
            for e in lan.log:
                stop = e["stop"] if e["stop"] is not None else lan.t
                msk = ((df["attack_name"] == e["name"]) & (df["time_s"] >= e["start"]) & (df["time_s"] <= stop + 0.01)).to_numpy()
                row = {"Attack": ATTACK_SHORT[e["name"]], "Launched (s)": round(e["start"], 1),
                       "Attack pkts": int(msk.sum())}
                for d in DETECTORS:
                    hit = np.where(msk & (flags[SHORT[d]].to_numpy() == 1))[0]
                    row[SHORT[d]] = (f"{df['time_s'].iloc[hit[0]] - e['start']:.2f} s" if len(hit)
                                     else ("missed" if msk.sum() else "-"))
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            mt = metrics_table(df, {d: flags[SHORT[d]].to_numpy() for d in DETECTORS}, lan.log[0]["start"])
            mt["Role"] = np.where(mt["Detector"].isin(SHIPPED), "shipped", "comparison")
            mt = mt.sort_values("Role", key=lambda r: r != "shipped", kind="stable")     # shipped detectors first
            st.subheader("Overall result so far")
            ms = mt[mt["Role"] == "shipped"]
            if y.sum() and ms["F1"].notna().any():
                msg = (f"Among the shipped detectors - best F1: **{ms.loc[ms['F1'].idxmax(), 'Detector']}** "
                       f"({ms['F1'].max():.3f}) · highest recall: **{ms.loc[ms['Recall'].idxmax(), 'Detector']}** "
                       f"({ms['Recall'].max():.1%})")
                if ms["FPR clean traffic"].notna().any():
                    msg += (f" · lowest false alarms on clean traffic: **{ms.loc[ms['FPR clean traffic'].idxmin(), 'Detector']}** "
                            f"({ms['FPR clean traffic'].min():.1%})")
                st.success(msg + ".")
            cols_ = ["Detector", "Role", "TP", "FN", "FP", "TN", "Recall", "Precision", "FPR clean traffic",
                     "Legit pkts flagged during attack", "F1"]
            fmt = {k: "{:.3f}" for k in ["Recall", "Precision", "F1", "FPR clean traffic", "Legit pkts flagged during attack"]}
            st.dataframe(mt[cols_].style.format(fmt, na_rep="-"), hide_index=True, use_container_width=True)
            if hybrid:
                st.caption("**FPR clean traffic = false alarms on REAL live traffic** before your first injection (real packets "
                           "are assumed normal; a genuine attack on your network would show up as a 'false' alarm).")
            st.caption("Clean-traffic false alarms are counted before your first launch. After a launch, legitimate packets "
                       "from a poisoned IP stay suspicious because `ip_mac_count` never decays.")

    st.fragment(run_every=1.0 if (ss.running and not lan.full) else None)(body)()


# ------------------------------------------------------------------ UI
st.title("🛡️ ARP Spoofing Detection - Simulator")
models = load_models()

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", [MODE_HYBRID, MODE_LIVE, MODE_SCRIPTED, MODE_REPLAY])
    if mode == MODE_HYBRID:
        ifaces = LiveCapture.interfaces()
        iface = st.selectbox("Capture interface (passive, ARP only)", ifaces,
                             index=ifaces.index("Wi-Fi") if "Wi-Fi" in ifaces else 0)
        intensity = st.slider("Attack intensity (x)", 0.5, 3.0, 1.0, 0.5)
        mask_ids = st.checkbox("Mask real MAC / IP addresses on screen", True)
        seed = st.number_input("Random seed", 0, 9999, 7)
        normal_rate, n_hosts, tick_s = 3.0, 8, 1
        st.caption("Real ARP traffic is captured read-only and kept in memory. Attacks are added to that in-memory "
                   "stream only; nothing is transmitted on the network.")
        run = False
    elif mode == MODE_LIVE:
        normal_rate = st.slider("Normal traffic (requests/s)", 0.5, 10.0, 3.0, 0.5)
        n_hosts = st.slider("Hosts on the LAN", 3, 30, 8)
        intensity = st.slider("Attack intensity (x)", 0.5, 3.0, 1.0, 0.5)
        tick_s = st.select_slider("Simulation speed (x real time)", [1, 2, 5], 1)
        seed = st.number_input("Random seed", 0, 9999, 7)
        st.caption("Press **Start / Reset** on the main page, then launch attacks whenever you like.")
        run = False
    elif mode == MODE_SCRIPTED:
        scenario = st.selectbox("Scenario", list(SCENARIOS), index=1, format_func=lambda k: SCENARIOS[k])
        duration = st.slider("Duration (s)", 20, 180, 60, 10)
        normal_rate = st.slider("Normal traffic (requests/s)", 0.5, 10.0, 3.0, 0.5)
        n_hosts = st.slider("Hosts on the LAN", 3, 30, 8)
        intensity = st.slider("Attack intensity (x)", 0.5, 3.0, 1.0, 0.5)
        seed = st.number_input("Random seed", 0, 9999, 7)
    else:
        caps = {p.name: p for p, _ in discover_pcaps()}
        caps.update({"holdout/" + p.name: p for p in sorted((ROOT / "pcap_files" / "holdout").glob("*.pcap*"))})
        cap_path = caps[st.selectbox("Capture", list(caps))]
    if mode not in (MODE_LIVE, MODE_HYBRID):
        speed = st.slider("Replay steps (more = smoother)", 10, 80, 40)
        animate = st.checkbox("Animate live replay", True)
        run = st.button("▶ Run", type="primary", use_container_width=True)
    st.caption(f"Logistic Regression / Random Forest trained on {models['n_train']} real packets. "
               "Naive Bayes / Isolation Forest are the shipped models.")

if mode != MODE_HYBRID and st.session_state.get("cap") is not None:      # leave live capture when switching away
    st.session_state["cap"].stop()
    st.session_state["cap"] = None
if mode in (MODE_LIVE, MODE_HYBRID):
    live_page(models, normal_rate, n_hosts, seed, intensity, tick_s,
              hybrid=(mode == MODE_HYBRID), iface=iface if mode == MODE_HYBRID else None,
              mask_ids=mask_ids if mode == MODE_HYBRID else False)
    st.stop()

if run:
    if mode == MODE_SCRIPTED:
        with st.spinner("Generating virtual-LAN traffic..."):
            df, meta = simulate(scenario, duration, normal_rate, n_hosts, intensity, int(seed))
        df["info"] = np.where(df["is_attack"] == 1, "attacker-injected", "legitimate")
        meta["title"] = SCENARIOS[scenario]
    else:
        with st.spinner("Extracting features from the capture..."):
            df = extract_arp_features(cap_path)
        label = 0 if "normal" in cap_path.name.lower() else 1
        df = df.reset_index(drop=True)
        df["is_attack"] = label
        df["time_s"] = df["pkt_time"] - df["pkt_time"].iloc[0]
        df["kind"] = np.where(df["op"] == 1, "request", "reply")
        df["info"] = "file label: " + ("attack" if label else "normal")
        in_holdout = cap_path.parent.name == "holdout"
        meta = {"title": cap_path.name, "attack_start_s": None, "packets": len(df),
                "attack_packets": int(label) * len(df), "replay": True, "unseen": in_holdout}
    preds = score(df, models)
    st.session_state["res"] = (df, preds, meta)
    st.session_state["animate"] = animate

if "res" not in st.session_state:
    st.info("Pick a scenario in the sidebar and press **▶ Run**. Try *Basic ARP spoofing*, then *Normal traffic only* "
            "to see false alarms, then *Flooding*.")
    st.stop()

df, preds, meta = st.session_state["res"]
replay = meta.get("replay", False)
if replay:
    st.warning("**REPLAY of a recorded capture.** Per-packet truth is not known: every packet carries the capture's "
               "file-level label, so 'recall' here is the share flagged. "
               + ("This capture was never used for training." if meta["unseen"] else
                  "**This capture was part of the training pool, so scores are optimistic.** Use `holdout/` for an honest demo."))
else:
    st.warning("**SIMULATED traffic on a virtual LAN.** Nothing is sent to any network. The attacker is a script inside this "
               "app, so per-packet truth is exact. Simulated attacks are cleaner than the lab recordings, so scores here "
               "illustrate behaviour; they are **not** accuracy evidence (see the *Overall* tab).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Scenario / capture", meta["title"][:32])
c2.metric("Packets", meta["packets"])
c3.metric("Attack packets" if not replay else "File label", meta["attack_packets"] if not replay else ("attack" if df.is_attack.iloc[0] else "normal"))
c4.metric("Duration", f"{df['time_s'].iloc[-1]:.0f} s")

tab_live, tab_models, tab_overall, tab_pkts = st.tabs(["Live replay", "Model results", "Overall", "Packets"])

flags = pd.DataFrame({SHORT[n]: preds[n] for n in DETECTORS})
cum = pd.concat([df[["is_attack"]].rename(columns={"is_attack": "ACTUAL ATTACK" if not replay else "FILE LABEL"}).cumsum(),
                 flags.cumsum()], axis=1)

with tab_live:
    st.caption("Cumulative packets flagged by each detector vs. real attack packets. Lines that hug **ACTUAL ATTACK** "
               "detect well; lines above it are false alarms.")
    chart_ph, feed_ph = st.empty(), st.empty()

    def show(upto):
        chart_ph.line_chart(cum.iloc[:upto], height=320)
        sub = df.iloc[max(0, upto - 12):upto]
        feed = pd.DataFrame({
            "t (s)": sub["time_s"].round(3), "sender IP": sub["src_ip"], "sender MAC": sub["src_mac"], "type": sub["kind"],
            "truth": np.where(sub["is_attack"] == 1, "ATTACK" if not replay else "attack-file", "normal"),
            **{SHORT[n]: np.where(flags.loc[sub.index, SHORT[n]] == 1, "🔴 alert", "·") for n in DETECTORS}})
        feed_ph.dataframe(feed.iloc[::-1], use_container_width=True, hide_index=True)

    n = len(df)
    if st.session_state.get("animate"):
        steps = speed
        bar = st.progress(0.0)
        for i in range(1, steps + 1):
            show(max(1, int(n * i / steps)))
            bar.progress(i / steps)
            time.sleep(0.08)
        bar.empty()
        st.session_state["animate"] = False
    else:
        show(n)

mt = metrics_table(df, preds, meta.get("attack_start_s") if not replay else None)
best = None
if mt["F1"].notna().any():
    best = mt.loc[mt["F1"].idxmax()]

with tab_models:
    if replay:
        st.subheader("Share of packets flagged per detector")
        share = pd.DataFrame({"Detector": DETECTORS, "Flagged": [float(preds[d].mean()) for d in DETECTORS]})
        st.bar_chart(share.set_index("Detector"))
        label_txt = "recall (capture is attack)" if df.is_attack.iloc[0] else "false-positive rate (capture is normal)"
        st.caption(f"Because the capture has a single file-level label, the flagged share equals the {label_txt}.")
        st.dataframe(share.style.format({"Flagged": "{:.1%}"}), hide_index=True, use_container_width=True)
    else:
        show_cols = ["Detector", "TP", "FN", "FP", "TN", "Recall", "Precision", "FPR clean traffic",
                     "Legit pkts flagged during attack", "F1", "Time to detect (s)", "Attack pkts before 1st alert"]
        show_cols = [c for c in show_cols if c in mt.columns]
        fmt = {c: "{:.3f}" for c in ["Recall", "Precision", "F1", "FPR clean traffic", "Legit pkts flagged during attack"]}
        fmt["Time to detect (s)"] = "{:.2f}"
        st.dataframe(mt[show_cols].style.format(fmt, na_rep="-"), hide_index=True, use_container_width=True)
        st.bar_chart(mt.set_index("Detector")[["Recall", "Precision", "F1", "FPR clean traffic"]], height=320)
        st.markdown(
            "- **Recall** = share of attacker packets flagged. **FPR clean traffic** = legitimate packets flagged "
            "*before* the attack began (true false alarms).\n"
            "- **Legit pkts flagged during attack**: `ip_mac_count` never decays, so once an IP has been claimed by two "
            "MACs, every later packet from that IP (the real owner's included) still looks suspicious. These count as "
            "false positives here because only attacker-injected packets are 'attack', though an alert on a poisoned "
            "identity is arguably a correct warning. It is also a real limitation: alarms persist after the attack stops.\n"
            "- **Time to detect** = seconds from the first attack packet until the detector first flags an attacker packet.")

with tab_overall:
    st.subheader("Overall result")
    if replay:
        top = pd.Series({d: preds[d].mean() for d in DETECTORS})
        st.write(f"On **{meta['title']}** the detectors flagged between {top.min():.1%} and {top.max():.1%} of packets.")
    else:
        recall_all = mt.set_index("Detector")["Recall"]
        if meta["attack_packets"] == 0:
            clean = mt.set_index("Detector")["FPR clean traffic"]
            st.success(f"No attack in this run: false alarms on normal traffic range from {clean.min():.1%} "
                       f"({clean.idxmin()}) to {clean.max():.1%} ({clean.idxmax()}).")
        else:
            caught = [d for d in DETECTORS if recall_all[d] > 0]
            st.success(f"**{len(caught)} of {len(DETECTORS)} detectors raised an alert** on this attack. "
                       f"Best F1: **{best['Detector']}** ({best['F1']:.3f}); highest recall: "
                       f"**{recall_all.idxmax()}** ({recall_all.max():.1%}).")
            fastest = mt.dropna(subset=["Time to detect (s)"]).sort_values("Time to detect (s)")
            if len(fastest):
                st.write(f"Fastest alert: **{fastest.iloc[0]['Detector']}** after {fastest.iloc[0]['Time to detect (s)']:.2f} s.")
    st.subheader("Benchmark on REAL recorded packets (offline evaluation)")
    st.caption("These are the numbers to quote; simulator runs above are illustrations.")
    cmp_path = ROOT / "results" / "model_comparison.json"
    if cmp_path.exists():
        cmp = json.loads(cmp_path.read_text(encoding="utf-8"))
        rows = []
        for name, m in cmp["holdout"].items():
            same = cmp["same_network"].get("192.168", {}).get("results", {}).get(name, {})
            rows.append({"Model": name, "Recall (all-data)": m["recall"], "FPR (all-data)": m["false_positive_rate"],
                         "MCC (all-data)": m["mcc"], "Recall (same network)": same.get("recall"),
                         "FPR (same network)": same.get("false_positive_rate"),
                         "FPR unseen evening capture": cmp["unseen_holdout_fpr"].get(name)})
        st.dataframe(pd.DataFrame(rows).style.format({c: "{:.3f}" for c in
                     ["Recall (all-data)", "FPR (all-data)", "MCC (all-data)", "Recall (same network)",
                      "FPR (same network)", "FPR unseen evening capture"]}, na_rep="-"),
                     hide_index=True, use_container_width=True)
        st.markdown("**Reading it:** Random Forest scores best overall but has the highest false-positive rate on the "
                    "same-network test, so part of its gain may be recognising the network. Naive Bayes is the most "
                    "stable and is the shipped model. Isolation Forest / ensemble numbers are in `results/metrics.md`.")
    else:
        st.info("Run `python model_scripts/compare_models.py` to create `results/model_comparison.json`.")

with tab_pkts:
    out = df[["time_s", "src_ip", "src_mac", "dst_ip", "kind", "is_attack"] + FEATURES].copy()
    for d in DETECTORS:
        out[SHORT[d]] = preds[d]
    st.dataframe(out, use_container_width=True, hide_index=True, height=420)
    st.download_button("Download per-packet results (CSV)", out.to_csv(index=False).encode(),
                       "simulation_results.csv", "text/csv")
