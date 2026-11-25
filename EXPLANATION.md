# ARP Spoofing Detection — Master Explanation (ML Lab)

Walkthrough of the project in this folder: the research context (the base paper and the gap we address), the problem, the
data, how the dataset/labels/features are built, the models, how they are evaluated, tuning/tracking/tests, the simulator,
what every file is, the known limitations, and a prep list of likely teacher questions. The shipped detectors are Gaussian
Naive Bayes and an Isolation Forest (scikit-learn); Logistic Regression and Random Forest are trained for comparison. No deep
learning is used.

Exact numbers live in [`results/metrics.md`](results/metrics.md) and [`results/model_comparison.md`](results/model_comparison.md)
(auto-generated); this document quotes the latest run.

Authors: Garv Noor (1024170189), Manya Gupta (1024170192).

---

## 0. Research context: the base paper and the gap

**Base paper** (PDF in `Base Paper/`): Singhal and Kumar, *"A cost-sensitive random forest framework for ARP spoofing
detection in Internet of Medical Things networks"*, Frontiers in Big Data 9 (2026), DOI 10.3389/fdata.2026.1878242. A
peer-reviewed, open-access journal article, published 22 July 2026.

**What the paper does** (checked against the PDF):
- A cost-sensitive Random Forest (SH-IORF) for ARP spoofing in medical IoT.
- Public dataset CICIoMT2024: 347,721 network *flows*, 47 flow features, from a testbed of "real and simulated devices".
  The task is binary (ARP spoofing vs benign).
- 18 added timing features (65 total), pruned by correlation to 52.
- One stratified 70/15/15 train/validation/test split. Class weights and the decision threshold (0.35, chosen under a
  false-negative-rate constraint of at most 0.5%) are tuned on the validation part only.
- Results: 99.90% accuracy, 0.17% missed attacks (40 of 23,390), 12 false alarms among 28,768 benign flows.
- One extra check on a second public dataset, CICIDS2017: 99.32% accuracy.
- Its comparison table lists accuracy numbers reported by other papers on their own datasets.
- Its authors state the framework was not validated on real devices and is offline.

**The gap, and the reason we built this project.** The paper's headline score comes from splitting one public dataset into
train and test. It added one check on a second public dataset, which is good. But both datasets are ready-made, and neither is
traffic the authors recorded or an attack type held out of training. **Our research area is ARP-spoofing detection tested beyond
a train/test split.**

**Research questions.** RQ1: does detection work on ARP traffic we recorded ourselves? RQ2: does it hold up on attack types and
recordings held out of training? RQ3: can it be shown working on live traffic?

**What we did instead** (details in the sections below):

| | Base paper | This project |
|---|---|---|
| Data | public benchmark of flows | 9,243 real ARP packets we recorded (Wi-Fi traffic + lab attacks) |
| Attacks | one class | 5 ARP attack types, each tested on its own |
| Features | 47 flow + 18 timing features, pruned to 52 | 10 per-packet ARP features, no raw IP/MAC |
| Testing | one 70/15/15 split + a second public dataset | chronological split, same-network, leave-one-capture-out, unseen capture; tuned by cross-validation |
| Baselines | numbers copied from other papers | we trained NB, IF, LR and RF under the same protocol |
| Demo | offline | live simulator on real traffic |

**Answers (summary).** RQ1 yes: Naive Bayes reaches recall 0.87 at about 0.1% false alarms on the all-data hold-out and recall 0.82
at 1.8% on the same-network test. RQ2 partly: flooding 0.94 and unicast spoof 0.91 are caught on captures held out of training,
basic spoof 0.48 and gratuitous 0.43 are lower (partly label noise), and the unseen evening capture gives 2.7% false alarms.
RQ3 yes as a demonstration (the injected attacks are simulated, so scores there are illustrations).

**What not to claim.** The data and the task differ (flows vs. packets), so the numbers are **not directly comparable**: we claim
*stronger testing*, not higher accuracy. Their dataset is much larger and their scores are higher, and their cost-sensitive
threshold is something we did not do. Our "unseen" evening capture is the same campus Wi-Fi at a different time, not a different
network, and our attacks are lab-staged (real packets, not attacks in the wild).

Other papers read are indexed in `reference_papers/README.md`; the earlier base paper (Ali et al., 2025, an arXiv preprint) is
kept there for reference.

---

## 1. The problem: what is ARP spoofing

Every LAN device has an **IP address** (logical) and a **MAC address** (physical). To send
data to an IP, device A needs the matching MAC. **ARP** resolves it:

- A broadcasts *"Who has 192.168.1.5?"* -> **request, opcode 1**
- B answers *"192.168.1.5 is at xx:xx:..."* -> **reply, opcode 2**

ARP has **no authentication**: anyone can send an unsolicited reply claiming any IP, and
others update their ARP cache. **ARP spoofing** abuses this to redirect traffic through the
attacker (man-in-the-middle) or to flood the network. Variants in the data:

- **Basic spoof** — attacker claims another host's/router's IP
- **ARP flooding** — floods of forged replies with random IP/MAC pairs
- **Gratuitous ARP** — unsolicited announcements that pre-poison caches
- **Race condition** — forged reply raced against the real one
- **Unicast spoofing** — the forged reply sent directly to one victim (stealthier)

## 2. The raw data: `.pcapng` files

A `.pcapng` is a timestamped recording of raw packets (Wireshark/tcpdump). Nine captures are in
`pcap_files/`: four normal and five attack captures (plus one unseen normal capture in `holdout/`). New captures can simply be dropped in:
the **file name decides the label** (`normal` in the name -> 0, otherwise 1). Captures in
`pcap_files/holdout/` are not used by training and serve as unseen test data.
**Network matters:** the five attack captures and `arp_normal_traffic` were recorded on one LAN (192.168.1.x, gateway
192.168.1.1, about six devices, recorded in July 2025); the other three normal captures and the holdout capture come from
campus Wi-Fi (172.16.x.x; recorded passively with `dumpcap -f arp`). Training tags each capture
with its network and reports a same-network evaluation, because mixing networks lets a model learn the network
instead of the attack. The lab network could not be revisited, so no more normal traffic could be recorded on it.
How to record a new normal capture: [`docs/RECORDING_NORMAL_TRAFFIC.md`](docs/RECORDING_NORMAL_TRAFFIC.md).

**Privacy.** Captures and the raw CSV contain real MAC/IP addresses of other people's devices; anonymise before publishing.

## 3. Scapy's role (and only its role)

Scapy is only a **decoder** (`rdpcap`, `pkt[ARP].psrc`, `pkt[Ether].src`, ...). It has no concept
of spoofing. All feature engineering and detection logic is plain Python/pandas in
[`model_scripts/features.py`](model_scripts/features.py), which is imported by `train.py`, `predict.py` and the simulator so
training and inference can never drift apart. `FeatureExtractor` keeps its state between calls: one big call on a whole capture
and many small calls on a live stream give identical rows (tested), and checked identical on all 7,786 real packets.

## 4. Labelling (and its main limitation)

Labels are **per capture file**: every packet in a normal capture is 0, every packet in an attack
capture is 1. This is *weak labelling* — practical (nobody hand-labels thousands of packets) but
imperfect: an attack capture also contains ordinary background ARP.

**Measured:** in `arp_spoof_basic` and `arp_gratuitous_announcement`, about half of the packets
(47% / 51%) are plain ARP requests with no attack signal at all (no MAC mismatch, no conflict,
not unsolicited, no multi-IP claims) yet are labelled "attack". That is largely label noise, not model
failure, and it is the main reason Naive Bayes recall on those two captures is only ~45-50%. (Random Forest scores higher on them,
so the noise is *partly*, not wholly, the explanation.) The proper fix is per-packet ground truth (e.g. the attacker's MAC for each
capture), which this dataset does not provide; rule-derived labels would be circular because the same signals are
model features. It is therefore reported as a limitation rather than "fixed".

## 5. Feature engineering

Per ARP packet, computed **separately inside each capture** (state resets per file), using only past packets:

| Feature | Meaning | Why it matters |
|---|---|---|
| `op` | 1 request / 2 reply | attacks are mostly forged replies |
| `is_broadcast` | Ethernet dst is `ff:ff:ff:ff:ff:ff` | gratuitous/flood traffic is broadcast |
| `is_gratuitous` | sender IP == target IP | announcement-style poisoning |
| `mac_conflict` | IP seen before with a different MAC | the literal definition of spoofing |
| `ip_mac_count` | distinct MACs claimed for this IP | repeated identity changes |
| `l2_ip_count` | distinct IPs claimed by this Ethernet sender | flooding/poisoning many IPs |
| `l2_reply_rate_1s` | replies from this sender in the last 1 s | flood rate |
| `arp_reply_count` | running reply count for the sender IP | volume |
| `log_interarrival_ms` | log(1+gap to previous packet) | burst timing |
| `log_reply_delay_ms` | log(1+request->reply delay), 0 if unmatched | forged replies have no real request |

Two further columns are computed but **not used as model inputs**: `is_unsolicited_reply` (reply with no earlier request -
it depends on the sniffer's viewpoint: 0% in one normal capture, 53-64% in others) and `eth_arp_mac_mismatch` (Ethernet
src MAC != ARP sender MAC - in the attack captures this is an artifact of the attack tool, whose forged frames have
Ethernet source `00:00:00:00:00:00`; a real attacker would use their own MAC). Dropping both did not change attack recall
in leave-one-capture-out tests but removed a false-positive problem on new networks.

ARP probes (sender IP `0.0.0.0`) are excluded from the IP<->MAC bookkeeping because they carry no real sender IP.
Timing features are log-transformed because raw gaps are extremely heavy-tailed, which suits neither
Gaussian Naive Bayes nor a fair comparison across features.

**What changed from the first version, and why** (useful to say in the viva):

| Problem found in review | Fix |
|---|---|
| `is_broadcast` was always 0 (it tested the ARP target MAC, which is `00:..` for requests) | now uses the **Ethernet** destination; added `is_gratuitous` and the windowed/count features (`ip_mac_count`, `l2_ip_count`, `l2_reply_rate_1s`) |
| Label-encoded IP/MAC columns let the model memorise the capture file; fake ordinal order; crash on unseen addresses | identities removed from the model inputs (kept in the raw CSV); encoders deleted |
| `interarrival_time` was computed across merged captures (gaps up to ~5 days) | computed per capture; log-scaled |
| Random packet split; neighbouring near-identical packets straddled train/test; no `random_state` | chronological split per capture + leave-one-capture-out; fixed seeds, fully deterministic |
| Isolation Forest/ensemble never evaluated; demo ran on a training file | NB, IF and ensemble evaluated on held-out data; `predict.py` takes any pcap and supports a `holdout/` folder |
| Feature code copy-pasted in two scripts | single shared `features.py` (now also used by the simulator) |
| README and EXPLANATION quoted different "latest" numbers | one source of truth: `results/metrics.md` |

## 6. Dataset files

**Size:** 9,243 *real* recorded packets = 7,786 training pool (5,883 normal, 1,903 attack; ~3:1) + 1,457 in a
never-trained-on hold-out capture. Plus 3,980 *synthetic* rows, kept separate and not counted as recorded data (below).

`train.py` assembles the rows and saves:

- `dataset/arp_dataset_raw.csv` — **real**, human-readable (IPs, MACs, Ethernet addresses, all features, diagnostic columns, label, capture name, network)
- `dataset/arp_dataset_features.csv` — **real**, the exact model input (feature columns + label + capture)
- `dataset/arp_dataset_holdout.csv` — **real**, the unseen hold-out capture
- `dataset/arp_dataset_synthetic.csv` — **synthetic only** (`synthetic=1`)
- `dataset/arp_dataset_balanced.csv` — real + synthetic with a `synthetic` column (0/1)

### Synthetic augmentation (disclosure)

The teacher asked for 8k-10k records. We recorded 9,243 real packets, but the data is 3:1 normal-heavy and attack traffic can
only be recorded in the lab, so `model_scripts/augment.py` generates 3,980 **synthetic** attack rows (SMOTE-style: a new row is
an interpolation between a real attack packet and one of its 5 nearest neighbours in the same capture; binary/count columns
are rounded; the 9-packet race-condition capture is not oversampled) to balance the classes at 5,883 / 5,883.

- Synthetic rows are **always labelled** (`synthetic=1`, capture name `synthetic<-...`) and stored separately; they are never
  described as recorded traffic.
- They add class balance, **not information**: every row is an interpolation of packets we already have.
- In every evaluation they are generated from the *training* part only; **all reported metrics use real packets only**.
- **Result: they do not help.** Same-network NB recall is 0.792 with them vs 0.816 without; all-data results are identical;
  and on leave-one-out `arp_normal_campus_wifi_03` the ensemble false-positive rate is 0.13 with them vs 0.02 without. The
  shipped models are therefore trained on real packets only (`train.py --augment` trains with them). A fair thing to say in
  the viva: we tried class balancing, measured it, and it did not improve the detector, so we kept the simpler model.

## 7. Models and why

### Gaussian Naive Bayes — supervised
Assumes features are Gaussian and conditionally independent given the class, and applies Bayes' theorem.
Fast, needs little data, a good interpretable baseline, and it was the steadiest model across networks. It is not claimed optimal.
Its probabilities are overconfident (log loss ~4.4) because the independence assumption is violated
(e.g. `l2_ip_count` and `l2_reply_rate_1s` are correlated) — rely on its class/ranking, not its probabilities.

### Isolation Forest — unsupervised anomaly detector
Random trees isolate points; outliers isolate in few splits. Trained **only on normal packets** so it flags *any*
deviation, including attack types never seen. `contamination=0.005` (tuned from 0.02, see Section 8b): by construction ~0.5% of
normal training packets are flagged, which sets its false-positive budget. It catches flooding but misses quiet attacks.

### Ensemble
Attack if **either** model flags it (OR). Missing an attack is costlier than a false alarm, so OR is meant to trade a few
extra false positives for a few points more recall. After tuning, the Isolation Forest is less trigger-happy and on this data adds
little beyond Naive Bayes (same-network ensemble recall 0.816 = Naive Bayes; it was 0.862 before tuning), so the OR rule now
mostly adds false alarms on the all-data test.

### Logistic Regression and Random Forest — comparison only
Trained under the same protocol (Section 8c). Random Forest is the model family the base paper uses.

## 8. Evaluation

Four views, all produced by `train.py` into `results/metrics.md` (real packets only):

1. **Same-network hold-out (192.168 only)** - the fairest view: normal and attack traffic from one environment,
   chronological 70/30 split per capture (test: 56 normal / 571 attack).
2. **All-data hold-out** - both networks (test: 1,765 normal / 571 attack). More normal packets, but network-confounded.
3. **Leave-one-capture-out** - train on every other capture, test on the held-out one (generalisation to an unseen
   attack type or unseen normal capture).
4. **Unseen capture** - `holdout/arp_normal_holdout_2.pcapng` (1,457 evening packets), never part of training.

Same-network hold-out:

| Detector | Accuracy | Precision | Recall | F1 | FPR | ROC AUC | MCC |
|---|---|---|---|---|---|---|---|
| Naive Bayes | 0.831 | 0.998 | 0.816 | 0.898 | 0.018 | 0.911 | 0.522 |
| Isolation Forest | 0.568 | 1.000 | 0.525 | 0.689 | 0.000 | 0.897 | 0.300 |
| Ensemble (OR) | 0.831 | 0.998 | 0.816 | 0.898 | 0.018 | n/a | 0.522 |

All-data hold-out: NB accuracy 0.968 / recall 0.872 / FPR 0.001 / AUC 0.981; Isolation Forest recall 0.536 / FPR 0.036;
ensemble accuracy 0.941 / precision 0.886 / recall 0.872 / FPR 0.036.

Leave-one-capture-out, ensemble: flooding 0.94, unicast 0.91, race condition 0.56 (9 packets), basic spoof 0.48,
gratuitous 0.43; false-positive rate on held-out normal captures 0.03 (192.168, 187 pkts), 0.00 / 0.00 / 0.02 (the three
172.16 captures).
**Unseen normal capture** (`holdout/arp_normal_holdout_2.pcapng`, never trained on): NB flags 2.6%, IF 0.1%, ensemble
**2.7% false-positive rate**.

**Honest reading:**
- Recall is 82-87% on the hold-outs and 91-94% on cleanly-signatured attacks never seen in training. Precision is 99.8% on
  the same-network view and 89% on the all-data view.
- The ~45-50% on basic spoof and gratuitous is largely **label noise** (Section 4), not detector failure.
- **The false-positive rate on genuinely new traffic is the weak spot.** An earlier 1.4% came from one daytime capture; the
  evening capture gave 10.6% when training had ~6.2k real records, 5.3% after a 60-minute evening capture was added, and
  2.7% after cross-validated tuning (Section 8b). Normal traffic drifts with time of day; do not quote 1.4%.
- The 192.168 false-positive rate rests on only 56 test packets and a single 187-packet normal capture on that network.
- Earlier versions of this project reported AUC 0.954 on a leaky random split; removing the leakage is why the numbers moved.
  This is exactly the kind of effect a plain random train/test split hides.

Accuracy alone would mislead on imbalanced data, so precision/recall/F1/FPR/AUC/MCC are reported.

## 8b. Tuning, tracking and tests (training you can repeat)

An experiment answers "does it work once". Training you can repeat also needs settings that were chosen on purpose,
results you can trace, and checks that fail when something breaks. The project has four pieces:

1. **Hyper-parameter tuning** (`model_scripts/tune.py`, output `results/tuning.md`, settings in `config.json`).
   Each candidate setting is scored by **leave-one-capture-out cross-validation on the training pool only**: hold one whole
   capture out, train on the rest, and measure what share of the held-out capture is flagged (recall for an attack capture,
   false-alarm rate for a normal one). Score = mean attack recall - mean normal false-alarm rate, averaged per capture so a
   big capture cannot dominate. A setting changes only if a candidate beats the current value by more than 0.005. The unseen
   evening capture is never used, so it stays an honest final test. Result: Isolation Forest `contamination` 0.02 -> 0.005
   (score 0.640 -> 0.653), Random Forest `min_samples_leaf` 1 -> 5, Logistic Regression `C` 1 -> 10; Naive Bayes smoothing unchanged.
   *Effect:* unseen-capture false alarms 5.3% -> 2.7%; *cost:* ensemble same-network recall 0.862 -> 0.816, and the Isolation
   Forest now adds little beyond Naive Bayes on this data.
2. **Run tracking** (`model_scripts/runlog.py`). Every `train.py` run writes `results/runs/<id>/` with the settings, metrics,
   a SHA-256 hash of every capture file and the git commit. `models/metadata.json` stores the run id, so any saved model can be
   traced back to its exact data and settings.
3. **Automated tests** (`tests/`, run with `python -m pytest`; 28 tests, ~25 s). They check that features are causal (adding later
   packets never changes earlier rows) and deterministic; that the pcap and live paths compute identical features; that batch and
   incremental extraction give identical rows; that the chronological split never puts a training packet after a test packet;
   that synthetic rows are labelled and stay out of test data; that training with the same seed gives identical predictions; that
   saved models reload identically; that the simulator's ground truth is exact; and that the alert rule stays quiet on noise and
   busy hosts. A deliberately leaky random split makes the split test fail, so the tests do detect leakage.
4. **Pinned environment** (`requirements.txt`). Exact versions, because pickled scikit-learn models only load reliably with the
   version that trained them.

## 8c. Model comparison (Logistic Regression and Random Forest)

`python model_scripts/compare_models.py` runs the same real-packet tests for the two extra models (settings from `config.json`):

| Test | Naive Bayes | Logistic Regression | Random Forest |
|---|---|---|---|
| All-data hold-out: recall / FPR / MCC | 0.872 / 0.001 / 0.914 | 0.869 / 0.011 / 0.888 | **0.949 / 0.007 / 0.951** |
| Same-network 192.168: recall / FPR | 0.816 / 0.018 | 0.860 / 0.018 | 0.918 / **0.071** |
| Unseen evening capture: FPR | **0.030** | 0.038 | 0.031 |
| Leave-one-out, normal 192.168 capture: FPR | **0.03** | 0.42 | 0.32 |
| Leave-one-out recall: flooding / unicast / basic / gratuitous | 0.94 / 0.91 / 0.48 / 0.43 | 1.00 / 0.66 / 0.58 / 0.59 | 1.00 / 0.91 / 0.94 / 0.84 |

Random Forest has the best recall, but it flags 7.1% of normal packets in the same-network test and 32% of the lab-LAN normal
capture in leave-one-out, so part of its gain may be recognising the network rather than the attack; with one 187-packet normal
capture on the attack network we cannot tell. Naive Bayes stays the shipped model because it is the most stable.

## 9. Inference

`python model_scripts/predict.py <capture.pcapng> [--label 0|1]` extracts the same features via `features.py`,
runs both models, ORs them, prints the flagged share per detector (and FPR/recall when `--label` is given), and writes
per-packet verdicts to `results/prediction_output.csv`. The old demo on a training pcap was misleading; use a capture
from `pcap_files/holdout/` for an unbiased demonstration (e.g. `holdout/arp_normal_holdout_2.pcapng`: 2.7% flagged).

## 9b. The simulator web app

`streamlit run simulator/app.py` (details and demo script: README and `docs/SIMULATOR_DEMO_GUIDE.md`).

- **What it is.** A traffic generator and live dashboard, not a replay of the dataset. In the default mode it captures the ARP
  packets of the real network passively (ARP filter only, in memory, nothing saved or transmitted), runs the detectors on them,
  and lets you inject simulated attacks into that stream with buttons. Other modes: a virtual LAN (works offline), a scripted
  run, and a replay of recorded captures.
- **What is real and what is simulated.** Background traffic is real (captured live). Injected attacks are simulated forged
  packets built from real hosts seen so far, added to the in-memory stream only. The detectors were trained on the recorded
  dataset. Because the attacks are simulated and cleaner than the lab ones, simulator scores are illustrations, not accuracy
  evidence; the accuracy numbers come from the recorded captures (Section 8).
- **Detectors shown.** Only the three shipped detectors: Naive Bayes, Isolation Forest and the ensemble. Logistic Regression
  and Random Forest were removed from the simulator because their cards over-alerted on a network far busier than their training
  data (campus Wi-Fi at ~100-200 ARP packets/s vs a training maximum of 64/s) and confused the demo; they are still compared
  offline in `model_scripts/compare_models.py`. A "busy network" banner appears above ~33 packets/s (99.9th percentile of
  normal training traffic), because the shipped detectors may also be less reliable there.
- **Alert rule** (`model_scripts/alerting.py`): ALERT needs a sender with at least 3 flagged packets and over half of its packets
  flagged in 10 s; "stray flags" are flagged packets that do not look like an attacker (noise, or a busy host such as a scanner
  with ~2% of its packets flagged). A single flagged packet is not an alert. In tests all five attack types raised an ALERT on
  Naive Bayes/ensemble within 1-2 s.
- **Performance.** Feature extraction is incremental, so an update costs ~0.3 s even at ~200 packets/s, and a session can run
  60,000 packets.

## 10. Files

```
Base Paper/                 the base paper PDF (Singhal and Kumar, 2026)
reference_papers/           other papers read + index (README.md); includes the earlier Ali et al. 2025 preprint
Presentation/               project slides and the design reference deck
pcap_files/                 raw captures (top level = training data; holdout/ = unseen test data)
dataset/                    real: arp_dataset_raw/features/holdout.csv; SYNTHETIC: arp_dataset_synthetic.csv, arp_dataset_balanced.csv
model_scripts/features.py   shared feature extraction (FeatureExtractor: incremental, same rows as batch)
model_scripts/augment.py    labelled synthetic (SMOTE-style) oversampling
model_scripts/train.py      dataset build, training, evaluation, report, run record
model_scripts/tune.py       leave-one-capture-out hyper-parameter selection
model_scripts/compare_models.py   Logistic Regression / Random Forest vs Naive Bayes
model_scripts/predict.py    ensemble inference on any capture
model_scripts/simulate.py   virtual LAN, live capture and attack injection for the simulator
model_scripts/alerting.py   when a detector raises an ALERT in the simulator (repeated flags from one sender)
model_scripts/config.py, model_factory.py   settings loader and model builders (one place for every setting)
model_scripts/runlog.py     run tracking (data SHA-256, settings, git commit)
simulator/app.py            Streamlit web app
config.json                 every training setting (seed, split, contamination, RF/LR parameters), written by tune.py
models/saved_models/        naive_bayes_model.pkl, isolation_forest_model.pkl (joblib)
models/metadata.json        feature list, sklearn version, settings, training captures, run id, data hash, git commit
results/                    metrics.md/json, model_comparison.md/json, tuning.md/json, evaluation.png, prediction_output.csv, runs/<id>/
tests/                      28 pytest tests (leakage, determinism, round-trip, incremental extraction, alert rule, simulator ground truth)
docs/                       RECORDING_NORMAL_TRAFFIC.md (Wireshark guide), SIMULATOR_DEMO_GUIDE.md (how to demo)
requirements.txt            pinned versions
```

`joblib` is used instead of `pickle` because it handles the large numpy arrays inside sklearn models efficiently.
There are no label encoders any more: the models take only numeric behavioural features.

## 11. Likely teacher questions

**What is your base paper, and what did you do better?** Singhal and Kumar (2026, Frontiers in Big Data), a cost-sensitive Random
Forest for ARP spoofing. Its headline score comes from splitting one public dataset into train and test, plus one extra check on
a second public dataset. Both datasets are ready-made, and neither is traffic the authors recorded or an attack type held out of
training. We tested on traffic we recorded, on recordings and attack types held out of training, on a separate evening recording,
and on live traffic. We claim stronger testing, not higher accuracy: the data and task differ, so the numbers are not comparable.

**Why did you change the base paper?** To use a peer-reviewed journal paper newer than November 2025 that is directly about ARP
spoofing and whose evaluation design leaves the gap above.

**Did the paper do anything better than you?** Yes: a much larger dataset, higher headline scores, a cost-sensitive threshold that
caps missed attacks at 0.5%, and a cross-dataset check. We did not do the cost-sensitive threshold; it is our first future-work item.

**Did you test on a completely different network?** No. The held-out evening capture is the same campus Wi-Fi at a different time.
The only different-network test is the lab-LAN normal capture held out in leave-one-out (3% false alarms), and the lab network could
not be revisited. We list this as a limitation.

**Why Naive Bayes?** Fast, simple, little data needed, interpretable, and the steadiest across networks (Section 8c). We also
trained Logistic Regression and Random Forest to compare; Random Forest has the best recall but may be recognising the network.

**What does "naive" mean?** Features are assumed conditionally independent given the class — false here (rate and count
features correlate), which mainly shows up as overconfident probabilities.

**Why train Isolation Forest only on normal data?** It models "normal" and flags deviations, so it can catch unseen attack
types; training on both would turn it into a worse classifier. Honest note: after tuning it adds little beyond Naive Bayes on our
data (it only helps on flooding).

**How did you choose the settings?** By leave-one-capture-out cross-validation on the training pool only (Section 8b); the unseen
capture was never used. It changed Isolation Forest `contamination` from 0.02 to 0.005, which cut unseen false alarms from 5.3% to
2.7% but lowered the ensemble's same-network recall from 0.862 to 0.816.

**How are labels assigned?** Per capture file (Section 4) — weak labels, with measured noise (~half of two attack captures
is benign-looking). Per-packet ground truth would be the fix.

**Why is recall below precision?** Mostly the label noise above plus first-packet-of-a-burst effects; the cleanly-signatured
attacks reach 91-94% in leave-one-capture-out.

**Did Scapy detect the attack?** No — it only decodes packets. All detection logic is custom Python.

**Most important features?** `mac_conflict`, `ip_mac_count` and `l2_ip_count` — direct on-wire evidence of one identity
claiming another (conflict, repeated MAC changes, one sender claiming many IPs). Limit: a slow, careful spoofer that never
changes the claimed MAC for an IP and sends at a low rate would evade the rate and count features.

**Why did you not use `eth_arp_mac_mismatch` / `is_unsolicited_reply`?** Both are computed but excluded. The mismatch is an
artifact of the attack tool (forged frames had Ethernet src 00:00:00:00:00:00), and the unsolicited-reply share depends on
the sniffer's viewpoint (0% to 64% in normal captures). Dropping them did not hurt recall and cut false positives on a new network.

**Do the results hold on another network?** Partly tested: an unseen 1,457-packet evening capture from campus Wi-Fi gives a
2.7% false-positive rate (a daytime one gave 1.4% earlier; the evening one was 10.6% at first, 5.3% after more evening data, 2.7% after tuning).
All attacks and one normal capture are from the lab LAN, so a model could partly learn the network; that is why same-network
results are reported separately.

**How big is the dataset, and is any of it synthetic?** 9,243 real recorded packets (7,786 training + 1,457 held-out). A further
3,980 rows are synthetic (SMOTE-style), stored in separate labelled files, never used in any test, and not counted in the 9,243.
They did not improve results, so the shipped models use real data only.

**Why did you not just add more data to reach 8-10k?** We did, for normal traffic (real recordings). Attack traffic can only be
recorded in the lab with the attack scripts; generating it on a shared network is not acceptable. Oversampling the attack class
synthetically was tried and measured, and it added balance but no information.

**Why did you remove the IP/MAC columns from the model?** With label encoding they imposed a fake order, let the model
memorise which capture a packet came from (attack captures contain 1108 distinct source MACs vs 7 in normal), and made
inference crash on any new address. Behavioural features generalise to new networks.

**Why a chronological split, not random?** Packets in a burst are near-duplicates; a random split puts twins in train and
test and inflates scores (our own earlier leaky random split gave AUC 0.954). Chronological + leave-one-capture-out is stricter.

**Why OR for the ensemble?** Security trade-off: false negatives cost more than false alarms. Honest note: after tuning, the
ensemble's recall equals Naive Bayes alone on the all-data test and it adds ~60 false alarms, so the OR rule currently buys little.

**Is the dataset balanced?** No: the real data is ~3:1 normal (5,883 vs 1,903). A synthetic-balanced file exists but did not
improve results, so accuracy is not used alone; precision/recall/FPR/MCC are reported.

**What is the simulator, and is it real?** A live dashboard. The background traffic is real ARP traffic captured passively
from the network; the attacks are simulated packets injected into the in-memory stream by buttons. Nothing is transmitted on the
network. Its scores are illustrations; the accuracy evidence is the recorded-capture results.

**Why not attack the real network?** It would poison other students' ARP caches on a shared network. Attacks are only injected
into the program's memory.

**Why doesn't the simulator show Logistic Regression and Random Forest?** On a network far busier than the training data they
flagged every packet, which confused the demo. They are compared offline (`compare_models.py`, Section 8), and the simulator
shows only the shipped detectors, with a busy-network warning.

**What does an ALERT mean in the simulator?** A sender had at least 3 flagged packets and over half of its recent packets flagged
in 10 seconds. A single flagged packet is only a "stray flag".

**Limitations?** Only 187 normal packets on the attack network, small attack captures (1,903 packets, 9 in one capture), a lab
network that could not be revisited, normal captures mostly from a different network; unseen evening false-positive rate 2.7%;
weak labels; GaussianNB probabilities are overconfident; forging both MAC fields defeats one feature; the Isolation Forest adds
little; results are lab-scale, not production-validated.

**Is there deep learning here?** No — deliberately classical. See the future work in the README.

## 12. Limitations and next steps

1. **Cost-sensitive, false-negative-constrained threshold** (the base paper's idea): choose the decision threshold on validation data
   so that missed attacks stay under a set limit; our basic-spoof recall of 0.48 is the kind of weakness this targets.
2. **Per-packet ground truth:** identify the attacking host's MAC for each attack capture and label by sender, removing the
   file-level label noise that caps Naive Bayes recall on basic-spoof/gratuitous at ~50%.
3. **Normal and attack traffic from more networks** (the lab network could not be revisited), to separate attack signal from
   network fingerprints and settle Random Forest vs Naive Bayes.
4. Calibrate probabilities (`CalibratedClassifierCV`), add confidence intervals to the small same-network test, and report
   feature drift on new captures.
5. Deep models from the literature (DNN, LSTM/1D-CNN on packet windows, autoencoders) if the data grows.
