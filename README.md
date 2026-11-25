# ARP Spoofing Detection — Machine Learning Pipeline

A research-paper-based ML lab project: detect ARP spoofing attacks from captured network traffic, and test the idea
more strictly than the base paper did. (See [`EXPLANATION.md`](EXPLANATION.md) for the full walkthrough and viva Q&A.)

Authors: Garv Noor (1024170189), Manya Gupta (1024170192).

## Research context: base paper and our research area

**Base paper** (in [`Base Paper/`](Base%20Paper/)): Singhal and Kumar, *"A cost-sensitive random forest framework for ARP
spoofing detection in Internet of Medical Things networks"*, Frontiers in Big Data 9 (2026), DOI
[10.3389/fdata.2026.1878242](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2026.1878242/full) —
a peer-reviewed, open-access journal article published 22 July 2026.

What it does: a cost-sensitive Random Forest (SH-IORF) on the public CICIoMT2024 dataset (347,721 network *flows*, 47
features, from a testbed of real and simulated devices) plus 18 added timing features (52 kept after pruning); one stratified
70/15/15 train/validation/test split, class weights and the decision threshold tuned on validation only; reported 99.90%
accuracy and 0.17% missed attacks; one extra check on a second public dataset (CICIDS2017, 99.32%). Its authors state it was
not validated on real devices.

**The gap — and the reason for this project (our research area).** Their headline score comes from splitting one public
dataset into train and test. They added one check on a second public dataset, which is good. But both datasets are
ready-made, and neither is traffic they recorded or an attack type held out of training. We test ARP-spoofing detection
**beyond a train/test split**: on traffic we recorded ourselves, on recordings and whole attack types held out of training, on a
separate evening recording, and on live traffic.

| | Base paper | This project |
|---|---|---|
| Data | public benchmark: 347,721 network flows | **9,243 real ARP packets we recorded** (Wi-Fi traffic + lab attacks) |
| Attacks | one class (ARP spoofing), no per-type results | **5 ARP attack types**, each tested on its own |
| Features | 47 flow features + 18 timing, pruned to 52 | 10 per-packet ARP features, no raw IP/MAC |
| Testing | one 70/15/15 split + a second public dataset | chronological split, same-network test, **leave-one-capture-out**, **unseen capture**; settings tuned by cross-validation |
| Demo | offline only | **live simulator** on real traffic |

**Not directly comparable.** The data and the task differ (flows vs. packets), so we claim *stronger testing*, not higher
accuracy. Their data is much larger and their scores are higher; their cost-sensitive, false-negative-constrained threshold is
something we did not do (see Future work). Our "unseen" evening capture is the same campus Wi-Fi at a different time, not a
different network, and the lab network could not be revisited (see Limitations).

Other papers we read are indexed in [`reference_papers/README.md`](reference_papers/README.md); the earlier base paper
(Ali et al., 2025, arXiv preprint) is kept there for reference.

## Problem

ARP (Address Resolution Protocol) has no authentication, so any device on a LAN can send
a forged ARP reply claiming an IP that belongs to someone else. This is the basis of
man-in-the-middle attacks, ARP flooding, gratuitous-ARP cache poisoning and race-condition
attacks. This project detects such attacks from captured traffic.

## Data

**9,243 real recorded ARP packets** (7,786 in the training pool + 1,457 in a never-trained-on hold-out), plus
**3,980 clearly-labelled synthetic rows** that are kept in separate files and are *not* counted as recorded data.

**Labels come from the file name:** a name containing `normal` -> 0, anything else -> 1. To add data, drop a
new capture into `pcap_files/` (see [`docs/RECORDING_NORMAL_TRAFFIC.md`](docs/RECORDING_NORMAL_TRAFFIC.md)).
Captures in `pcap_files/holdout/` are ignored by training and used only as unseen test data.

| File | Label | Network | ARP packets |
|---|---|---|---|
| `arp_normal_traffic.pcapng` | Normal (0) | 192.168.x.x (same LAN as the attacks) | 187 |
| `new_File for normal_traffic.pcapng` | Normal (0) | 172.16.x.x (campus Wi-Fi) | 1,149 |
| `arp_normal_campus_wifi_02.pcapng` | Normal (0) | 172.16.x.x | 1,530 |
| `arp_normal_campus_wifi_03.pcapng` | Normal (0) | 172.16.x.x (60-min evening capture) | 3,017 |
| `arp_spoof_basic`, `arp_request_flooding`, `arp_gratuitous_announcement`, `arp_reply_race_condition`, `arp_spoofing_unicast_reply` (`.pcapng`) | Attack (1) | 192.168.x.x | 213 / 1,348 / 182 / 9 / 151 |
| **Training pool** | 5,883 normal / 1,903 attack | | **7,786** |
| `holdout/arp_normal_holdout_2.pcapng` | Normal (0), **never trained on** | 172.16.x.x (evening) | 1,457 |

The three newest normal captures were recorded passively on campus Wi-Fi with Wireshark's `dumpcap` (`-f arp`);
the five attack captures are the original lab recordings (real packets from lab-staged attacks). No attack traffic was
generated on a shared network.

`train.py` extracts every ARP packet and writes:

| File | Contents |
|---|---|
| `dataset/arp_dataset_raw.csv` | **real** packets, human-readable IPs/MACs, diagnostics, network tag |
| `dataset/arp_dataset_features.csv` | **real** packets, exactly the model features |
| `dataset/arp_dataset_holdout.csv` | **real** unseen hold-out packets (not trained on) |
| `dataset/arp_dataset_synthetic.csv` | **SYNTHETIC** rows only (`synthetic=1`, `capture="synthetic<-…"`) |
| `dataset/arp_dataset_balanced.csv` | real + synthetic, with a `synthetic` column (0 = recorded, 1 = generated) |

> **Privacy.** The raw captures and `arp_dataset_raw.csv` contain real MAC and IP addresses of other people's devices on
> campus Wi-Fi. Anonymise them before sharing or publishing the repository.

### Synthetic augmentation — what it is and what it is not

The recorded data is 3:1 normal-heavy (5,883 vs 1,903), and attack traffic can only be recorded in the lab. To give a
balanced file, `model_scripts/augment.py` creates **3,980 synthetic attack rows** with SMOTE-style interpolation
(a new row lies between a real attack packet and one of its 5 nearest neighbours *from the same capture*; binary and
count columns are rounded back to valid values). The synthetic budget is spread over the four attack captures that have
≥30 packets (the 9-packet race-condition capture is not oversampled).

- **They are synthetic, not recorded.** Every row carries `synthetic=1` and lives in its own files. They must not be
  reported as captured traffic, and the real record count above does not include them.
- **They add balance, not information** — each row is an interpolation of packets we already have.
- **They never touch a test set.** Inside every evaluation, synthetic rows are generated from the training part only;
  every number in this README is computed on **real** packets only.
- **They did not help.** The ablation (below and in `results/metrics.md`) shows identical or marginally lower recall with
  augmentation, and on one leave-one-out capture (`arp_normal_campus_wifi_03`) the false-positive rate rose from 0.02 to
  0.13. The shipped models are therefore trained on **real packets only**; `python model_scripts/train.py --augment`
  trains them with the synthetic rows instead.

> **Important caveat — the networks differ.** All attack captures and the first normal capture were recorded on one
> LAN (192.168.x.x); the newer normal captures come from a different network (172.16.x.x). A model can partly learn
> "which network" instead of "attack", so results are also reported **inside the 192.168 network only**. The lab network
> could not be revisited, so no further normal traffic could be recorded on it.

## Features (`model_scripts/features.py`, shared by training, prediction and the simulator)

Computed per capture file with Scapy + custom Python. **No raw IP/MAC identities are model inputs**
(they let the model memorise which capture a row came from, and crashed on unseen addresses).

| Feature | Meaning |
|---|---|
| `op` | 1 = request, 2 = reply |
| `is_broadcast` | Ethernet destination is `ff:ff:ff:ff:ff:ff` |
| `is_gratuitous` | sender IP == target IP (announcement) |
| `mac_conflict` | this IP was earlier seen with a different MAC |
| `ip_mac_count` | distinct MACs ever claimed for this sender IP |
| `l2_ip_count` | distinct IPs claimed by this Ethernet sender |
| `l2_reply_rate_1s` | replies from this Ethernet sender in the last second |
| `arp_reply_count` | running count of replies sent by this IP |
| `log_interarrival_ms` | log(1 + gap to previous packet in the same capture, ms) |
| `log_reply_delay_ms` | log(1 + request->reply delay, ms), 0 if unmatched |

Two columns are computed but **deliberately not used as inputs**: `is_unsolicited_reply` (depends on how many replies to
*other* hosts the sniffer happens to see: 0% of replies in one normal capture, 53-64% in others) and
`eth_arp_mac_mismatch` (in the attack captures it is an artifact of the attack tool - forged frames carry Ethernet source
`00:00:00:00:00:00`). Removing them did not reduce attack recall and made the model far steadier on new networks.

## Models

1. **Gaussian Naive Bayes** (supervised) — normal vs. attack.
2. **Isolation Forest** (unsupervised, `contamination=0.005`, tuned) — trained on *normal packets only*; flags deviations.
3. **Ensemble** — attack if *either* model flags it (OR logic; missing an attack costs more than a false alarm).

Logistic Regression and Random Forest are trained for comparison only (see *Model comparison*). Models are saved in
`models/saved_models/`; `models/metadata.json` records the feature list, sklearn version, settings, data hash, git commit and
the run that produced them. All settings live in `config.json` (see *Training lifecycle*).

## Evaluation

All numbers are auto-generated into [`results/metrics.md`](results/metrics.md) (plots in `results/evaluation.png`; runs are
deterministic). Chronological = first 70% of every capture trains, last 30% tests. **Real packets only.**

**1. Same-network hold-out (192.168.x.x only) — the fairest view.** Test: 56 normal / 571 attack.

| Detector | Accuracy | Precision | Recall | F1 | FPR | ROC AUC | MCC |
|---|---|---|---|---|---|---|---|
| Naive Bayes | 0.831 | 0.998 | 0.816 | 0.898 | 0.018 | 0.911 | 0.522 |
| Isolation Forest | 0.568 | 1.000 | 0.525 | 0.689 | 0.000 | 0.897 | 0.300 |
| **Ensemble (NB OR IF)** | 0.831 | 0.998 | 0.816 | 0.898 | 0.018 | n/a | 0.522 |

**2. All-data hold-out** (both networks; test: 1,765 normal / 571 attack).

| Detector | Accuracy | Precision | Recall | F1 | FPR | ROC AUC | MCC |
|---|---|---|---|---|---|---|---|
| Naive Bayes | 0.968 | 0.998 | 0.872 | 0.931 | 0.001 | 0.981 | 0.914 |
| Isolation Forest | 0.860 | 0.829 | 0.536 | 0.651 | 0.036 | 0.906 | 0.589 |
| Ensemble (NB OR IF) | 0.941 | 0.886 | 0.872 | 0.879 | 0.036 | n/a | 0.840 |

**3. Leave-one-capture-out** — train on every other capture, test on the held-out one (share flagged as attack):

| Held-out capture | Class | NB | IF | Ensemble |
|---|---|---|---|---|
| request flooding | attack | 0.94 | 0.81 | **0.94** |
| unicast reply spoof | attack | 0.91 | 0.00 | **0.91** |
| reply race condition (9 pkts) | attack | 0.56 | 0.00 | 0.56 |
| basic spoof | attack | 0.48 | 0.00 | 0.48 |
| gratuitous announcement | attack | 0.43 | 0.00 | 0.43 |
| normal 192.168 (187 pkts) — false-positive rate | normal | 0.03 | 0.00 | 0.03 |
| normal 172.16 `new_File` (1,149) — FPR | normal | 0.00 | 0.00 | 0.00 |
| normal 172.16 `wifi_02` (1,530) — FPR | normal | 0.00 | 0.00 | 0.00 |
| normal 172.16 `wifi_03` (3,017) — FPR | normal | 0.01 | 0.02 | 0.02 |

**4. Unseen normal capture** (`holdout/arp_normal_holdout_2.pcapng`, 1,457 packets, recorded in the evening, never
trained on): NB flags 2.6%, Isolation Forest 0.1%, ensemble **2.7% false-positive rate**.

**Reading this honestly**

- **Unseen false-positive rate: 10.6% -> 5.3% -> 2.7%.** An earlier 1.4% came from a single daytime capture, so it
  should not be quoted. On the evening capture the first model scored 10.6%; adding a 60-minute evening capture to training
  brought it to 5.3%; cross-validated tuning of the Isolation Forest (`contamination` 0.02 -> 0.005) brought it to 2.7%.
  Traffic shifts with time of day, so this number should be re-checked on every new capture.
- **Trade-off from tuning.** The ensemble's same-network recall fell from 0.862 to 0.816 (same false-positive rate 0.018),
  because the less trigger-happy Isolation Forest no longer adds the packets it used to. On the all-data hold-out the
  ensemble now has the same recall as Naive Bayes alone (0.872) and ~60 extra false alarms (precision 0.886 vs 0.998), so on
  this data **the Isolation Forest adds little beyond Naive Bayes** (it only contributes on flooding, recall 0.81).
- Precision is ~99.8% on the same-network view; on the larger all-data view the ensemble's precision is 89%.
- Attacks with a clear on-wire signature (flooding, unicast spoof) are caught 91-94% of the time on a capture the model
  never saw; recall is ~82-87% on the hold-outs.
- **Known limitation — file-level labels.** Every packet in an attack capture is labelled "attack", including ordinary
  background ARP requests. In the basic-spoof and gratuitous captures about half the packets (47% / 51%) are plain requests
  with no attack signal, which is why recall on those two captures is ~45-50% with Naive Bayes. That is largely label noise,
  not a detector failure (Random Forest does better on them, so it is *partly* noise); fixing it needs per-packet ground truth
  (e.g. the attacker's MAC for each capture).
- **Known limitation — only one normal capture on the attack network** (187 packets), so the 192.168 false-positive rate
  rests on 56 test packets (1 of 56 flagged). The lab network could not be revisited to record more.
- **Known limitation — attack data is small** (1,903 packets from five lab captures, 9 in one of them) and normal data is
  mostly from a different network.
- Naive Bayes log-loss is high: its probabilities are overconfident (independence assumption violated), so use its
  predicted class / ranking, not its probabilities.

### Model comparison: Logistic Regression and Random Forest

`python model_scripts/compare_models.py` (after `train.py`) scores both against Naive Bayes on the same real-packet tests;
full tables in [`results/model_comparison.md`](results/model_comparison.md). Their settings come from `config.json`
(tuned by cross-validation); the saved models are unchanged.

| Test | Naive Bayes | Logistic Regression | Random Forest |
|---|---|---|---|
| All-data hold-out: recall / FPR / MCC | 0.872 / 0.001 / 0.914 | 0.869 / 0.011 / 0.888 | **0.949 / 0.007 / 0.951** |
| Same-network 192.168: recall / FPR | 0.816 / 0.018 | 0.860 / 0.018 | 0.918 / **0.071** |
| Unseen evening capture: FPR | **0.030** | 0.038 | 0.031 |
| Leave-one-out, normal 192.168 capture: FPR | **0.03** | 0.42 | 0.32 |
| Leave-one-out recall: flooding / unicast / basic / gratuitous | 0.94 / 0.91 / 0.48 / 0.43 | 1.00 / 0.66 / 0.58 / 0.59 | 1.00 / 0.91 / 0.94 / 0.84 |

- **Random Forest is the strongest on the all-data test** and finds many more basic-spoof and gratuitous
  packets than Naive Bayes (0.94 / 0.84 vs 0.48 / 0.43).
- **But it is the worst on the same-network view** (4 of 56 normal packets flagged, and 32% of the only normal capture
  recorded on the attack LAN in leave-one-out). Those are the tests least affected by "which network is this?", so part of
  its gain may be network fingerprinting. With one 187-packet normal capture on the attack network we cannot tell the two
  apart.
- **Logistic Regression** is no better than Naive Bayes overall and flags 42% of the 192.168 normal capture when it is left out.
- Both extra models use `class_weight='balanced'`; unseen-capture numbers here train on the whole pool, so Naive Bayes shows
  3.0% instead of the 2.6% from `train.py` (which trains on the first 70% only).
- Decision: Naive Bayes stays the shipped model because it is the most stable across networks.

### Effect of the synthetic rows (real test packets)

| View | Detector | Recall (synthetic / real-only) | FPR (synthetic / real-only) |
|---|---|---|---|
| all data hold-out | Naive Bayes | 0.872 / 0.872 | 0.001 / 0.001 |
| all data hold-out | Ensemble | 0.872 / 0.872 | 0.036 / 0.036 |
| same-network 192.168 | Naive Bayes | 0.792 / 0.816 | 0.018 / 0.018 |
| same-network 192.168 | Ensemble | 0.792 / 0.816 | 0.018 / 0.018 |
| unseen evening capture | Ensemble | — | 0.029 / 0.027 |
| leave-one-out `wifi_03` | Ensemble | — | 0.13 / 0.02 |

## Training lifecycle: tuning, tracking, tests

Turning the experiment into repeatable training:

| Piece | What it does | Where |
|---|---|---|
| **Config** | One place for every setting (seed, split, Isolation Forest `contamination`, Naive Bayes smoothing, Random Forest, Logistic Regression). `train.py`, `compare_models.py`, `tune.py` and the app all read it. | `config.json`, `model_scripts/config.py`, `model_scripts/model_factory.py` |
| **Tuning** | Leave-one-capture-out cross-validation on the **training pool only** (the unseen capture is never used). Score = mean recall on held-out attack captures minus mean false-alarm rate on held-out normal captures. A setting only changes if a candidate beats the current one by more than 0.005. | `model_scripts/tune.py` -> `results/tuning.md` |
| **Run tracking** | Every training run writes `results/runs/<id>/` with the settings, metrics, SHA-256 of every capture and the git commit; `models/metadata.json` points to its run. | `model_scripts/runlog.py` |
| **Tests** | 28 pytest tests: features are causal and deterministic, pcap and live paths agree, batch and incremental feature extraction are identical, the chronological split never lets a training packet follow a test packet, synthetic rows stay out of test data, training is deterministic, models reload identically, the simulator's ground truth is exact, and the alert rule stays quiet on noise and busy hosts. | `tests/` |
| **Pinned versions** | Exact library versions, because the pickled models only load with the scikit-learn version that trained them. | `requirements.txt` |

What tuning changed (full table in `results/tuning.md`): Isolation Forest `contamination` 0.02 -> 0.005 (CV score 0.640 ->
0.653), Random Forest `min_samples_leaf` 1 -> 5, Logistic Regression `C` 1 -> 10; Naive Bayes smoothing stayed at its
default. Effect on the unseen evening capture: false alarms 5.3% -> 2.7%; cost: ensemble same-network recall 0.862 -> 0.816.

## Inference

```bash
python model_scripts/predict.py path/to/capture.pcapng            # prints flagged share per detector
python model_scripts/predict.py capture.pcapng --label 0          # you know it's normal -> prints FPR
```

Writes per-packet NB / IF / ensemble verdicts to `results/prediction_output.csv`. Unseen IPs/MACs are fine (no encoders).
For an unbiased demo use a capture not used for training (`pcap_files/holdout/`).

## Simulator web app

```bash
streamlit run simulator/app.py
```

Four modes (sidebar):

- **REAL live traffic + injected attacks (default):** captures your network's ARP traffic live and read-only
  (Scapy + Npcap, ARP filter, in memory only, nothing saved) and feeds it to the detectors packet by packet. When you press a
  launch button, forged packets are built from real hosts seen so far (e.g. a forged reply claiming the real gateway's IP) and
  **added to the in-memory stream only - nothing is transmitted on the network**. Real packets are assumed normal, so alarms
  before your first injection are false alarms on real traffic. Real MAC/IP addresses are masked on screen by default.
  Needs Npcap/Wireshark and permission to capture on the chosen interface.
- **Virtual LAN: launch attacks manually:** a *virtual* LAN (`model_scripts/simulate.py`, nothing is sent to any network) runs
  normal traffic in real time; you press **Launch / Stop** for basic spoof, flooding, gratuitous, race condition and unicast
  spoof, several at once if you want. Works offline.
- **Scripted simulation:** the same traffic with the attack started automatically 30% into a fixed-length run.
- **Replay recorded capture:** scores any capture from `pcap_files/` (use `holdout/` for an unbiased demo).

In the live modes the page shows a live chart, a packet feed, per-launch detection times ("first flagged attack packet") and
the running overall result, updated every second. A session stops at 60,000 packets (press Start / Reset). How it behaves:

- **Detectors shown.** Only the three shipped detectors: Naive Bayes, Isolation Forest and the ensemble. Logistic Regression
  and Random Forest are not in the simulator (on a very busy campus Wi-Fi, ~100-200 ARP packets/s against a training maximum
  of 64/s, they flagged *every* packet and confused the demo); they are still compared offline with
  `python model_scripts/compare_models.py`.
- **Alert rule** (`model_scripts/alerting.py`): 🔴 ALERT needs a sender with at least 3 flagged packets *and* over half of its
  packets flagged in the last 10 s; 🟡 stray flags are flagged packets that don't look like an attacker (normal noise, or a
  busy host such as a subnet scanner with ~2% of its packets flagged); 🟢 quiet means nothing flagged. A single flagged
  packet is not an alert.
- **Busy-network warning** when the live ARP rate is well above the busiest normal traffic in training (99.9th percentile,
  ~33 packets/s).
- **Incremental processing**: `FeatureExtractor` keeps per-capture state and only processes new packets each second, so the
  cost per update stays flat (~0.3 s at ~200 packets/s) and results are identical to batch extraction (tested).
- Tabs (scripted/replay): per-model results, overall verdict next to the real-data benchmark, per-packet CSV download.
- The injected attacks are simulated and cleaner than the lab recordings, so the app labels scores as illustrations; quote the
  real-data tables above, not simulator scores. Demo script: [`docs/SIMULATOR_DEMO_GUIDE.md`](docs/SIMULATOR_DEMO_GUIDE.md).

## Project layout

| Path | What it is |
|---|---|
| `Base Paper/` | the base paper PDF (Singhal and Kumar, 2026) |
| `reference_papers/` | other papers read, with an index (`README.md`) |
| `Presentation/` | project slides (`ARP_Spoofing_Detection_v2.pptx`) and the design reference deck |
| `report/` | the UML501 project report (`ARP_Spoofing_Detection_Report.docx` and `.pdf`, 20 pages), its `figures/`, and the reference report it follows |
| `pcap_files/` | raw captures (top level = training data; `holdout/` = unseen) |
| `dataset/` | extracted datasets (real files and clearly-labelled synthetic files) |
| `model_scripts/` | `features.py`, `train.py`, `tune.py`, `compare_models.py`, `predict.py`, `augment.py`, `simulate.py`, `alerting.py`, `config.py`, `model_factory.py`, `runlog.py` |
| `simulator/app.py` | the Streamlit web app |
| `models/` | saved models and `metadata.json` |
| `results/` | `metrics.md/json`, `model_comparison.md/json`, `tuning.md/json`, `evaluation.png`, `runs/<id>/` |
| `tests/` | 28 pytest tests |
| `docs/` | recording guide and simulator demo guide |
| `config.json`, `requirements.txt`, `pytest.ini` | settings, pinned versions, test config |

## How to run

```bash
pip install -r requirements.txt
python model_scripts/train.py             # builds datasets, trains on real packets, evaluates, writes results/ + a run record
python model_scripts/train.py --augment   # same, but trains Naive Bayes with the synthetic rows
python model_scripts/tune.py              # re-select settings by cross-validation (updates config.json), then re-run train.py
python model_scripts/compare_models.py    # Logistic Regression / Random Forest vs Naive Bayes
python -m pytest                          # 28 automated tests
python model_scripts/predict.py pcap_files/holdout/<capture>.pcapng --label 0
streamlit run simulator/app.py            # the simulator
```

Scripts resolve paths from their own location, so they work from any directory.

## Limitations and future work

**Limitations (all stated in the results above):** attack data is small and recorded on one lab network that could not be
revisited; labels are per file, so about half of two attack captures look like plain requests; the same-network test has only
56 normal packets; the "unseen" evening capture is the same campus Wi-Fi at a different time; the Isolation Forest adds little
beyond Naive Bayes after tuning; Random Forest's higher recall may partly be network recognition; injected attacks in the
simulator are simulated, not real.

**Future work:**
1. **Cost-sensitive, false-negative-constrained threshold** — the base paper's idea: pick the decision threshold on a validation
   set so that missed attacks stay under a set limit (our basic-spoof recall of 0.48 is the kind of weakness this targets).
2. **Per-packet ground truth** (the attacker's MAC for each attack capture) to remove file-level label noise.
3. **Normal and attack traffic from more networks**, to separate attack signal from network fingerprints.
4. Ideas from the literature (Ali et al., 2025): a **DNN / MLP**, an **LSTM / 1D-CNN** on windows of packets, **autoencoders**
   trained on benign traffic, and **graph neural networks** on the IP-MAC graph.
