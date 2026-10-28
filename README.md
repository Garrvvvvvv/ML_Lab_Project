# ARP Spoofing Detection — Machine Learning Pipeline

This folder contains the **classical machine learning** portion of the ARP
Spoofing project, prepared for the ML lab submission.

## Problem

ARP (Address Resolution Protocol) has no built-in authentication, so any
device on a LAN can send a forged ARP reply claiming an IP address that
belongs to someone else. This is called **ARP spoofing** and is the basis of
man-in-the-middle attacks, ARP flooding, gratuitous-ARP cache poisoning, and
race-condition attacks. This project detects such attacks from captured
network traffic using machine learning.

## Data

Six `.pcapng` packet captures in [`pcap_files/`](pcap_files/), one normal
baseline and five attack variants:

| File | Label |
|---|---|
| `arp_normal_traffic.pcapng` | Normal (0) |
| `arp_spoof_basic.pcapng` | Attack (1) — basic spoof |
| `arp_request_flooding.pcapng` | Attack (1) — flooding |
| `arp_gratuitous_announcement.pcapng` | Attack (1) — gratuitous ARP |
| `arp_reply_race_condition.pcapng` | Attack (1) — race condition |
| `arp_spoofing_unicast_reply.pcapng` | Attack (1) — unicast spoof |

## Dataset

Running `train.py` extracts every ARP packet from the 6 pcaps above into a
single table and saves it to [`dataset/`](dataset/) in two forms:

- **`arp_dataset_raw.csv`** — one row per ARP packet, straight out of the
  pcap, with human-readable IPs/MACs. This is the dataset "as created."
- **`arp_dataset_encoded.csv`** — the same rows after label-encoding the
  IP/MAC columns to integers, which is exactly what gets fed into the
  models.

On the current pcaps this produces **2090 total rows** (187 normal, 1903
attack — the attack pcaps are longer/denser captures, which is why the
split isn't 50/50).

A real example from the raw dataset that shows *why* this works: the IP
`192.168.1.1` appears with MAC `54:47:e8:37:e2:44` in some packets and MAC
`5a:9d:7f:00:5a:2c` in others. One IP, two different physical addresses —
that is the literal definition of ARP spoofing, and it's exactly what the
`mac_conflict` feature is built to catch (it fires 927 times across this
dataset).

## Feature Engineering

Each ARP packet is parsed with **Scapy** and turned into a feature row:

- `src_ip`, `dst_ip`, `src_mac`, `dst_mac` — label-encoded identifiers
- `op` — ARP opcode (1 = request, 2 = reply)
- `is_broadcast` — whether the reply was broadcast instead of targeted
- `mac_conflict` — **1 if this source IP was previously seen with a
  different MAC address** (the core ARP-spoofing signature)
- `interarrival_time` — time since the previous packet
- `arp_reply_count` — running count of replies sent by this source
- `arp_reply_delay_ms` — time between a request and its matching reply

## Models

Two classical ML models, trained in [`model_scripts/train.py`](model_scripts/train.py):

1. **Gaussian Naive Bayes** (supervised) — trained on the full labeled
   feature set (normal + attack) to classify each packet as normal or
   spoofed.
2. **Isolation Forest** (unsupervised anomaly detection) — trained only on
   normal traffic, so it learns what "normal" ARP behavior looks like and
   flags anything that deviates as an anomaly.

Saved models live in [`models/saved_models/`](models/saved_models/), and the
label encoders for IP/MAC fields live in [`models/encoders/`](models/encoders/).

## Evaluation

`train.py` reports a full metrics suite on a held-out test split (30% of
the data, stratified by label). Latest run:

| Metric | Value |
|---|---|
| Accuracy | 0.6874 (68.74%) |
| Precision (Spoofed) | 0.9973 (99.73%) |
| Recall (Spoofed) | 0.6585 (65.85%) |
| F1 Score (Spoofed) | 0.7932 |
| ROC AUC | 0.9541 (95.41%) |
| Log Loss | 1.7561 |

Confusion matrix:

```
              Pred Normal   Pred Spoofed
Actual Normal        55            1
Actual Spoofed      195          376
```

Reading this honestly: **precision on the "Spoofed" class is near-perfect
(0.9973)** — when the model says "attack," it is right 99.7% of the time,
producing only 1 false alarm out of 56 normal packets. **Recall
is lower (0.6585)** — Naive Bayes misses some attack packets in isolation
(e.g., initial packets in an attack stream before conflict/delay stats accumulate).
However, **ROC-AUC of 0.9541** demonstrates superior overall discriminatory power.

## Ensemble / Inference

[`model_scripts/predict.py`](model_scripts/predict.py) loads both saved
models and runs them on a test packet capture (`arp_gratuitous_announcement.pcapng`).
Each model votes independently, and the final ensemble decision uses **logical OR logic**:
a packet is flagged as an attack if *either* the Naive Bayes classifier or the
Isolation Forest flags it.

Latest test run summary (`Summary Count`):
- **Attack Packets Detected (`1`)**: 116
- **Normal Packets (`0`)**: 66
- Total: 182 packets evaluated

### Why the Ensemble Works:
From the actual run logs:
```
src_ip  dst_ip  nb_prediction  if_prediction  final_prediction
2       0       0              1              1                   40
221     2       0              1              1                   40
```
Naive Bayes alone missed these 80 attack packets (`nb_prediction = 0`), but Isolation
Forest caught their timing/volume anomalies (`if_prediction = 1`), allowing the final
ensemble (`final_prediction = 1`) to correctly classify them as attacks.

Results are written to [`prediction_output.csv`](prediction_output.csv).

## How to run

Dependencies: `scapy`, `pandas`, `scikit-learn`, `joblib`.

```bash
pip install scapy pandas scikit-learn joblib

# from inside this folder
python model_scripts/train.py      # extracts dataset, saves it to dataset/, trains both models, prints metrics
python model_scripts/predict.py    # runs ensemble inference on a sample pcap, writes prediction_output.csv
```

## Future Roadmap: Deep Learning & Architectural Extensions

Drawing directly from our base paper (*"Intelligent ARP Spoofing Detection using Multi-layered Machine Learning Techniques for IoT Networks"*, Ali et al., 2025), the future evolution of this project focuses on deploying **Deep Learning (DL)** at the central aggregation tier to complement our lightweight edge models:

### 1. Sequence & Temporal Deep Learning (LSTM / Bi-LSTM / 1D-CNN)
- **Current Limitation**: Classical ML inspects packets individually in isolation (stateless evaluation).
- **Deep Learning Extension**: Network traffic is fundamentally a continuous time-series. We plan to process packet streams using sliding windows of length $W$ (e.g., $W = 50$ consecutive packets) fed into **Long Short-Term Memory (LSTM)** or **1D-CNN** networks.
- **Benefit**: The LSTM hidden state learns chronological dialogue patterns (`Request` $\to$ `Expected Latency` $\to$ `Single Matching Reply`). It instantly identifies temporal sequence violations (e.g., repeated unsolicited reply bursts without prior requests) that per-packet classifiers miss.

### 2. Deep Autoencoders for Zero-Day Threat Isolation
- **The Concept**: Inspired by deep hierarchical representation learning (*Shone et al.* cited in base paper), we plan to implement a **Deep Variational Autoencoder (VAE)**.
- **Implementation**: The autoencoder is trained strictly on massive volumes of benign, normal network traffic to compress and reconstruct normal packet flows with near-zero error.
- **Benefit**: When an unknown, zero-day spoofing variant or stealthy low-rate attack passes through the network, the model cannot reconstruct the unseen pattern, causing the **reconstruction loss to spike**. This enables autonomous anomaly detection without requiring prior attack labels or signatures.

### 3. Graph Neural Networks (GNNs) on LAN Topologies
- **The Concept**: An ARP network is inherently a dynamic bipartite graph $\mathcal{G} = (\mathcal{V}_{\text{IP}}, \mathcal{V}_{\text{MAC}}, \mathcal{E})$, where IP addresses and physical MAC addresses are nodes, and ARP packets form directed edges.
- **Deep Learning Extension**: Using **Graph Convolutional Networks (GCN)** or **Graph Attention Networks (GAT)** to monitor graph topology in real time.
- **Benefit**: In normal traffic, the graph maintains stable 1-to-1 IP-MAC edge bindings. During spoofing, an attacker's MAC node spawns rapid, high-degree anomalous edges across multiple IP nodes. The GNN detects this structural perturbation instantly, pinpointing the attacker's true physical MAC address even if it spoofs multiple IPs simultaneously.

### 4. Hierarchical Edge-to-Cloud Pipeline (Integrating our `ANN_ENSEMBLE`)
- **Multi-Tier Architecture**: The project already contains a 3-tier Multi-Layer Perceptron ensemble in the [`ANN_ENSEMBLE/`](../ANN_ENSEMBLE/) folder (ANN-1: 64-32, ANN-2: 128-64, ANN-3: 32-16 with Soft Voting).
- **Future Integration**: Connect the lightweight Gaussian Naive Bayes model to run directly at the local router edge (<5ms latency), while escalating uncertain or borderline packets (confidence $< 0.6$) to the central Deep Neural Network aggregator for deep non-linear inspection and automated mitigation.

