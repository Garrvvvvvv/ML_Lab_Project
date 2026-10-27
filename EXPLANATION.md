# ARP Spoofing Detection — Master Explanation (ML Lab)

This is the full walkthrough of the classical-ML pipeline in this folder:
the problem, the raw data, how the dataset and labels are built, which
models were chosen and why, what every file in `models/` actually is, how
evaluation was done, and a prep-list of questions a teacher is likely to
ask with ready answers.

No deep learning is used anywhere in this folder — only Gaussian Naive
Bayes and Isolation Forest, both classical scikit-learn models.

---

## 1. The problem: what is ARP spoofing

Every device on a LAN has two addresses: an **IP address** (logical) and a
**MAC address** (physical, burned into the network card). To send data,
device A needs to know B's MAC address for a given IP. **ARP (Address
Resolution Protocol)** resolves this:

- A broadcasts: *"Who has 192.168.1.5?"* → **ARP request, opcode 1**
- B replies: *"192.168.1.5 is at MAC xx:xx:xx:xx:xx:xx"* → **ARP reply, opcode 2**

ARP has **no authentication**. Any device can send an unsolicited ARP
reply claiming to own any IP address, and other devices will blindly
update their ARP cache. **ARP spoofing** exploits exactly this: an attacker
sends forged ARP replies to make victims believe the attacker's MAC is the
router's (or another host's) MAC, redirecting traffic through the attacker
(man-in-the-middle), or flooding the network with fake replies.

Attack variants captured in this project's data:
- **Basic spoof** — attacker claims another host's/router's IP
- **ARP flooding** — attacker blasts huge volumes of fake replies with random IP/MAC pairs
- **Gratuitous ARP** — attacker sends an unsolicited reply nobody requested, to pre-poison caches
- **Race condition** — attacker tries to get its fake reply accepted before the real one
- **Unicast spoofing** — same trick sent directly to one victim instead of broadcast, to be stealthier

## 2. The raw data: `.pcapng` files

A `.pcapng` ("pcap Next Generation") file is a raw recording of network
packets with timestamps — nothing more. It was captured with tools like
Wireshark/tcpdump while deliberately running normal traffic or one of the
attack scripts.

Six capture files live in [`pcap_files/`](pcap_files/):

| File | What was happening during capture |
|---|---|
| `arp_normal_traffic.pcapng` | Ordinary network activity, no attack |
| `arp_spoof_basic.pcapng` | Basic ARP spoof running |
| `arp_request_flooding.pcapng` | ARP flood running |
| `arp_gratuitous_announcement.pcapng` | Gratuitous ARP announcements running |
| `arp_reply_race_condition.pcapng` | Race-condition spoof running |
| `arp_spoofing_unicast_reply.pcapng` | Unicast spoof running |

## 3. Reading the packets: Scapy's role (and only its role)

**Scapy** is a Python library for crafting, sending, and parsing network
packets. Here it is used purely as a **decoder** — it has no concept of
"spoofing"; it just hands back structured fields from each packet:

```python
packets = rdpcap(file_path)        # loads every packet in the pcapng file
for pkt in packets:
    if ARP in pkt:                  # is this an ARP packet?
        src_ip  = pkt[ARP].psrc     # sender IP field
        src_mac = pkt[ARP].hwsrc    # sender MAC field
        dst_mac = pkt[ARP].hwdst
        op      = pkt[ARP].op       # 1 = request, 2 = reply
        pkt_time = pkt.time
```

Everything past this point — the actual feature engineering and the
detection logic — is plain Python/pandas written for this project, not
anything Scapy provides.

## 4. Labelling: how each row gets its 0/1 class

Labelling here is **per-file, not per-packet** — a hardcoded lookup in
[`model_scripts/train.py`](model_scripts/train.py):

```python
LABEL_MAP = {
    "arp_normal_traffic.pcapng": 0,
    "arp_spoof_basic.pcapng": 1,
    "arp_request_flooding.pcapng": 1,
    "arp_gratuitous_announcement.pcapng": 1,
    "arp_reply_race_condition.pcapng": 1,
    "arp_spoofing_unicast_reply.pcapng": 1
}
```

The assumption: *"whatever was happening when I captured this file applies
to every packet in it."* If the file was captured while normal traffic was
running, every packet in it → label `0`. If it was captured while an attack
script was running, every packet in it → label `1`. This label is passed
into `extract_arp_features(file_path, label)` and stamped onto every row
pulled from that file.

This is called **coarse / weak labelling**. It's a practical necessity
(nobody manually labels thousands of individual packets), but it's an
approximation — a normal-looking ARP request that happens to occur inside
an attack-labeled capture still gets stamped "attack," even though in
isolation it looks benign. This is one honest explanation for why recall
isn't perfect (see Section 8).

**Concrete proof this happens, from the actual dataset** — this real row
comes from an attack-labeled file:

```
src_ip: 192.168.1.92   dst_ip: 192.168.1.1
op: 1                   ← an ARP REQUEST, not a forged reply
mac_conflict: 0         ← no conflicting MAC seen
label: 1                ← still stamped "attack" anyway
```

By every feature value this row looks like a completely ordinary ARP
request. It was labeled `1` purely because it came from a file where *an
attack was happening somewhere in the session* — the code never checks
whether this specific packet is actually malicious. See Section 12 for a
proposed fix (per-packet labelling) and Section 13 for why this limitation
is fine to report as-is without having fixed it yet.

## 5. Feature engineering

Each ARP packet becomes one row with these columns (built in
`extract_arp_features`):

| Feature | Meaning | Why it matters for detection |
|---|---|---|
| `src_ip`, `dst_ip` | sender/target IP (label-encoded) | identity of the hosts involved |
| `src_mac`, `dst_mac` | sender/target MAC (label-encoded) | identity of the hosts involved |
| `op` | ARP opcode, 1=request / 2=reply | attacks are almost always carried via forged *replies* |
| `is_broadcast` | 1 if destination MAC is `ff:ff:ff:ff:ff:ff` | gratuitous/broadcast spoofing sends unsolicited broadcast replies |
| `mac_conflict` | 1 if this source IP was previously seen with a **different** MAC | **the direct signature of ARP spoofing** — one IP, two physical identities |
| `interarrival_time` | time since the previous packet | floods produce abnormally small gaps |
| `arp_reply_count` | running count of replies sent by this source | flooding/race attacks send many more replies than normal hosts |
| `arp_reply_delay_ms` | time between a request and its matching reply | real replies vs. unsolicited/forged ones behave differently |

`mac_conflict` is maintained with a simple dictionary that remembers the
last MAC seen for each IP:

```python
mac_conflict = 0
if src_ip in ip_mac_map and ip_mac_map[src_ip] != src_mac:
    mac_conflict = 1
ip_mac_map[src_ip] = src_mac
```

**Concrete proof it works, from the actual dataset:** `192.168.1.1` shows
up with MAC `54:47:e8:37:e2:44` in some rows and `5a:9d:7f:00:5a:2c` in
others. Same IP, two different MACs — `mac_conflict` fires 927 times
across the 2090-row dataset.

## 6. Building and saving the dataset

[`train.py`](model_scripts/train.py) assembles every pcap's rows into one
pandas `DataFrame`, sorts by timestamp, fills in `interarrival_time`, and
saves it to [`dataset/`](dataset/) in two stages:

- **`arp_dataset_raw.csv`** — human-readable IPs/MACs, straight after
  extraction. This is "the dataset as created" — open this to show how raw
  packets became structured rows.
- **`arp_dataset_encoded.csv`** — after label-encoding, which is exactly
  what gets fed into the models.

Current run: **2090 total rows** (187 normal, 1903 attack — the attack
captures are simply longer/denser, which is why the split isn't 50/50).

## 7. Models used, and why these specific ones

Two models, deliberately chosen to complement each other:

### Gaussian Naive Bayes — supervised classifier
- **What it is:** assumes each feature is normally (Gaussian) distributed
  within each class, and uses Bayes' theorem to compute which class
  (Normal vs. Spoofed) a new row most likely belongs to, given its feature
  values.
- **Why chosen:** it's fast, simple, needs very little data to produce
  reasonable estimates, and works well as a first supervised baseline for
  a project where the feature set is small and largely independent
  (timing, counts, flags) — exactly the kind of features Naive Bayes
  handles well despite its "naive" independence assumption.
- **Role here:** the main classifier — it directly learns "what attack
  traffic looks like" from labeled examples.

### Isolation Forest — unsupervised anomaly detector
- **What it is:** builds many random decision trees that try to isolate
  each data point by repeatedly splitting on random features/thresholds.
  Points that are "normal" (similar to the bulk of the data) take many
  splits to isolate; outliers get isolated in very few splits. Points that
  isolate fast are flagged as anomalies.
- **Why chosen:** it's trained **only on normal traffic**, so it doesn't
  need attack examples to work — it learns "what normal ARP behavior looks
  like" and flags *any* deviation, including attack patterns the model
  never explicitly saw. This is valuable because real-world attacks may
  not look exactly like the 5 attack types captured here; an anomaly
  detector generalizes better to unseen attack variants than a classifier
  trained only on known attack labels.
- **Role here:** a safety net / second opinion — catches anomalies Naive
  Bayes might miss, without needing attack data to train on.

**Why both, instead of just one:** Naive Bayes is precise on attack
patterns it has seen labeled examples of, but can't recognize a brand-new
attack style it wasn't trained on. Isolation Forest can flag *any*
deviation from normal without needing attack examples, but it can't
explain *which known attack type* it resembles and may also flag unusual-
but-harmless traffic. Combining them (Section 9) covers both gaps.

## 8. Evaluation of the Naive Bayes model

30% of the data held out as a test set (stratified by label). Latest run:

| Metric | Value | Meaning |
|---|---|---|
| Accuracy | 0.686 | fraction of all predictions correct |
| Precision (Spoofed) | 0.995 | when it says "attack," it's almost never wrong |
| Recall (Spoofed) | 0.659 | it misses about a third of actual attack packets |
| F1 Score (Spoofed) | 0.792 | harmonic mean of precision/recall |
| ROC AUC | 0.950 | how well the model separates the two classes overall |
| Log Loss | 1.949 | penalizes confident-but-wrong probability estimates |
| Cohen's Kappa | 0.244 | agreement with ground truth beyond chance |
| Matthews Corr. | 0.363 | balanced correlation measure, robust to class imbalance |

Confusion matrix:

```
              Pred Normal   Pred Spoofed
Actual Normal        54            2
Actual Spoofed      195          376
```

**Honest reading:** precision is excellent (very few false alarms), but
recall is moderate — some attack packets get missed. The likely cause is
coarse labelling (Section 4): some packets inside an "attack" capture
behave identically to normal traffic (e.g., the first packet before a
repeated-reply/mac-conflict pattern builds up), so a per-packet feature
view can't distinguish them. ROC-AUC of 0.95 shows the classes are still
well separated overall.

## 9. Inference / ensemble logic

[`model_scripts/predict.py`](model_scripts/predict.py) loads both saved
models plus the saved encoders, extracts features from a **new** pcap the
same way `train.py` does, and runs both models:

```python
nb_preds = nb_model.predict(df[features])
iso_preds_raw = iso_model.predict(df[features])          # 1 = normal, -1 = anomaly
iso_preds = [0 if pred == 1 else 1 for pred in iso_preds_raw]

# OR logic: flagged as attack if EITHER model says attack
ensemble_preds = [1 if (nb or iso) else 0 for nb, iso in zip(nb_preds, iso_preds)]
```

Results (per-packet Normal/Attack verdicts from each model and the
combined ensemble) are written to [`prediction_output.csv`](prediction_output.csv).

OR logic was chosen deliberately: for a security detection system, a
**missed attack (false negative) is more costly than a false alarm (false
positive)** — an analyst can dismiss a false alarm, but a missed attack
goes undetected. OR logic trades some extra false positives for fewer
missed attacks.

## 10. `models/` folder — what every file is

```
models/
├── encoders/
│   ├── src_ip_encoder.pkl
│   ├── dst_ip_encoder.pkl
│   ├── src_mac_encoder.pkl
│   └── dst_mac_encoder.pkl
└── saved_models/
    ├── naive_bayes_model.pkl
    └── isolation_forest_model.pkl
```

- **`encoders/*.pkl`** — one `sklearn.preprocessing.LabelEncoder` per
  identifier column (`src_ip`, `dst_ip`, `src_mac`, `dst_mac`). Each one is
  a saved lookup table mapping a specific string (e.g. `"192.168.1.1"`) to
  the same integer every time (e.g. `2`). Saved with `joblib.dump()` during
  training, and reloaded with `.transform()` (not `.fit_transform()`) at
  prediction time, so a new pcap's IPs/MACs get encoded consistently with
  what the models were trained on.
- **`saved_models/naive_bayes_model.pkl`** — the fitted `GaussianNB`
  object: it already contains the learned mean/variance of each feature
  for each class, computed once during training. At prediction time it
  just plugs new rows into Bayes' theorem using those stored numbers — no
  retraining.
- **`saved_models/isolation_forest_model.pkl`** — the fitted
  `IsolationForest`: contains the actual trained ensemble of random
  isolation trees built from normal-only training data.

`joblib` is used instead of plain `pickle` because it's optimized for
objects containing large numpy arrays (which is what trained sklearn
models are internally) — faster to save/load.

## 11. Likely teacher questions (with answers)

**Q: Why Naive Bayes and not another classifier (e.g. decision tree, SVM)?**
A: Naive Bayes is fast, interpretable, works well with small/independent
numeric features, and needs little data to get stable probability
estimates — a reasonable baseline given the feature set size here. It's
explicitly a baseline, not claimed to be the best possible classifier; a
tree-based or boosting model could reasonably outperform it and is a
natural next step.

**Q: What does "naive" mean in Naive Bayes?**
A: It assumes all features are conditionally independent given the class —
an assumption that's technically false here (e.g., `arp_reply_count` and
`interarrival_time` are correlated), but the model still performs well in
practice despite that violated assumption, which is typical for Naive
Bayes.

**Q: Why train Isolation Forest only on normal data?**
A: Because it's designed for anomaly detection — it needs to learn what
"normal" looks like, then anything that deviates (including attack types
never seen in training) gets flagged. Training it on a mix of both would
defeat the purpose; it would just become a worse classifier.

**Q: How are labels assigned — did you label every packet by hand?**
A: No — labelling is per-file (Section 4): every packet from a capture
taken while an attack was running is labeled 1, every packet from a normal
capture is labeled 0. It's a practical weak-labelling approach, with the
limitation that some individual packets inside an "attack" file may still
look behaviorally normal.

**Q: Why is recall lower than precision?**
A: The model rarely raises a false alarm (precision 0.995) but misses
roughly a third of true attack packets (recall 0.659). Likely cause:
per-packet features can't always distinguish early/isolated packets
within an attack burst from genuinely normal traffic — e.g. before
`mac_conflict` or `arp_reply_count` builds up. Tracking behavior across a
packet window instead of single packets would likely improve recall.

**Q: What is Scapy's exact role — did it detect the attack?**
A: No. Scapy only decodes the pcapng file into structured packet fields
(`psrc`, `hwsrc`, `op`, etc.). All detection logic — `mac_conflict`,
reply counting, delay computation, labelling — is custom Python code
written on top of what Scapy extracts.

**Q: What is `mac_conflict` and why is it the most important feature?**
A: It's 1 if an IP address was previously seen associated with a different
MAC address. This is the literal definition of ARP spoofing on the wire —
one IP claimed by two physical identities — so it's the most direct
signal available, confirmed by example in Section 5.

**Q: Why use label encoding instead of one-hot encoding for IP/MAC?**
A: IP/MAC columns have many unique values (high cardinality); one-hot
encoding would create hundreds of sparse columns. Label encoding keeps it
to a single integer column per feature, which Naive Bayes and Isolation
Forest can both consume directly, at the cost of implying an arbitrary
ordinal relationship that doesn't really exist (a trade-off worth
acknowledging if pushed on it).

**Q: What happens if a new IP/MAC appears at prediction time that wasn't
in the training data?**
A: `LabelEncoder.transform()` will raise an error, since it only knows the
values it was fit on. This is a known limitation of this approach — in
production you'd want either a fallback "unknown" bucket or a feature
representation that doesn't require a fixed vocabulary.

**Q: Why combine two models with OR logic instead of just picking the
better one?**
A: They catch different things — Naive Bayes recognizes patterns similar
to known labeled attacks, Isolation Forest flags *any* deviation from
normal even if it's a novel attack pattern. OR logic favors catching more
attacks (fewer missed detections) at the cost of more false positives,
which is the right trade-off for a security detection use case.

**Q: Is this dataset balanced?**
A: No — 1903 attack rows vs. 187 normal rows (roughly 10:1), because the
attack captures are longer/denser than the normal baseline capture. This
imbalance is part of why metrics beyond plain accuracy (precision/recall/
F1/ROC-AUC/Kappa/MCC) are reported — accuracy alone would be misleading on
an imbalanced dataset.

**Q: Is there any deep learning in this part of the project?**
A: No — this folder is deliberately classical ML only (Naive Bayes +
Isolation Forest, both scikit-learn). A separate experimental pipeline in
the full project explores a neural-network (ANN) ensemble, but it's kept
out of this submission.

---

## 12. Proposed improvement: per-packet labelling (not yet implemented)

The labelling limitation in Section 4 has a concrete fix, scoped but
intentionally not yet built:

**Tier 1 — rule-based heuristics (no extra information needed):**

| Rule | Catches |
|---|---|
| `mac_conflict == 1` on this specific row | the forged reply itself (basic spoof, race condition) |
| `op == 2` and `is_broadcast == 1` with no matching prior request | gratuitous ARP (unsolicited broadcast reply) |
| `arp_reply_count` for this source crossing a threshold in a short time window | flooding |

Packets inside an attack-labeled file that don't trip any rule would stay
labeled `0` — directly fixing the exact problem shown in Section 4 (the
`op=1` request wrongly stamped `1`).

**Tier 2 — known-attacker-identity rule:** only possible where the
attack-generation script documents a fixed attacker MAC/IP. Checked this
repo: only [`flood.py`](../Results/Types/ARP%20Flooding/flood.py) hardcodes
one (`ATTACKER_MAC = "b2:25:ee:01:b9:b6"`), and even there it randomizes
the forged MAC per packet (`random_mac()`), so this rule turns out to be
less reliable than it first sounds — Tier 1's broadcast/reply-rate
heuristic is the more realistic signal for that attack type in practice.

**Honest limitation of this fix itself:** it replaces "trust the file"
with "trust a hand-written rule" — better because it's packet-content-
aware, but still a heuristic requiring manual threshold tuning (e.g. what
reply-rate counts as "flooding"), validated by spot-checking flagged vs.
unflagged packets in Wireshark rather than independently certified ground
truth. This is worth stating plainly if asked, rather than presenting the
improved labels as perfect.

If/when implemented, this would run as a separate relabelling step
(`arp_dataset_raw.csv` → per-packet-ruled relabel → `arp_dataset_perpacket.csv`),
kept alongside the original dataset so both labelling strategies — and
the resulting model metrics — can be compared directly.

## 13. Reporting progress without this being implemented yet

Not having implemented Section 12 yet is not a weakness to hide — it's a
legitimate, explainable project state:

- **Per-file labelling is a standard, defensible first-pass approach** for
  bootstrapping a labeled dataset without manually annotating every
  packet — not a mistake, a normal starting point.
- **Identifying the limitation yourself, with a concrete fix scoped out,
  is worth more than silently patching it.** Being able to explain *why*
  `mac_conflict`/broadcast/reply-rate are reasonable heuristics, and what
  their false-positive risk is, demonstrates understanding — mechanically
  implementing a fix you can't defend under questioning is the riskier
  option.
- **What actually matters for evaluation:** having a working end-to-end
  pipeline (pcap → features → dataset → trained models → metrics), being
  able to explain every choice made (labelling, feature engineering,
  model choice, metric interpretation), and being upfront about
  limitations. This submission has all of that already.
- **A reasonable week-by-week narrative:** this week — pipeline, dataset,
  two trained models, full metrics, limitations identified. Next
  checkpoint (if applicable) — implement Section 12's per-packet
  labelling and show a before/after metrics comparison. Showing a project
  visibly evolve across checkpoints generally reads better in evaluation
  than presenting everything finished at once with no visible process.
