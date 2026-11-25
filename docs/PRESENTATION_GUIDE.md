# Presentation Guide: slide by slide

Deck: `Presentation/ARP_Spoofing_Detection_v2.pptx` (20 slides).
Authors: Garv Noor (1024170189) and Manya Gupta (1024170192), UML501, TIET.

Each slide below has what it shows, what to say in plain words, and (at the end) questions your teacher might ask.
Each slide also has speaker notes in the .pptx itself (use presenter view).
About 1.5 minutes per slide gives roughly 30 minutes. For a shorter talk, go fast on slides 3, 4 and 10, and spend your time on 6, 12, 15 and 17.

## Terms to know first

| Term | Plain meaning |
|---|---|
| **Recall** | Of all real attacks, what share did we catch? 0.87 means 87 of 100 attacks caught. |
| **False-positive rate (FPR) / false alarm** | Of all normal packets, what share did we wrongly flag as attack? Lower is better. |
| **Precision** | Of the packets we flagged, what share really were attacks? |
| **MCC** | One summary score that stays honest when one class is much bigger than the other. |
| **Data leakage** | Test data secretly resembles training data, so the score looks better than it really is. |
| **Cross-validation (CV)** | Choosing settings by repeatedly training on part of the data and testing on the rest. |
| **Hold-out / unseen** | Data kept completely away from training and used only for the final test. |

## Slide 1: Title
**On the slide:** "ARP Spoofing Detection: Extending a 2026 Paper", both names and roll numbers.

**Say:**
- "Our project is built on a research paper from 2026 (Singhal and Kumar, in the journal Frontiers in Big Data)."
- "Their score comes from one split of a public dataset. We test on traffic we recorded ourselves and on data held out of training."
- Give the roadmap: the paper, the gap we found, our research questions, our method, a live demo, our results and their limits.

## Slide 2: Abstract
**On the slide:** four big numbers: 9,243 real ARP packets, 5 attack types, 4 strict test views, 2.7% false alarms on an unseen capture.

**Say (the whole project in 30 seconds):**
- ARP has no authentication, so anyone can send a forged reply and redirect a victim's traffic.
- We recorded 9,243 real ARP packets covering 5 attack types.
- We tuned settings by cross-validation and checked results with four tests designed to prevent leakage.
- We built a live simulator that injects attacks into real traffic.
- The headline is 2.7% false alarms on a capture the model had never seen.

## Slide 3: Introduction and Problem
**On the slide:** 9,243 packets, 5 attack types, 10 features, and the challenge list.

**Say:**
- ARP turns an IP address into a MAC address, and it trusts every reply it receives (RFC 826, from 1982).
- Because of that, attackers can do man-in-the-middle attacks or flood the network.
- There are two practical challenges:
  1. **Our labels are noisy.** We label a whole file as "attack" or "normal", so some packets inside an attack file are really ordinary requests.
  2. **Models must work on networks they have not seen.**
- "10 features, no IP/MAC identity" means the model learns behaviour, not which device sent the packet.

## Slide 4: How ARP Spoofing Works
**On the slide:** a normal ARP diagram, a spoofing diagram, and the five attack types.

**Say:**
- **Normal:** Host A asks "Who has 192.168.1.1?". The router replies "I'm at MAC R", and Host A remembers (caches) that.
- **Attack:** the attacker sends a forged reply saying "192.168.1.1 is at MY MAC". The victim updates its cache, so its traffic now goes through the attacker.
- The five types in our data:
  - **Basic spoof:** a forged reply claiming another host's IP.
  - **Flooding:** many forged replies with random IPs and MACs.
  - **Gratuitous:** an unrequested broadcast announcement.
  - **Race condition:** the forged reply arrives before the real one.
  - **Unicast spoof:** the forged reply is sent straight to one victim.

## Slide 5: The Base Paper
**On the slide:** the paper's setup and four numbers: 99.90% accuracy, 0.17% attacks missed, 12 false alarms among 28,768 normal flows, 99.32% on the second dataset.

**Say:**
- It is a peer-reviewed journal paper (Front. Big Data 9, 2026), not a conference paper. The model is a cost-sensitive Random Forest for medical IoT networks.
- **Their data:** the public CICIoMT2024 dataset, 347,721 flows with 47 features. They added 18 timing features and kept 52.
- **Their method:** one 70/15/15 split (train, validation, test). Class weights and the decision threshold were tuned on the validation part only.
- **Their extra check:** CICIDS2017, a second public dataset, at 99.32%.
- "The authors themselves say it was not validated on real devices." This is the opening for our work.
- Be fair to them: their dataset is much bigger and their scores are higher than ours.

## Slide 6: Why We Built This
The most important slide. Spend time here.

**On the slide:** three columns: what the paper did, what is missing, what we did.

**Say:**
- "The paper's headline score comes from splitting one public dataset into train and test. They did add a second public dataset, which is good."
- "But both datasets are ready-made. Neither is traffic the authors recorded, and no attack type was held out of training."
- "We tested on recordings held out of training, including whole attack types and a separate evening recording."
- "So our research area is ARP-spoofing detection tested beyond a train/test split. This gap is the main reason we built the project."

**Say honestly:**
- We are not claiming a better score.
- We are not claiming to have tested on "completely different networks". Our evening capture is the same campus Wi-Fi at a different time.

## Slide 7: Gaps and Research Questions
**On the slide:** four gaps on the left, three research questions (RQ1 to RQ3) on the right.

**Say:**
- **Gaps in the paper's evidence:**
  1. The score comes from one split of one dataset.
  2. The second check is also a public dataset.
  3. ARP spoofing is a single class, so there are no per-attack results and no attack type is held out.
  4. The comparison table uses numbers from other papers on other datasets.
- **Our questions:**
  - **RQ1:** Does detection work on ARP traffic we recorded ourselves?
  - **RQ2:** Does it hold up on attack types and recordings held out of training?
  - **RQ3:** Can it be shown working on live traffic?
- Slide 19 answers these three questions.

## Slide 8: Base Paper vs Our Work
**On the slide:** a comparison table with rows for Data, Attacks, Features, Testing and Demo.

**Say, row by row:**
- **Data:** 347,721 public flows versus our 9,243 real packets (Wi-Fi traffic plus lab attacks).
- **Attacks:** one class versus five types tested separately.
- **Features:** 47 flow features plus 18 timing features, pruned to 52, versus our 10 per-packet ARP features with no raw IP or MAC.
- **Testing:** one split plus a second public dataset, versus held-out recordings, held-out attack types and an unseen capture.
- **Demo:** offline only versus a live simulator.

**Footnote to read aloud:** "Not directly comparable (different data and task): we claim stronger testing, not higher accuracy. Not done by us: their cost-sensitive threshold."

## Slide 9: Methodology, Dataset
**On the slide:** a table of 10 captures with class, network and packet count.

**Say:**
- Total is 9,243 real packets: **7,786 for training** (5,883 normal, 1,903 attack) plus **1,457 unseen** (the evening capture, kept apart, never trained on). 7,786 + 1,457 = 9,243.
- **Labels come from the file name.** If the name says "normal" the label is 0, otherwise 1.
- There are two networks:
  - **192.168 (lab):** all five attack captures plus one small normal capture (187 packets).
  - **172.16 (campus Wi-Fi):** the other normal captures.
- That is why we added a same-network test (slide 12).
- The **3,980 synthetic rows** are labelled, kept in separate files and never used in a test. They did not help, so the final models use real data only.

## Slide 10: Methodology, Features
**On the slide:** the 10 features with what each measures and what it helps catch.

**Say:**
- All features are computed from the current packet and past packets only, never the future.
- Examples:
  - `mac_conflict` asks whether this IP has been seen before with a different MAC. That is the definition of spoofing.
  - `l2_ip_count` asks how many IPs one sender is claiming. One host claiming many IPs is suspicious.
  - `l2_reply_rate_1s` is replies per second from a sender, which catches flooding.
- **Left out on purpose:**
  - **Raw IP and MAC:** the model could memorise which file a row came from.
  - **Unsolicited-reply flag:** it depends on where the sniffer sits. It is 0% in one normal capture and up to 64% in others.
  - **Ethernet/ARP MAC mismatch:** an artifact of our attack tool.
- Removing them did not lower attack recall and made the model steadier on new networks.

## Slide 11: Methodology, Models
**On the slide:** the shipped models (Naive Bayes, Isolation Forest, Ensemble) and the compared ones (Logistic Regression, Random Forest).

**Say:**
- **Naive Bayes:** supervised. It learns normal versus attack from labelled packets, and it is fast and simple.
- **Isolation Forest:** unsupervised. It trains on normal traffic only and flags anything unusual. It can catch unseen attacks but misses quiet ones.
- **Ensemble (NB OR IF):** alert if either model flags the packet. Missing an attack costs more than a false alarm.
- **Compared but not shipped:**
  - **Logistic Regression** flags 42% of the lab-network normal capture when that capture is held out.
  - **Random Forest** has the best recall but 7.1% false alarms on the attack network. Random Forest is also the base paper's model family, so this comparison links back to the paper.
- **Why we did not ship Random Forest:** it may be recognising the network rather than the attack, which is risky.

## Slide 12: Methodology, Evaluation Design
**On the slide:** the four test views.

**Say:** "This is why our numbers are honest."
1. **Chronological split:** the first 70% of each capture trains and the last 30% tests. Packets in a burst are near-duplicates, and a random split puts them on both sides, which inflates scores.
2. **Same-network test:** only the 192.168 captures, so the model cannot win just by recognising the network.
3. **Leave-one-capture-out:** train on all other captures and test on the one left out. This tests an attack type or capture it never saw.
4. **Unseen capture:** the 1,457-packet evening recording, never used for training or tuning.

- **Example to give:** an early random split gave an AUC of 0.954. After fixing the leakage the numbers dropped, and the lower numbers are the honest ones.
- We report precision, recall, FPR and MCC because accuracy alone misleads when normal traffic outnumbers attacks.

## Slide 13: Tuned and Reproducible Training
**On the slide:** a bar chart of cross-validation score by Isolation Forest contamination, plus three boxes (Tuned, Tracked, Tested).

**Say:**
- **Tuned:** our original Isolation Forest setting (contamination 0.02) was a guess. We scored four values using leave-one-capture-out CV on the training pool only. The score is mean attack recall minus mean false-alarm rate. 0.005 won with 0.653 against 0.640, so we changed it. The unseen capture was never used for tuning.
- **Tracked:** every training run saves the SHA-256 hash of the data, the settings and the git commit, so any model can be traced to its data.
- **Tested:** 28 automated tests check for no leakage, that features use only past packets, that the same seed gives the same result, and that saved models reload identically.
- **Result:** unseen-capture false alarms fell from 5.3% to 2.7%, but same-network recall fell from 0.86 to 0.82.
- **Trade-off to admit:** tuning also showed the Isolation Forest adds little beyond Naive Bayes.

## Slide 14: How the Simulator Works
**On the slide:** a four-step flow: Capture, Inject, Detect, Show.

**Say:**
1. **Capture:** Scapy and Npcap read ARP packets from the real network. It is read-only and kept in memory.
2. **Inject:** when you press an attack button, forged packets are built from real hosts it has already seen, so a forged reply really contradicts what the detectors know. They are added to the in-memory stream only.
3. **Detect:** every packet goes through the same 10-feature code used in training. The three shipped detectors (Naive Bayes, Isolation Forest, Ensemble) score it. An alert fires only on repeated flags from one sender, so one stray packet does not cause an alarm.
4. **Show:** three detector cards, a live chart and a time-to-detect table.

- "Nothing is transmitted on the network and nothing is saved to disk."
- There are also a virtual-LAN mode and a recorded-replay mode.

## Slide 15: Live Demo
**On the slide:** four steps (Start, Watch, Attack, Read) and the command `streamlit run simulator/app.py`, which opens at `http://localhost:8501`.

**Do this on stage:**
1. Open the app and press **Start / Reset**. Wait about 30 seconds until at least three real hosts are seen.
2. Point at the counters: real packets captured and real hosts seen. Shipped cards should be green or yellow.
3. Press **Launch Basic spoof**, then **Flooding**. The cards turn red and the table shows each model's detection time.
4. Press **Stop all attacks** and show the Overall result panel.

- **Say clearly:** the attacks exist only in the app's memory, and nothing is sent onto the network.
- **Fallback:** if live capture fails, switch the sidebar mode to **Virtual LAN**.
- **Before class:** start the app and test it. The server was stopped earlier because the PC ran out of memory, so close heavy programs first.

## Slide 16: Results, Model Comparison
**On the slide:** two bar charts: recall on the all-data hold-out and false-positive rate on the same-network test, for five models.

**Say:**
- **Recall, all-data hold-out:** Random Forest 0.949, Naive Bayes and Ensemble 0.872, Logistic Regression 0.869, Isolation Forest 0.536.
- **False-positive rate, same-network test:** Random Forest 7.1%, Naive Bayes, Ensemble and Logistic Regression 1.8%, Isolation Forest 0.
- **Conclusion:** Random Forest looks best on recall, but it flags the most normal packets on the attack network. Naive Bayes is steadier across networks, so we ship Naive Bayes with the Isolation Forest.
- **Caveat:** the same-network test has only 56 normal packets, so the FPR figures rest on few packets.

## Slide 17: Results, Attack by Attack
**On the slide:** two charts. Left: ensemble recall per held-out attack. Right: false alarms per held-out normal capture.

**Say:**
- **Left chart:** each attack capture was never in training when scored.
  - Flooding 0.94 and unicast spoof 0.91 are caught well.
  - Race condition 0.56 should be treated with caution, because the capture has only 9 packets.
  - Basic spoof 0.48 and gratuitous 0.43 are lower, partly from label noise: many packets in those files look like plain requests.
- **Right chart:** false alarms on normal captures held out one at a time are 0 to 3%. The unseen evening capture is 2.7%.
- **Progress of the unseen result:** 10.6% at first, then 5.3% after adding more evening training data, then 2.7% after tuning.

## Slide 18: Results, Findings and Limits
**On the slide:** three headline numbers and two lists.

**Say, headline numbers:**
- **2.7%** false alarms on the unseen evening capture.
- **0.816** recall on the same-network test (FPR 0.018).
- **0.94** recall on a held-out flooding capture.

**Say, what we found:**
- Weaker attacks: unicast spoof 0.91, basic spoof 0.48, gratuitous 0.43.
- Isolation Forest adds little over Naive Bayes.
- Synthetic rows gave no gain.

**Say, limitations (volunteering these builds trust):**
- Labels are per file, so about half the packets in two attack captures look like plain requests.
- Only 187 normal packets exist on the attack network. The same-network test uses the 30% test part, which is the 56 normal packets on slide 16.
- Simulator attacks are illustrations, not accuracy evidence. Quote the real-data tables, not simulator scores.

## Slide 19: Discussion, Research Questions
**On the slide:** a table with the answer and evidence for RQ1 to RQ3, plus surprises and a limitation.

**Say:**
- **RQ1, Yes:** Naive Bayes gets recall 0.87 at about 0.1% false alarms on the all-data hold-out, and recall 0.82 at 1.8% inside the attack network.
- **RQ2, Partly:** flooding 0.94 and unicast 0.91 are caught, but basic 0.48 and gratuitous 0.43 are weaker, and unseen false alarms are 2.7%.
- **RQ3, Yes, as a demo:** the simulator flags attacks injected into live traffic.
- **Surprises:**
  - Isolation Forest adds little over Naive Bayes.
  - Random Forest may recognise the network.
  - False alarms drift with time of day.
- **Limitation:** the attacks were recorded on one lab network we could not revisit.

## Slide 20: Conclusion and References
**On the slide:** three contributions, future work and four references.

**Say:**
- **Contributions:**
  1. We tested the paper's idea on 9,243 real ARP packets we recorded.
  2. We used four leakage-resistant tests with tuned, tracked training.
  3. We built a live simulator that injects attacks into real traffic.
- **Future work:** a cost-sensitive threshold like the paper's, per-packet labels, and more networks.
- End with "Thank you."

## Questions your teacher may ask

| Question | Answer |
|---|---|
| "Your accuracy is lower than the paper. Why?" | "Different data and task. We claim stronger testing, not higher accuracy. Their 99.9% is on one split of a public dataset, and our held-out tests are deliberately harder." |
| "Why Naive Bayes and not Random Forest?" | "Random Forest has the best recall but 7.1% false alarms on the attack network, and it may be recognising the network. Naive Bayes is steadier across networks." |
| "Why is the Isolation Forest in the ensemble if it adds little?" | "It can catch attacks never seen in training, and it keeps the ensemble unsupervised on normal traffic. Tuning showed it adds little on our data, and we say so." |
| "Is the simulator's detection real?" | "The detectors and features are the real trained ones. The attacks are injected into memory, so simulator scores are demonstrations. Accuracy comes from the real-data tables." |
| "Did you attack a real network?" | "No. Nothing is transmitted. Capture is passive and read-only, and attacks exist only in memory." |
| "Why are basic spoof and gratuitous recall low?" | "Labels are per file, and about half the packets in those captures look like plain requests, so some 'attack' labels are really normal packets." |
| "What is the weakest part of your work?" | "Attack data came from one lab network we couldn't revisit, labels are per file, and the same-network test has only 56 normal packets." |
| "What is synthetic data in your project?" | "3,980 labelled SMOTE-style rows, kept separate, never used in a test. They gave no gain, so the final models use real data only." |

## Before you present
- Check the report's cover page details ("BE Third Year, CoE/CSE", "Submitted to: Dr. Swati"); they were not confirmed.
- Start the simulator and test it before class (`streamlit run simulator/app.py`). Close heavy programs first.
- Rename the deck to `ARP_Spoofing_Detection.pptx` after closing it in PowerPoint, if you want a cleaner name.
