# Simulator demo guide (for the presentation, slide 5)

## Before the presentation (5 minutes)

1. Connect to the Wi-Fi you want to capture (the app captures ARP only, read-only).
2. Start the app from the project folder: `streamlit run simulator/app.py`, then open <http://localhost:8501>.
3. Do one dry run: press **Start / Reset**, wait ~30 s, launch **Basic spoof**, then **Stop all attacks**.
4. Leave the page open in a browser tab, in the default mode **REAL live traffic + injected attacks**.
5. Fallback if capture fails (no Wi-Fi, no Npcap): switch the sidebar mode to **Virtual LAN: launch attacks manually**.
   It works with no network at all and has the same buttons.

## What the simulator is (say this first, in one breath)

"It reads the real ARP traffic of this network live, passively, and runs our models on it packet by packet.
When I press an attack button, forged packets are added to that stream inside the program, so we can see the models
react. Nothing is ever sent on the network, so no real device is attacked."

## Demo script (about 3 minutes)

| Step | What you do | What you say |
|---|---|---|
| 1 | Show the page, press **Start / Reset** | "This is live ARP traffic from this network, captured read-only. Real packets are assumed normal." |
| 2 | Point at *Real packets captured* and *Real hosts seen* | "The counters rise as real devices talk. I wait until it has seen at least three real hosts." |
| 3 | Point at the **shipped detectors** (Naive Bayes, Isolation Forest, Ensemble). Green = quiet, yellow = stray flags (noise) | "Before any attack, a red card on a shipped detector would be a false alarm on real traffic. Yellow means the odd packet was flagged but not like an attacker. The comparison models below can over-alert on a new network, so we show them separately." |
| 4 | Press **Launch Basic spoof** | "This forges replies claiming the real gateway's address, using hosts it actually saw. The shipped cards turn red." |
| 5 | Point at the packet feed (truth column shows `INJECTED: basic`) | "Every injected packet is labelled, so we know exactly what should be caught." |
| 6 | Press **Launch Flooding** as well | "Flooding is a different attack: many forged identities fast. Attacks can run together." |
| 7 | Scroll to *How fast was each launch detected?* | "This table shows, per launch, how many seconds each model took to raise the first alert, or 'missed'." |
| 8 | Scroll to *Overall result so far* | "This ranks the models. Best F1, highest recall and lowest false alarms on clean traffic." |
| 9 | Press **Stop all attacks** | "Alerts on the poisoned address can persist afterwards, which is a known limitation." |

Try one more launch if time allows: **Unicast spoof** or **Race condition** shows that the Isolation Forest can miss
quiet attacks that Naive Bayes catches (that is why the ensemble uses both).

## Points to say out loud (honesty)

- The injected attacks are **simulated**, so these scores are an illustration, not accuracy evidence.
  The accuracy numbers are in the Results slides (real recorded packets only).
- Real packets are assumed normal; a genuine attack on the network would appear as a "false" alarm.
- Real MAC and IP addresses are masked on screen by default (other people's devices are on this network).
- Nothing is transmitted and nothing is saved to disk.

## Likely questions

**Is this real traffic?** The background is real ARP traffic captured live. The attacks are injected in memory.

**Why not attack the network for real?** It would poison other students' ARP caches on a shared network. We test
attack detection on recorded lab attacks and on injected ones instead.

**Why do some models stay quiet on some attacks?** The Isolation Forest only knows what normal looks like and
misses low-rate attacks such as unicast spoof and race condition; Naive Bayes learned those from labelled attacks.

**Which model do you recommend?** Naive Bayes with the Isolation Forest (ensemble). Random Forest has the best recall
on the combined test but flags 7.1% of normal packets in the attack-network test.

**What do the red/green cards mean?** Red = that detector flagged a packet in the last 3 seconds of network time.

## Other modes (use if asked)

- **Virtual LAN: launch attacks manually** - same buttons on invented traffic, works offline.
- **Scripted simulation** - the attack starts automatically 30% into a fixed-length run.
- **Replay recorded capture** - plays a recorded pcap through the models. Use `holdout/arp_normal_holdout_2.pcapng`
  for an unbiased replay; captures from the training pool give optimistic scores.

## If the network is very busy
If a yellow **Busy network** banner appears (the live ARP rate is far above the training data), say so: "This network is
busier than anything the models were trained on. The comparison models over-alert, which is a real generalisation limit;
the shipped detectors stay calm." Press **Start / Reset** just before you launch attacks; a session can run up to 60,000
packets, and detection works the same at any rate. If it is still confusing, switch the sidebar to **Virtual LAN**.

An **ALERT** means a sender had at least 3 flagged packets and over half of its recent packets flagged in 10 seconds.
**Stray flags** are flagged packets that do not look like an attacker.
