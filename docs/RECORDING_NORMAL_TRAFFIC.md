# Recording a new *normal* ARP capture with Wireshark (Windows)

Why: only **one** normal capture (187 ARP packets) was recorded on the same network as the
attack captures (192.168.x.x). The two other normal captures are from campus Wi-Fi (172.16.x.x), so
models can partly learn "which network" instead of "attack". **The most valuable capture you can
add is a normal one recorded on the attack network** (same machine, same interface, same setup as
the attack captures, with no attack running). `train.py` tags every capture with its network and
reports a same-network evaluation that this capture will strengthen directly.

## 0. Before you start

- Install **Wireshark** and tick **Npcap** during setup. If no interfaces appear, start
  Wireshark with *Run as administrator*.
- **Use the same network and the same capturing machine/interface as your attack
  captures** (they show `192.168.1.x`). If the viewpoint differs, the model can learn the
  viewpoint instead of the attack.
- Make sure **no attack script, attack VM, `nmap`/`arp-scan` or ping-sweep is running**.
  Anything like that would contaminate a "normal" capture.

## 1. Set up the capture

1. Open Wireshark -> **Capture -> Options** (Ctrl+K).
2. **Input tab:** tick only the interface with a live activity graph (Wi-Fi or Ethernet).
   Leave *Enable promiscuous mode* ticked.
3. In **Capture filter for selected interfaces** type `arp` (field turns green).
   This stores only ARP packets - the old normal file was 9 MB for 187 ARP packets - and
   it avoids saving the rest of your private traffic.
4. **Output tab:** *File* -> browse to `F:\Projects\ML_Lab_Submission\pcap_files\` and
   name it `arp_normal_traffic_2.pcapng`; *Format* = **pcapng**.
   (Packets are written to disk as they arrive, so nothing is lost if Wireshark crashes.)
   The file name **must contain the word `normal`** - that is how `train.py` labels it 0.
5. Click **Start**.

## 2. Generate genuine ARP traffic (20-30 minutes)

ARP entries are cached for roughly 15-45 s when idle, so ordinary browsing produces very
little ARP. Force natural resolution:

1. Open **PowerShell / cmd as Administrator**, run `ipconfig` and note your *Default
   Gateway* (e.g. `192.168.1.1`) and the IP of 1-2 other real devices (phone, other PC, printer).
2. Run this loop (replace the IPs) and leave it going:

   ```powershell
   while ($true) {
     arp -d *
     ping -n 2 192.168.1.1   | Out-Null
     ping -n 2 192.168.1.50  | Out-Null
     Start-Sleep -Seconds 20
   }
   ```
3. Meanwhile use the network normally: browse, stream a video, copy a file between two
   machines, **connect/disconnect a phone from the Wi-Fi**, print something. Device joins
   produce normal ARP probes and gratuitous announcements - the model needs to see those as benign.
4. **Do not** run anything that spoofs, floods or scans.

Aim for **500+ ARP packets** (watch the packet counter in the status bar; more is better).

## 3. Stop and save

1. Click the red **Stop** button, and Ctrl+C the PowerShell loop.
2. Because you set the Output file, the capture is already saved. (Otherwise:
   **File -> Save As...** -> format *Wireshark/... - pcapng*.)

## 4. Sanity-check that it really is normal

- Display filter `arp.duplicate-address-detected || arp.duplicate-address-frame`
  -> should show **nothing**.
- **Statistics -> Conversations -> Ethernet**: only devices you recognise.
- **Analyze -> Expert Information**: no ARP warnings.
- Display filter `arp` -> the count should match the captured count.

If anything looks malicious, delete the file and record again.

## 5. Use it

Record **2 or 3** captures at different times/activities if you can, then:

```bash
python model_scripts/train.py
```

`train.py` auto-discovers every `.pcap/.pcapng` directly inside `pcap_files/`. Open
`results/metrics.md`: the leave-one-capture-out table now shows a real **false-positive
rate** for each normal capture (previously "skipped").

To demo `predict.py` on traffic the models have never seen, keep one capture in
`pcap_files/holdout/` (not read by training) and run:

```bash
python model_scripts/predict.py pcap_files/holdout/arp_normal_holdout_2.pcapng --label 0
```

### Command-line alternative (no GUI)

```bash
"C:\Program Files\Wireshark\dumpcap.exe" -D                      # list interfaces, note the number/name
"C:\Program Files\Wireshark\dumpcap.exe" -i "Wi-Fi" -f "arp" -a duration:1800 ^
    -w pcap_files\arp_normal_traffic_2.pcapng
```

(`-a duration:1800` stops automatically after 30 minutes; run the generation loop from step 2 meanwhile.)
