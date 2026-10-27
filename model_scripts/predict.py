import os
import pandas as pd
import joblib
from scapy.all import rdpcap, ARP
from collections import defaultdict

NB_MODEL_PATH = "models/saved_models/naive_bayes_model.pkl"
IF_MODEL_PATH = "models/saved_models/isolation_forest_model.pkl"
ENCODER_PATH = "models/encoders"

def extract_features_from_pcap(file_path):
    packets = rdpcap(file_path)
    data = []
    ip_mac_map = {}
    arp_request_timestamps = {}
    arp_reply_counter = defaultdict(int)

    for pkt in packets:
        if ARP in pkt:
            src_ip = pkt[ARP].psrc
            dst_ip = pkt[ARP].pdst
            src_mac = pkt[ARP].hwsrc
            dst_mac = pkt[ARP].hwdst
            op = pkt[ARP].op
            pkt_time = pkt.time
            is_broadcast = int(dst_mac == "ff:ff:ff:ff:ff:ff")

            if op == 2:
                arp_reply_counter[src_ip] += 1

            reply_delay_ms = -1
            if op == 1:
                arp_request_timestamps[(dst_ip, src_ip)] = pkt_time
            elif op == 2 and (src_ip, dst_ip) in arp_request_timestamps:
                req_time = arp_request_timestamps[(src_ip, dst_ip)]
                reply_delay_ms = (pkt_time - req_time) * 1000
                del arp_request_timestamps[(src_ip, dst_ip)]

            mac_conflict = 0
            if src_ip in ip_mac_map and ip_mac_map[src_ip] != src_mac:
                mac_conflict = 1
            ip_mac_map[src_ip] = src_mac

            data.append({
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_mac": src_mac,
                "dst_mac": dst_mac,
                "op": op,
                "is_broadcast": is_broadcast,
                "mac_conflict": mac_conflict,
                "pkt_time": pkt_time,
                "arp_reply_count": arp_reply_counter[src_ip],
                "arp_reply_delay_ms": reply_delay_ms
            })
    return pd.DataFrame(data)

def load_encoders():
    encoders = {}
    for col in ["src_ip", "dst_ip", "src_mac", "dst_mac"]:
        encoders[col] = joblib.load(os.path.join(ENCODER_PATH, f"{col}_encoder.pkl"))
    return encoders

def predict_with_ensemble(file_path):
    df = extract_features_from_pcap(file_path)
    if df.empty:
        print("No ARP packets found in the provided file.")
        return

    df = df.sort_values(by="pkt_time")
    df["interarrival_time"] = df["pkt_time"].diff().fillna(0)
    df["arp_reply_delay_ms"] = df["arp_reply_delay_ms"].fillna(-1)

    encoders = load_encoders()
    for col in encoders:
        df[col] = encoders[col].transform(df[col])

    features = [
        "src_ip", "dst_ip", "src_mac", "dst_mac", "op",
        "is_broadcast", "mac_conflict", "interarrival_time",
        "arp_reply_count", "arp_reply_delay_ms"
    ]

    # Load models
    nb_model = joblib.load(NB_MODEL_PATH)
    iso_model = joblib.load(IF_MODEL_PATH)

    # Naive Bayes predictions
    nb_preds = nb_model.predict(df[features])

    # Isolation Forest predictions (1=normal, -1=anomaly → we map to 0 and 1)
    iso_preds_raw = iso_model.predict(df[features])
    iso_preds = [0 if pred == 1 else 1 for pred in iso_preds_raw]

    # Ensemble: mark attack if any model detects it
    ensemble_preds = [1 if (nb or iso) else 0 for nb, iso in zip(nb_preds, iso_preds)]

    df["nb_prediction"] = nb_preds
    df["if_prediction"] = iso_preds
    df["final_prediction"] = ensemble_preds

    print("\nEnsemble Prediction Summary:")
    print(df[["src_ip", "dst_ip", "nb_prediction", "if_prediction", "final_prediction"]].value_counts())
    print("\nSummary Count (0=Normal, 1=Attack):")
    print(df["final_prediction"].value_counts())

    df.to_csv("prediction_output.csv", index=False)
    print("\nDetailed prediction saved to 'prediction_output.csv'.")

if __name__ == "__main__":
    predict_with_ensemble(r"pcap_files\arp_gratuitous_announcement.pcapng")
