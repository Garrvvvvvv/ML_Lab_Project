import os
import time
import pandas as pd
from scapy.all import rdpcap, ARP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, log_loss, cohen_kappa_score, matthews_corrcoef
)
import joblib
from collections import defaultdict

PCAP_DIR = "pcap_files"
LABEL_MAP = {
    "arp_normal_traffic.pcapng": 0,
    "arp_spoof_basic.pcapng": 1,
    "arp_request_flooding.pcapng": 1,
    "arp_gratuitous_announcement.pcapng": 1,
    "arp_reply_race_condition.pcapng": 1,
    "arp_spoofing_unicast_reply.pcapng": 1
}

def extract_arp_features(file_path, label):
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
                "arp_reply_delay_ms": reply_delay_ms,
                "label": label
            })
    return pd.DataFrame(data)

def load_all_pcap_files():
    all_dfs = []
    for filename, label in LABEL_MAP.items():
        full_path = os.path.join(PCAP_DIR, filename)
        print(f"Processing file: {filename}")
        df = extract_arp_features(full_path, label)
        all_dfs.append(df)
    return pd.concat(all_dfs).reset_index(drop=True)

def main():
    os.makedirs("dataset", exist_ok=True)

    print("Extracting features from PCAP files...")
    df = load_all_pcap_files()
    df = df.sort_values(by="pkt_time")
    df["interarrival_time"] = df["pkt_time"].diff().fillna(0)
    df["arp_reply_delay_ms"] = df["arp_reply_delay_ms"].fillna(-1)

    # Save the dataset exactly as extracted from the pcaps, before any
    # encoding — human-readable IPs/MACs, for showing "how the dataset
    # was created" from raw packets.
    raw_path = os.path.join("dataset", "arp_dataset_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"[INFO] Saved raw extracted dataset ({len(df)} rows) to {raw_path}")

    encoders = {}
    for col in ["src_ip", "dst_ip", "src_mac", "dst_mac"]:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col])
        encoders[col] = enc

    features = [
        "src_ip", "dst_ip", "src_mac", "dst_mac", "op",
        "is_broadcast", "mac_conflict", "interarrival_time",
        "arp_reply_count", "arp_reply_delay_ms"
    ]
    X = df[features]
    y = df["label"]

    # Save the encoded dataset actually fed into the models.
    encoded_path = os.path.join("dataset", "arp_dataset_encoded.csv")
    df[features + ["label"]].to_csv(encoded_path, index=False)
    print(f"[INFO] Saved encoded training dataset to {encoded_path}")

    print("Splitting data and training Naive Bayes classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    start_time = time.time()
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = nb_model.predict(X_test)
    y_proba = nb_model.predict_proba(X_test)

    print("\nNaive Bayes Model Evaluation:")
    print(f"Training time: {training_time:.4f} seconds")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Spoofed"]))

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    lloss = log_loss(y_test, y_proba)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    print("\nAdditional Metrics:")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision       : {prec:.4f}")
    print(f"Recall          : {rec:.4f}")
    print(f"F1 Score        : {f1:.4f}")
    print(f"ROC AUC         : {auc:.4f}")
    print(f"Log Loss        : {lloss:.4f}")
    print(f"Cohen’s Kappa   : {kappa:.4f}")
    print(f"Matthews Corr.  : {mcc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)

    print("\nTraining Isolation Forest on normal traffic only...")
    df_normal = df[df["label"] == 0]
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    iso_model.fit(df_normal[features])

    print("Saving models and encoders...")
    os.makedirs("models/saved_models", exist_ok=True)
    os.makedirs("models/encoders", exist_ok=True)

    joblib.dump(nb_model, "models/saved_models/naive_bayes_model.pkl")
    joblib.dump(iso_model, "models/saved_models/isolation_forest_model.pkl")

    for col, enc in encoders.items():
        joblib.dump(enc, f"models/encoders/{col}_encoder.pkl")

    print("Training complete. Models and encoders saved.")

if __name__ == "__main__":
    main()
