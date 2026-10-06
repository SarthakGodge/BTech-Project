"""Shared helpers. Must be importable wherever scaler.pkl is unpickled
(training, evaluation AND the serving container)."""
import re
import numpy as np


def signed_log1p(X):
    """Compress heavy-tailed flow statistics; keeps the sign (some are -1)."""
    X = np.asarray(X, dtype=np.float64)
    return np.sign(X) * np.log1p(np.abs(X))


def norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# Canonical names = CSE-CIC-IDS2018 style. Columns that are spelled the same
# in both datasets (modulo case/spaces) need no alias.
CANON = """Dst Port|Protocol|Flow Duration|Tot Fwd Pkts|Tot Bwd Pkts|TotLen Fwd Pkts|TotLen Bwd Pkts|
Fwd Pkt Len Max|Fwd Pkt Len Min|Fwd Pkt Len Mean|Fwd Pkt Len Std|Bwd Pkt Len Max|Bwd Pkt Len Min|
Bwd Pkt Len Mean|Bwd Pkt Len Std|Flow Byts/s|Flow Pkts/s|Flow IAT Mean|Flow IAT Std|Flow IAT Max|
Flow IAT Min|Fwd IAT Tot|Fwd IAT Mean|Fwd IAT Std|Fwd IAT Max|Fwd IAT Min|Bwd IAT Tot|Bwd IAT Mean|
Bwd IAT Std|Bwd IAT Max|Bwd IAT Min|Fwd PSH Flags|Bwd PSH Flags|Fwd URG Flags|Bwd URG Flags|
Fwd Header Len|Bwd Header Len|Fwd Pkts/s|Bwd Pkts/s|Pkt Len Min|Pkt Len Max|Pkt Len Mean|Pkt Len Std|
Pkt Len Var|FIN Flag Cnt|SYN Flag Cnt|RST Flag Cnt|PSH Flag Cnt|ACK Flag Cnt|URG Flag Cnt|
CWE Flag Count|ECE Flag Cnt|Down/Up Ratio|Pkt Size Avg|Fwd Seg Size Avg|Bwd Seg Size Avg|
Fwd Byts/b Avg|Fwd Pkts/b Avg|Fwd Blk Rate Avg|Bwd Byts/b Avg|Bwd Pkts/b Avg|Bwd Blk Rate Avg|
Subflow Fwd Pkts|Subflow Fwd Byts|Subflow Bwd Pkts|Subflow Bwd Byts|Init Fwd Win Byts|
Init Bwd Win Byts|Fwd Act Data Pkts|Fwd Seg Size Min|Active Mean|Active Std|Active Max|Active Min|
Idle Mean|Idle Std|Idle Max|Idle Min""".replace("\n", "").split("|")

# (canonical, CIC-IDS2017-style variant)
_PAIRS = [
    ("Dst Port", "Destination Port"), ("Tot Fwd Pkts", "Total Fwd Packets"),
    ("Tot Bwd Pkts", "Total Backward Packets"),
    ("TotLen Fwd Pkts", "Total Length of Fwd Packets"),
    ("TotLen Bwd Pkts", "Total Length of Bwd Packets"),
    ("Fwd Pkt Len Max", "Fwd Packet Length Max"), ("Fwd Pkt Len Min", "Fwd Packet Length Min"),
    ("Fwd Pkt Len Mean", "Fwd Packet Length Mean"), ("Fwd Pkt Len Std", "Fwd Packet Length Std"),
    ("Bwd Pkt Len Max", "Bwd Packet Length Max"), ("Bwd Pkt Len Min", "Bwd Packet Length Min"),
    ("Bwd Pkt Len Mean", "Bwd Packet Length Mean"), ("Bwd Pkt Len Std", "Bwd Packet Length Std"),
    ("Flow Byts/s", "Flow Bytes/s"), ("Flow Pkts/s", "Flow Packets/s"),
    ("Fwd IAT Tot", "Fwd IAT Total"), ("Bwd IAT Tot", "Bwd IAT Total"),
    ("Fwd Header Len", "Fwd Header Length"), ("Bwd Header Len", "Bwd Header Length"),
    ("Fwd Pkts/s", "Fwd Packets/s"), ("Bwd Pkts/s", "Bwd Packets/s"),
    ("Pkt Len Min", "Min Packet Length"), ("Pkt Len Max", "Max Packet Length"),
    ("Pkt Len Mean", "Packet Length Mean"), ("Pkt Len Std", "Packet Length Std"),
    ("Pkt Len Var", "Packet Length Variance"),
    ("FIN Flag Cnt", "FIN Flag Count"), ("SYN Flag Cnt", "SYN Flag Count"),
    ("RST Flag Cnt", "RST Flag Count"), ("PSH Flag Cnt", "PSH Flag Count"),
    ("ACK Flag Cnt", "ACK Flag Count"), ("URG Flag Cnt", "URG Flag Count"),
    ("ECE Flag Cnt", "ECE Flag Count"),
    ("Pkt Size Avg", "Average Packet Size"), ("Fwd Seg Size Avg", "Avg Fwd Segment Size"),
    ("Bwd Seg Size Avg", "Avg Bwd Segment Size"),
    ("Fwd Byts/b Avg", "Fwd Avg Bytes/Bulk"), ("Fwd Pkts/b Avg", "Fwd Avg Packets/Bulk"),
    ("Fwd Blk Rate Avg", "Fwd Avg Bulk Rate"), ("Bwd Byts/b Avg", "Bwd Avg Bytes/Bulk"),
    ("Bwd Pkts/b Avg", "Bwd Avg Packets/Bulk"), ("Bwd Blk Rate Avg", "Bwd Avg Bulk Rate"),
    ("Subflow Fwd Pkts", "Subflow Fwd Packets"), ("Subflow Fwd Byts", "Subflow Fwd Bytes"),
    ("Subflow Bwd Pkts", "Subflow Bwd Packets"), ("Subflow Bwd Byts", "Subflow Bwd Bytes"),
    ("Init Fwd Win Byts", "Init_Win_bytes_forward"), ("Init Bwd Win Byts", "Init_Win_bytes_backward"),
    ("Fwd Act Data Pkts", "act_data_pkt_fwd"), ("Fwd Seg Size Min", "min_seg_size_forward"),
]

# normalised column name -> canonical name
LOOKUP = {norm(c): c for c in CANON}
for canon, variant in _PAIRS:
    LOOKUP[norm(variant)] = canon
LOOKUP[norm("Label")] = "Label"
LOOKUP[norm("Timestamp")] = "Timestamp"
