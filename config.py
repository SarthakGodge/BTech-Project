"""
Central configuration for the hybrid CNN+LSTM IDS pipeline.
Edit paths here only - every script imports from this file.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).parent
RAW_DIR = Path("/data/cicids2018/raw")        # where the CSVs land
PARQUET_DIR = Path("/data/cicids2018/parquet")  # cleaned per-day parquet
SEQ_DIR = Path("/data/cicids2018/sequences")    # windowed .npy memmaps
ARTIFACT_DIR = ROOT / "artifacts"               # scaler, selector, model

for d in (PARQUET_DIR, SEQ_DIR, ARTIFACT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Day files -> split assignment
#
# Chronological split. Sequences are time-ordered, so a random split would
# leak overlapping windows between train and test. Splitting by DAY makes
# the evaluation honest: the model is tested on traffic and attack
# executions it has never seen.
#
# 02-20 is excluded by default: it is ~8 GB and carries four extra columns
# (Flow ID, Src IP, Src Port, Dst IP) that no other file has. Set
# INCLUDE_0220 = True to use it - 01_ingest.py normalises the schema.
# ----------------------------------------------------------------------
INCLUDE_0220 = False

DAY_SPLITS = {
    "Wednesday-14-02-2018": "train",   # FTP-BruteForce, SSH-Bruteforce
    "Thursday-15-02-2018":  "train",   # DoS GoldenEye, DoS Slowloris
    "Friday-16-02-2018":    "train",   # DoS Hulk, DoS SlowHTTPTest
    "Thursday-20-02-2018":  "train",   # DDoS LOIC-HTTP  (large, optional)
    "Wednesday-21-02-2018": "train",   # DDoS LOIC-UDP, DDOS HOIC
    "Thursday-22-02-2018":  "train",   # Brute Force Web/XSS, SQL Injection
    "Friday-23-02-2018":    "train",   # Brute Force Web/XSS, SQL Injection
    "Wednesday-28-02-2018": "val",     # Infiltration
    "Thursday-01-03-2018":  "test",    # Infiltration
    "Friday-02-03-2018":    "test",    # Bot
}

# ----------------------------------------------------------------------
# Label engineering
# ----------------------------------------------------------------------
CLASS_NAMES = ["No Attack", "Pre-Attack", "Attack"]

# Raw CIC-IDS2018 label strings -> base class.
# Infilteration (their spelling) is mapped to Pre-Attack directly: those
# scenarios capture lateral movement / probing rather than payload delivery.
LABEL_MAP = {
    "Benign": 0,
    "Infilteration": 1,
    "Infiltration": 1,
    "Bot": 2,
    "DoS attacks-GoldenEye": 2,
    "DoS attacks-Slowloris": 2,
    "DoS attacks-Hulk": 2,
    "DoS attacks-SlowHTTPTest": 2,
    "DDoS attacks-LOIC-HTTP": 2,
    "DDOS attack-LOIC-UDP": 2,
    "DDOS attack-HOIC": 2,
    "FTP-BruteForce": 2,
    "SSH-Bruteforce": 2,
    "Brute Force -Web": 2,
    "Brute Force -XSS": 2,
    "SQL Injection": 2,
}

# Fraction of each contiguous attack episode (leading flows, in time order)
# that gets relabelled 0 -> 1. This is the temporal Pre-Attack derivation.
PRE_ATTACK_FRACTION = 0.12
PRE_ATTACK_MAX_FLOWS = 500   # cap per episode so huge DDoS bursts don't dominate

# Columns that must never become features: they leak identity or time
# rather than behaviour.
LEAK_COLUMNS = [
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp", "Label",
    "__y", "__day",
]

# ----------------------------------------------------------------------
# Feature selection / windowing
# ----------------------------------------------------------------------
VARIANCE_THRESHOLD = 0.01
K_FEATURES = 35          # matches the design figure (84 -> 35)
WINDOW = 20              # T: flows per sequence
STRIDE = 2               # step between window starts
WINDOW_DTYPE = "float32"

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 1e-3
SEED = 42

# Focal loss (handles the 2% Pre-Attack class better than plain CE)
FOCAL_GAMMA = 2.0
USE_FOCAL_LOSS = True

# Cap on total training windows held in memory. Set to None to use all.
# 3M windows x 20 x 35 x 4 bytes ~= 8.4 GB, so keep this in line with
# your instance RAM.
MAX_TRAIN_WINDOWS = 3_000_000
