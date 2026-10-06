"""
Central configuration - v2.

Changes vs v1
-------------
* Two datasets (CIC-IDS2017 + CSE-CIC-IDS2018) harmonised to one schema.
* Block-wise stratified split instead of a split by day (see 01_prepare.py).
* Pre-Attack = reconnaissance/lateral-movement classes that really exist in
  the data (Infiltration, PortScan) instead of relabelled benign flows.
* Window label = label of the last flow.
* Balancing strategy is chosen at train time: 05_train.py --balance ...

Set the environment variable IDS_DATA_ROOT to relocate all data.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_ROOT = Path(os.environ.get("IDS_DATA_ROOT", "/data"))

# ----------------------------------------------------------------------
# Paths. Point each dataset to the folder holding its CSVs. Use the
# CORRECTED releases (Engelen et al. / Liu et al.) if you can get them.
# Any CSV whose columns can be mapped is accepted; see ALIASES in common.py
# ----------------------------------------------------------------------
RAW_DIR = DATA_ROOT / "cicids2018" / "raw"          # kept for 00_download.sh
DATASET_DIRS = {
    "cic2018": RAW_DIR,
    "cic2017": DATA_ROOT / "cicids2017" / "raw",
}
PARQUET_DIR = DATA_ROOT / "ids" / "parquet"
SEQ_DIR = DATA_ROOT / "ids" / "sequences"
ARTIFACT_DIR = ROOT / "artifacts"
for d in (PARQUET_DIR, SEQ_DIR, ARTIFACT_DIR):
    d.mkdir(parents=True, exist_ok=True)

INCLUDE_0220 = False        # skip the ~8 GB 2018 file named *20-02*

# ----------------------------------------------------------------------
# Labels. First matching rule wins (case-insensitive substring); anything
# not matched is an Attack. Edit to taste.
# ----------------------------------------------------------------------
CLASS_NAMES = ["No Attack", "Pre-Attack", "Attack"]
LABEL_RULES = [
    ("benign", 0),
    ("infil", 1),        # Infiltration / Infilteration  -> Pre-Attack
    ("portscan", 1),     # CIC-IDS2017 reconnaissance    -> Pre-Attack
    ("port scan", 1),
]
LABEL_DEFAULT = 2        # DoS, DDoS, brute force, web attacks, bot, ...
# Sub-type code stored in the parquet column __sub, so 04_sequences.py can
# re-map PortScan without re-running ingestion (--portscan-as attack).
SUBCODES = {"benign": 0, "infil": 1, "portscan": 2, "port scan": 2}   # other attacks = 3
DEDUP_FLOWS = True       # drop exact duplicate flows (same features + label) per file

# ----------------------------------------------------------------------
# Split / windowing
# ----------------------------------------------------------------------
BLOCK_SIZE = 4000                    # flows per contiguous block
SPLIT_FRACS = (0.70, 0.10, 0.20)     # train / val / test
WINDOW = 20
STRIDE = 2
WINDOW_DTYPE = "float32"
WINDOW_LABEL_MODE = "last"           # "last" | "max"

# ----------------------------------------------------------------------
# Feature selection (fit on TRAIN rows only)
# ----------------------------------------------------------------------
EXCLUDE_FEATURES = []                # e.g. ["Dst Port"] for cross-dataset tests
VARIANCE_THRESHOLD = 0.01            # applied after signed-log transform
CORR_THRESHOLD = 0.98                # drop near-duplicate features
K_FEATURES = 35
MI_SAMPLE_PER_CLASS = 50_000         # class-balanced sample for MI

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
BATCH_SIZE = int(os.environ.get("IDS_BATCH", 1024))   # T4 16 GB: larger batches keep the GPU busier
EPOCHS = int(os.environ.get("IDS_EPOCHS", 50))
LEARNING_RATE = 1e-3
SEED = 42
FOCAL_GAMMA = 2.0
USE_FOCAL_LOSS = True                # legacy flag; --loss overrides
MAX_TRAIN_WINDOWS = int(os.environ.get("IDS_MAX_TRAIN", 3_000_000))   # run_all.sh lowers this to fit the time budget
MAX_EVAL_WINDOWS = int(os.environ.get("IDS_MAX_EVAL", 500_000))

# Window-level SMOTE (05_train.py --balance smote)
PATIENCE = int(os.environ.get("IDS_PATIENCE", 6))        # early-stopping patience (epochs)
GBM_MAX_TRAIN = int(os.environ.get("IDS_GBM_MAX", 1_000_000))   # windows used to fit the window-stats GBM
SMOTE_TARGET_FRAC = 0.5      # minority classes grown to this share of the majority count
SMOTE_SOURCE_CAP = int(os.environ.get("IDS_SMOTE_SRC", 30_000))    # real minority windows fed to SMOTE (neighbour search is O(n^2))
SMOTE_MAJOR_CAP = int(os.environ.get("IDS_SMOTE_MAJ", 400_000))    # benign windows kept in the SMOTE set (RAM)

# ----------------------------------------------------------------------
# Legacy names, kept so old scripts (06_eval.py, serve/app.py) still import
# ----------------------------------------------------------------------
LABEL_MAP = {}
PRE_ATTACK_FRACTION = 0.0
PRE_ATTACK_MAX_FLOWS = 0
LEAK_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp", "Label",
                "__y", "__blk", "__split"]

# CSV reading: rows per chunk (smaller = less RAM; run_all.py lowers it after a memory failure)
CHUNK_ROWS = int(os.environ.get("IDS_CHUNK_ROWS", 200_000))
