#!/usr/bin/env bash
# Download ONLY the processed CSVs from the CSE-CIC-IDS2018 public bucket.
#
# The full bucket is ~220 GB (PCAPs + logs). The ML-ready CSVs are ~6.5 GB
# and live under "Processed Traffic Data for ML Algorithms/".
#
# Citation required by UNB if you use this data:
#   Sharafaldin, Lashkari & Ghorbani, "Toward Generating a New Intrusion
#   Detection Dataset and Intrusion Traffic Characterization", ICISSP 2018.
#   https://www.unb.ca/cic/datasets/ids-2018.html

set -euo pipefail

DEST="${1:-/data/cicids2018/raw}"
REGION="ca-central-1"   # bucket's home region - use this for fastest sync

mkdir -p "$DEST"

echo ">> Listing available objects (no credentials needed)..."
aws s3 ls --no-sign-request --region "$REGION" \
  "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"

echo
echo ">> Syncing CSVs to $DEST ..."
aws s3 sync --no-sign-request --region "$REGION" \
  "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" \
  "$DEST"

echo
echo ">> Done. Files:"
ls -lh "$DEST"
du -sh "$DEST"

# ----------------------------------------------------------------------
# EC2 notes
# ----------------------------------------------------------------------
# Launch in ca-central-1 so this transfer stays in-region (free + fast).
#
#   Training instance : g5.xlarge  (A10G 24 GB, 16 GB RAM)  ~$1.0/hr
#   Budget option     : g4dn.xlarge (T4 16 GB, 16 GB RAM)   ~$0.5/hr
#   Storage           : 250 GB gp3 EBS volume
#   AMI               : Deep Learning AMI (Ubuntu 22.04) - TF preinstalled
#
# Use a SPOT instance for training runs and checkpoint every epoch to S3;
# it cuts the cost by roughly 70%. Stop the instance between sessions -
# you pay for EBS either way but not for the GPU.
