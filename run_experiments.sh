#!/usr/bin/env bash
# Full pipeline + the experiments for the report. Run from the repo root.
# Tip: smoke-test first with  IDS_EPOCHS=3  and  MAX_TRAIN_WINDOWS=300_000 in config.py
set -e
SEEDS="${SEEDS:-1 2 3}"

python 01_prepare.py                       # both datasets -> parquet, features, scaler
python 04_sequences.py                     # pooled block split

# ---- Table 1: baselines + imbalance ablation, same data, same test blocks ----
for s in $SEEDS; do
  python 07_baseline.py --seed $s --tag lgbm_s$s
  for m in none weights sampler smote; do
    python 05_train.py --balance $m --seed $s --tag ${m}_s$s
  done
done

# ---- Table 2: cross-dataset generalisation (PortScan only exists in 2017,
#      so it is moved to Attack to keep the Pre-Attack class comparable) ----
python 04_sequences.py --holdout cic2017 --portscan-as attack
python 04_sequences.py --holdout cic2018 --portscan-as attack
for s in $SEEDS; do
  python 05_train.py --seq holdout_cic2017_psattack --seed $s --tag train2018_test2017_s$s
  python 05_train.py --seq holdout_cic2018_psattack --seed $s --tag train2017_test2018_s$s
  python 07_baseline.py --holdout cic2017 --portscan-as attack --seed $s --tag lgbm_2018to2017_s$s
  python 07_baseline.py --holdout cic2018 --portscan-as attack --seed $s --tag lgbm_2017to2018_s$s
done

python collect_results.py                  # mean +/- std table
python 06_eval.py --tag weights_s1         # figures for the headline model
