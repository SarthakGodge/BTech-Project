# Hybrid CNN + LSTM IDS: No Attack / Pre-Attack / Attack
CIC-IDS2017 + CSE-CIC-IDS2018 · leakage-free block split · baselines, ablations, cross-dataset and leakage experiments

## One command

```bash
git clone https://github.com/SarthakGodge/BTech-Project.git && cd BTech-Project
export IDS_DATA_ROOT=/data              # needs ~60 GB free
bash run_all.sh
```

That is all. It sets up the Python environment, checks the machine and **both datasets** (fails in
seconds, before wasting time, if something is wrong), then trains everything in the background.
You can close the terminal. Later:

```bash
bash run_all.sh status     # progress, what is running, what failed
bash run_all.sh log        # live log
bash run_all.sh gpu        # GPU / RAM use now (also logged every minute to logs/resources.log)
```

When it finishes it writes `artifacts/.progress/FINISHED.txt` and **`results_bundle.zip`**
(copy that to your laptop: `RESULTS.md`, `summary.csv`, tables, figures, per-run reports, logs).

### Data it needs
| Folder | Content |
|---|---|
| `$IDS_DATA_ROOT/cicids2018/raw/*.csv` | CSE-CIC-IDS2018 (if empty, `00_download.sh` is run automatically) |
| `$IDS_DATA_ROOT/cicids2017/raw/*.csv` | CIC-IDS2017, copy it there yourself (`scp`); zips and sub-folders are handled |

Use the 2017 version that has a **`Timestamp` column** (the "GeneratedLabelledFlows" / "TrafficLabelling"
CSVs, not the shuffled "MachineLearningCVE" ones) and put only ONE version in the folder.
Without timestamps the flows cannot be put in time order and windows are meaningless; the checks
warn about this. Corrected releases (Engelen et al., Liu et al.) are better still.

## What "cannot break" means here
* **Checks first:** packages, GPU (training without one is refused unless `ALLOW_CPU=1`), disk space,
  both datasets present with readable columns and labels. Nothing starts if a check fails.
* **Every step is resumable.** Finished steps leave a marker and their output files. Re-running
  `bash run_all.sh` after any interruption (SSH drop, reboot, crash) continues where it stopped.
* **Failed steps are retried**, up to 3 passes. After a memory failure (OOM / killed) the retry uses
  smaller memory settings and the step is marked `degraded` in the status table (so you know those numbers
  are not strictly comparable; usually it will not happen).
* **One failed optional step never stops the rest.** Only the data steps are critical.
* **Disk stays small:** big temporary window files are deleted as soon as nothing needs them.
* **Always ends with results:** the bundle is written even if some steps failed, and says which.
* Not protected against: the instance being stopped or terminated by AWS (start it and run the command
  again), running out of AWS credits, or a bug that only shows on your real data (read the first lines
  of `logs/prepare.log` once the data step is done).

## What gets trained (per seed; `SEEDS="1 2 3"` by default)
| Run | Meaning |
|---|---|
| `lgbm` | LightGBM on single flows (baseline) |
| `gbm` | LightGBM on 20-flow window statistics |
| `none`, `weights`, `sampler`, `smote` | CNN+LSTM with 4 imbalance strategies |
| `attn` | multi-scale CNN + BiGRU + attention |
| `ens`, `ensattn` | probability blends of CNN and `gbm` (weights chosen on validation only) |
| `rand_weights`, `rand_gbm` | same models on a **random window split** (leaky) to quantify inflation |
| `train2018_test2017`, `train2017_test2018` (+ `gbm_...`, `lgbm_...`) | cross-dataset generalisation |

`artifacts/results/summary.md` has the paper tables (Table 1 pooled, Table 2 cross-dataset,
Table 3 random vs block split with the inflation numbers). Use fewer seeds for a shorter run:
`SEEDS="1" bash run_all.sh`. Time depends on your GPU; budget roughly 1 hour per CNN run.

## Files
| File | Role |
|---|---|
| `run_all.sh` / `run_all.py` | the single command and the self-healing runner |
| `preflight.py` | checks before, between and after steps |
| `make_bundle.py` | builds `results_bundle.zip` |
| `01_prepare.py` | harmonise both datasets, labels, dedup, block split, features, scaler (chunked reading) |
| `04_sequences.py` | windows inside blocks; `--holdout`, `--portscan-as`, `--split random` |
| `05_train.py` | CNN+LSTM (`--arch base/attn`, `--balance ...`, `--seed`), val-tuned class scales |
| `07_baseline.py`, `08_window_gbm.py`, `09_ensemble.py` | tree baselines and the blend |
| `06_eval.py`, `collect_results.py` | figures; tables with mean ± std |
| `models.py`, `evalutil.py`, `common.py`, `config.py` | models, metrics, schema aliases, settings |
| `serve/app.py`, `Dockerfile` | inference API (see "Serving") |

Delete the old `01_ingest.py`, `02_labels.py`, `03_features.py` and the root `app.py` if they are still in your repo.

## Method in one paragraph
Flows from both datasets are mapped to one schema. Labels: Benign → No Attack; Infiltration and PortScan →
Pre-Attack (reconnaissance / early stage); everything else → Attack (`LABEL_RULES` in `config.py`). Each file is
time-sorted, exact duplicates removed, and cut into blocks of 4,000 flows; blocks are grouped by their most severe
label and split 70/10/20 into train/val/test. Windows (T=20, stride 2) are built inside a block and labelled by
their last flow. Feature selection and the scaler use TRAIN rows only. Validation blocks decide early stopping,
class scales and blend weights; test blocks are only scored.

## How to read the results
* Compare models on **Pre-Attack average precision** and **macro-F1**, not accuracy (benign dominates accuracy).
* The seeds change model initialisation and sampling, **not the split**, so ± reflects training noise only.
* PortScan exists only in 2017: in cross-dataset runs it is counted as Attack and Pre-Attack means Infiltration only.
  2017 has very few Infiltration flows, so train2017 → test2018 cannot learn Pre-Attack: report it as No Attack vs Attack.
* Known limits to state in the paper: adjacent blocks (no gap between train and test blocks), only exact duplicates
  removed, label errors in the original datasets, flows of many hosts interleaved in a window.

## Serving
`serve/app.py` loads `artifacts/saved_model*` and `scaler.pkl`. Training windows are time-ordered flows across all
hosts, so it keeps one global buffer by default (`GLOBAL_BUFFER=1`). `common.py` must sit next to it (the Dockerfile copies it).
`uvicorn serve.app:app --port 8000` (choose the model with `IDS_SAVED_MODEL=saved_model_weights_s1`).

## Citation (required by UNB)
Sharafaldin, Lashkari, Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
Characterization", ICISSP 2018 (CIC-IDS2017 and CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/).
If you use corrected releases, cite Engelen et al. / Liu et al.
