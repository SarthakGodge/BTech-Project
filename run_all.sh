#!/usr/bin/env bash
# ONE COMMAND for the whole project, safe to run unattended and safe to re-run.
#
#   bash run_all.sh            set up, check everything, then train in the background
#   bash run_all.sh status     how far it is, what is running, what failed
#   bash run_all.sh log        follow the main log (Ctrl+C leaves it running)
#   bash run_all.sh gpu        GPU / RAM use right now and over the last minutes
#   bash run_all.sh stop       stop the run (progress is kept; start again to resume)
#   bash run_all.sh fresh      forget progress (data and results are kept), then start again
#   bash run_all.sh foreground run in this terminal instead of the background
#
# Settings (all optional):
#   IDS_DATA_ROOT=/data        where datasets / parquet / windows live (needs ~60 GB free)
#   SEEDS="1 2 3"              seeds to run (more seeds = more time = stronger statistics)
#   ALLOW_SINGLE_DATASET=1     run on one dataset if the other is missing (NOT recommended)
#   ALLOW_CPU=1                allow training without a GPU (very slow)
#
# Data it needs (it will download 2018 itself via 00_download.sh if that folder is empty):
#   $IDS_DATA_ROOT/cicids2018/raw/*.csv    CSE-CIC-IDS2018
#   $IDS_DATA_ROOT/cicids2017/raw/*.csv    CIC-IDS2017  (copy these in yourself; zips are fine)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
CMD="${1:-start}"
export IDS_DATA_ROOT="${IDS_DATA_ROOT:-/data}"
export MIN_FREE_GB="${MIN_FREE_GB:-40}"
VENV="${IDS_VENV:-$HOME/venv}"
PROG=artifacts/.progress

say() { printf '%s\n' "$*"; }
die() { printf '\nSTOPPED: %s\n' "$*" >&2; exit 1; }

activate() {
  if [ -x "$VENV/bin/python" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
  fi
}

is_running() {
  [ -f "$PROG/run.pid" ] || return 1
  local pid; pid="$(cat "$PROG/run.pid" 2>/dev/null)"
  [ -n "$pid" ] && [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q run_all
}

case "$CMD" in
  status)
    activate; python run_all.py --status; exit 0 ;;
  gpu)
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
    free -h | head -2; echo "--- last samples (logs/resources.log, one per minute) ---"; tail -n 8 logs/resources.log 2>/dev/null
    exit 0 ;;
  log)
    mkdir -p logs; touch logs/run_all.log; tail -n 40 -f logs/run_all.log; exit 0 ;;
  stop)
    if is_running; then
      pid="$(cat "$PROG/run.pid")"; kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      sleep 2; pkill -f "05_train.py|08_window_gbm.py|07_baseline.py|04_sequences.py|01_prepare.py" 2>/dev/null
      say "stopped. Run 'bash run_all.sh' to resume where it left off."
    else say "not running."; fi
    exit 0 ;;
  fresh)
    is_running && die "it is running; run 'bash run_all.sh stop' first"
    activate; python run_all.py --fresh; CMD=start ;;
  start|foreground) ;;
  *) die "unknown command '$CMD' (start | status | log | gpu | stop | fresh | foreground)" ;;
esac

if is_running; then
  say "already running (pid $(cat "$PROG/run.pid")). Use 'bash run_all.sh status' or 'bash run_all.sh log'."
  exit 0
fi

say "== 1/4 data folder: $IDS_DATA_ROOT"
if ! mkdir -p "$IDS_DATA_ROOT/cicids2018/raw" "$IDS_DATA_ROOT/cicids2017/raw" 2>/dev/null; then
  sudo -n mkdir -p "$IDS_DATA_ROOT/cicids2018/raw" "$IDS_DATA_ROOT/cicids2017/raw" \
    && sudo -n chown -R "$(id -un)" "$IDS_DATA_ROOT" \
    || die "cannot create $IDS_DATA_ROOT. Run: sudo mkdir -p $IDS_DATA_ROOT && sudo chown \$USER $IDS_DATA_ROOT"
fi

say "== 2/4 python environment ($VENV)"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" 2>/dev/null || {
    sudo -n apt-get update -qq >/dev/null 2>&1; sudo -n apt-get install -y -qq python3-venv python3-pip >/dev/null 2>&1
    python3 -m venv "$VENV" || die "cannot create a virtualenv. Run: sudo apt install -y python3-venv python3-pip"
  }
fi
activate
STAMP="$VENV/.ids_requirements_$(sha1sum requirements.txt | cut -c1-12)"
if [ ! -f "$STAMP" ]; then
  say "   installing requirements (first time takes a few minutes) ..."
  pip install -q --upgrade pip >/dev/null 2>&1
  pip install -q -r requirements.txt || die "pip install failed. Fix the error above, then run this command again."
  touch "$STAMP"
fi

say "== 3/4 checks (packages, GPU, disk, BOTH datasets)"
python run_all.py --normalize-data >/dev/null 2>&1
if [ -z "$(ls "$IDS_DATA_ROOT/cicids2018/raw"/*.csv 2>/dev/null)" ] && [ -f 00_download.sh ]; then
  export IDS_WILL_DOWNLOAD="cic2018"
  say "   (CIC-IDS2018 will be downloaded automatically after the checks)"
fi
python preflight.py env || die "the checks above failed. Nothing was started and nothing was lost. Fix the [FAIL] lines and run this command again."

mkdir -p logs
if [ "$CMD" = "foreground" ]; then
  say "== 4/4 running in this terminal"
  exec python run_all.py
fi
say "== 4/4 starting in the background (survives closing this terminal)"
if command -v setsid >/dev/null 2>&1; then
  setsid nohup python run_all.py > logs/launcher.out 2>&1 < /dev/null &
else
  nohup python run_all.py > logs/launcher.out 2>&1 < /dev/null &
fi
sleep 3
say ""
say "Started. You can close this terminal. Later:"
say "   bash run_all.sh status     progress and failures"
say "   bash run_all.sh log        live log"
say "When it finishes it writes artifacts/.progress/FINISHED.txt and results_bundle.zip."
say "If the machine restarts or anything is interrupted, just run 'bash run_all.sh' again: it resumes."
