#!/usr/bin/env python3
"""
Self-healing, resumable experiment runner.  Normally started through run_all.sh.

  python run_all.py                  run / resume the whole pipeline
  python run_all.py --status         show progress
  python run_all.py --fresh          forget progress (does not delete data)
  python run_all.py --normalize-data link nested CSVs / unzip archives, then exit

What it guarantees
  * Every step is a "stage". A finished stage leaves a marker AND its output files;
    re-running the script skips finished stages, so any interruption (SSH drop,
    reboot, crash) just resumes where it stopped.
  * A failed stage is retried. After a memory-type failure (OOM / Killed / exit 137) the
    retry uses smaller memory settings (LEVELS below); this is recorded in the status
    table as "degraded" so you know which numbers are not strictly comparable.
  * Several passes over the unfinished stages (default 3), so a transient problem
    (full disk freed up, GPU hiccup) heals itself.
  * A failure in an optional stage never stops the others. Only the data steps
    (prepare, seq_pooled) are critical, because nothing can run without them.
  * Large temporary window files are deleted as soon as nothing needs them.
  * At the end it always writes the results bundle (results_bundle.zip), even when
    some stages failed, and says exactly which ones.

Environment: SEEDS="1 2 3"  MAX_PASSES=3  ATTEMPTS_PER_PASS=2  STAGE_TIMEOUT_H=12
"""
import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

ART = C.ARTIFACT_DIR
PROG = ART / ".progress"
DONE = PROG / "done"
LOGS = HERE / "logs"
for _d in (PROG, DONE, LOGS):
    _d.mkdir(parents=True, exist_ok=True)
STATUS = PROG / "status.tsv"
STATE = PROG / "state.json"
PIDFILE = PROG / "run.pid"
CURRENT = PROG / "current.json"
FINISHED = PROG / "FINISHED.txt"
MAIN_LOG = LOGS / "run_all.log"
PY = sys.executable

# Settings used after a memory-type failure (level 0 = normal, nothing changed).
LEVELS = [
    {},
    {"IDS_MAX_TRAIN": "1500000", "IDS_SMOTE_SRC": "15000", "IDS_SMOTE_MAJ": "200000",
     "IDS_MAX_EVAL": "300000", "IDS_GBM_MAX": "600000", "IDS_CHUNK_ROWS": "100000"},
    {"IDS_MAX_TRAIN": "700000", "IDS_SMOTE_SRC": "8000", "IDS_SMOTE_MAJ": "80000",
     "IDS_MAX_EVAL": "150000", "IDS_GBM_MAX": "300000", "IDS_CHUNK_ROWS": "40000"},
]
MEM_PATTERNS = ("MemoryError", "Killed", "out of memory", "Out of memory", "OOM",
                "ResourceExhausted", "bad_alloc", "Cannot allocate memory",
                "CUDA_ERROR_OUT_OF_MEMORY", "Failed to allocate", "OutOfMemory")
DISK_PATTERNS = ("No space left on device", "OSError: [Errno 28]")

SEEDS = [int(x) for x in os.environ.get("SEEDS", "1 2 3").split()]
MAX_PASSES = int(os.environ.get("MAX_PASSES", 3))
ATTEMPTS_PER_PASS = int(os.environ.get("ATTEMPTS_PER_PASS", 2))
TIMEOUT_S = float(os.environ.get("STAGE_TIMEOUT_H", 12)) * 3600
PASS_SLEEP = int(os.environ.get("PASS_SLEEP_S", 60))
INJECT = {}      # test hook: --inject-failure name:count[:mem]


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(MAIN_LOG, "a") as fh:
        fh.write(line + "\n")


# ------------------------------------------------------------------ state
STATE_D = json.loads(STATE.read_text()) if STATE.exists() else {}


def st_of(name):
    return STATE_D.setdefault(name, {"attempts": 0, "level": 0, "status": "pending",
                                     "seconds": 0.0, "note": "", "fatal": False})


def save_state(order=None):
    STATE.write_text(json.dumps(STATE_D, indent=1))
    names = order or list(STATE_D)
    with open(STATUS, "w") as fh:
        for n in names:
            s = st_of(n)
            fh.write(f"{n}\t{s['status']}\t{int(s['seconds'])}\t{s['note']}\n")


# ------------------------------------------------------------------ stages
class Stage:
    def __init__(self, name, cmds=None, fn=None, outputs=(), needs=(), critical=False,
                 cond=None, lazy_for=(), after=(), always_run=False):
        self.name, self.cmds, self.fn = name, cmds or [], fn
        self.outputs, self.needs = [Path(p) for p in outputs], [Path(p) for p in needs]
        self.critical, self.cond = critical, cond
        self.lazy_for, self.after, self.always_run = list(lazy_for), list(after), always_run


BY_NAME = {}


def applicable(stg):
    if stg.cond is None:
        return True, ""
    return stg.cond()


def complete(stg):
    return (DONE / stg.name).exists() and all(p.exists() for p in stg.outputs)


def done_like(stg):
    """Finished, not applicable, or (for temporary data) no longer needed."""
    if not applicable(stg)[0]:
        return True
    if complete(stg):
        return True
    if stg.lazy_for and all(done_like(BY_NAME[n]) for n in stg.lazy_for):
        return True
    return False


def mark_done(stg):
    (DONE / stg.name).write_text(now())


def tail(path, n=20000):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - n))
            return fh.read().decode("utf-8", "replace")
    except Exception:
        return ""


def run_cmd(cmd, name, env, timeout_s):
    logf = LOGS / f"{name}.log"
    shown = {k: v for k, v in env.items() if k.startswith("IDS_") and k in sum((list(l) for l in LEVELS), [])}
    with open(logf, "a") as fh:
        fh.write(f"\n===== {now()} $ {' '.join(map(str, cmd))}   overrides={shown}\n")
        fh.flush()
        proc = subprocess.Popen([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            fh.write(f"\n!! TIMEOUT after {timeout_s / 3600:.1f} h - killed\n")
            rc = -999
    return rc, tail(logf)


def execute(stg, max_attempts):
    """Try a stage up to max_attempts times. Returns True when it is complete."""
    s = st_of(stg.name)
    t_start = time.time()
    for attempt in range(1, max_attempts + 1):
        level = s["level"]
        env = dict(os.environ, PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="2")
        env.update(LEVELS[level])
        s["attempts"] += 1
        s["status"] = "running"
        s["note"] = f"attempt {s['attempts']}" + (f", degraded L{level}" if level else "")
        save_state(ORDER)
        CURRENT.write_text(json.dumps({"stage": stg.name, "since": now(), "log": f"logs/{stg.name}.log"}))
        log(f"START {stg.name} (attempt {s['attempts']}, level {level})")

        ok, deterministic, mem_like, why = True, False, False, ""
        inj = INJECT.get(stg.name)
        if inj and inj["left"] > 0:
            inj["left"] -= 1
            ok, mem_like, why = False, inj["mem"], "injected test failure"
            log(f"  (test hook) injected failure for {stg.name}")
        elif stg.fn is not None:
            try:
                stg.fn()
            except Exception as e:                           # noqa: BLE001
                ok, why = False, f"{type(e).__name__}: {e}"
        else:
            for cmd in stg.cmds:
                rc, out = run_cmd(cmd, stg.name, env, TIMEOUT_S)
                if rc != 0:
                    ok = False
                    deterministic = len(cmd) > 1 and Path(str(cmd[1])).name == "preflight.py"
                    mem_like = rc in (-9, 137) or any(p in out for p in MEM_PATTERNS)
                    disk_full = any(p in out for p in DISK_PATTERNS)
                    why = f"exit {rc}" + (" (memory)" if mem_like else "") + (" (DISK FULL)" if disk_full else "")
                    break
        if ok and not all(p.exists() for p in stg.outputs):
            ok, why = False, "finished but expected output files are missing"
        if ok:
            mark_done(stg)
            s["status"] = "done"
            s["seconds"] += time.time() - t_start
            parts = []
            if level:
                parts.append(f"degraded L{level}: smaller memory settings were used")
            if s["attempts"] > 1:
                parts.append(f"{s['attempts']} attempts")
            s["note"] = ", ".join(parts)
            save_state(ORDER)
            log(f"DONE  {stg.name} in {int(time.time() - t_start)}s {s['note']}")
            return True
        s["note"] = f"failed: {why}"
        log(f"FAIL  {stg.name}: {why}  (see logs/{stg.name}.log)")
        if deterministic:
            s["status"], s["fatal"] = "FAILED", True
            s["seconds"] += time.time() - t_start
            save_state(ORDER)
            return False
        if mem_like and s["level"] < len(LEVELS) - 1:
            s["level"] += 1
            log(f"  memory-type failure: next attempt uses smaller settings (level {s['level']})")
        if attempt < max_attempts:
            time.sleep(min(60, 15 * attempt))
    s["status"] = "FAILED"
    s["seconds"] += time.time() - t_start
    save_state(ORDER)
    return False


# --------------------------------------------------------------- data helpers
def datasets_present():
    return {f.name.split("__")[0] for f in C.PARQUET_DIR.glob("*.parquet")}


def both_datasets():
    have = datasets_present()
    ok = all(d in have for d in C.DATASET_DIRS)
    return ok, "" if ok else "needs both datasets (only one was ingested)"


def rm_dir(p):
    def _f():
        if Path(p).exists():
            shutil.rmtree(p)
    return _f


def normalize_data():
    """Make the CSVs visible at the top level of each dataset folder: unzip archives,
    and link CSVs found in sub-folders. Never deletes or overwrites anything."""
    for ds, folder in C.DATASET_DIRS.items():
        folder = Path(folder)
        if not folder.exists():
            continue
        for z in sorted(folder.glob("*.zip")):
            target = folder / f"_unzipped_{z.stem}"
            if not target.exists():
                try:
                    log(f"[{ds}] extracting {z.name}")
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(target)
                except Exception as e:                       # noqa: BLE001
                    log(f"[{ds}] could not extract {z.name}: {e}")
        top = {p.name for p in folder.glob("*.csv")}
        nested = [p for p in folder.rglob("*.csv") if p.parent != folder]
        for p in nested:
            if p.name in top:
                continue
            link = folder / p.name
            try:
                link.symlink_to(p.resolve())
                top.add(p.name)
            except Exception:                                # noqa: BLE001
                pass
        n = len(list(folder.glob("*.csv")))
        if n:
            log(f"[{ds}] {n} CSV files visible in {folder}")


def download_2018():
    d = Path(C.DATASET_DIRS["cic2018"])
    d.mkdir(parents=True, exist_ok=True)
    if list(d.glob("*.csv")) or list(d.rglob("*.csv")):
        return
    script = HERE / "00_download.sh"
    if not script.exists():
        raise RuntimeError("no 2018 CSVs and no 00_download.sh to fetch them")
    for attempt in range(1, 4):
        log(f"downloading CSE-CIC-IDS2018 (attempt {attempt}) ...")
        with open(LOGS / "download2018.log", "a") as fh:
            rc = subprocess.call(["bash", str(script), str(d)], stdout=fh, stderr=subprocess.STDOUT)
        normalize_data()
        if list(d.glob("*.csv")):
            return
        time.sleep(30)
    raise RuntimeError("download of CSE-CIC-IDS2018 failed - see logs/download2018.log")


# ------------------------------------------------------------------ pipeline
def build(seeds):
    A = lambda *p: ART.joinpath(*p)                                       # noqa: E731
    seq_files = lambda d: [Path(d) / f"{k}_{s}.npy" for k in "Xy" for s in ("train", "val", "test")]  # noqa: E731
    prep_out = [A("scaler.pkl"), A("selected_features.json")]
    pooled = seq_files(C.SEQ_DIR)
    S = []

    S.append(Stage("prepare", cmds=[[PY, "01_prepare.py"], [PY, "preflight.py", "prepared"]],
                   outputs=prep_out, critical=True))
    S.append(Stage("seq_pooled", cmds=[[PY, "04_sequences.py"], [PY, "preflight.py", "sequences"]],
                   outputs=pooled, needs=prep_out, critical=True))

    for s in seeds:
        S.append(Stage(f"lgbm_s{s}", cmds=[[PY, "07_baseline.py", "--seed", s, "--tag", f"lgbm_s{s}"]],
                       outputs=[A(f"report_lgbm_s{s}.json")], needs=prep_out))
        S.append(Stage(f"gbm_s{s}", cmds=[[PY, "08_window_gbm.py", "--seed", s, "--tag", f"gbm_s{s}"]],
                       outputs=[A(f"report_gbm_s{s}.json"), A(f"test_probs_gbm_s{s}.npy"),
                                A(f"val_probs_gbm_s{s}.npy")], needs=pooled))
        for m in ("none", "weights", "sampler", "smote"):
            S.append(Stage(f"{m}_s{s}", cmds=[[PY, "05_train.py", "--balance", m, "--seed", s, "--tag", f"{m}_s{s}"]],
                           outputs=[A(f"report_{m}_s{s}.json"), A(f"test_probs_{m}_s{s}.npy"),
                                    A(f"val_probs_{m}_s{s}.npy")], needs=pooled))
        S.append(Stage(f"attn_s{s}", cmds=[[PY, "05_train.py", "--arch", "attn", "--balance", "weights",
                                            "--seed", s, "--tag", f"attn_s{s}"]],
                       outputs=[A(f"report_attn_s{s}.json"), A(f"test_probs_attn_s{s}.npy"),
                                A(f"val_probs_attn_s{s}.npy")], needs=pooled))
        for nm, mem in ((f"ens_s{s}", f"weights_s{s}"), (f"ensattn_s{s}", f"attn_s{s}")):
            S.append(Stage(nm, cmds=[[PY, "09_ensemble.py", "--members", mem, f"gbm_s{s}", "--tag", nm]],
                           outputs=[A(f"report_{nm}.json")],
                           needs=[A(f"{k}_probs_{t}.npy") for k in ("val", "test") for t in (mem, f"gbm_s{s}")]))

    # leakage experiment: random window split (what many papers do) vs the block split
    rdir = C.SEQ_DIR / "random_split"
    rnames = [f"rand_{k}_s{s}" for s in seeds for k in ("weights", "gbm")]
    S.append(Stage("seq_random", cmds=[[PY, "04_sequences.py", "--split", "random"]],
                   outputs=seq_files(rdir), needs=prep_out, lazy_for=rnames))
    for s in seeds:
        S.append(Stage(f"rand_weights_s{s}", cmds=[[PY, "05_train.py", "--seq", "random_split", "--balance", "weights",
                                                    "--seed", s, "--tag", f"rand_weights_s{s}"]],
                       outputs=[A(f"report_rand_weights_s{s}.json")], needs=seq_files(rdir)))
        S.append(Stage(f"rand_gbm_s{s}", cmds=[[PY, "08_window_gbm.py", "--seq", "random_split", "--seed", s,
                                                "--tag", f"rand_gbm_s{s}"]],
                       outputs=[A(f"report_rand_gbm_s{s}.json")], needs=seq_files(rdir)))
    S.append(Stage("cleanup_random", fn=rm_dir(rdir), after=rnames))

    # cross-dataset generalisation (PortScan only exists in 2017 -> counted as Attack)
    for ho, src, dst in (("cic2017", "2018", "2017"), ("cic2018", "2017", "2018")):
        hdir = C.SEQ_DIR / f"holdout_{ho}_psattack"
        tags = [f"train{src}_test{dst}", f"gbm_{src}to{dst}", f"lgbm_{src}to{dst}"]
        names = [f"{t}_s{s}" for s in seeds for t in tags]
        S.append(Stage(f"seq_{ho}", cmds=[[PY, "04_sequences.py", "--holdout", ho, "--portscan-as", "attack"]],
                       outputs=seq_files(hdir), needs=prep_out, cond=both_datasets,
                       lazy_for=[n for n in names if not n.startswith("lgbm")]))
        for s in seeds:
            S.append(Stage(f"train{src}_test{dst}_s{s}",
                           cmds=[[PY, "05_train.py", "--seq", hdir.name, "--seed", s, "--tag", f"train{src}_test{dst}_s{s}"]],
                           outputs=[A(f"report_train{src}_test{dst}_s{s}.json")], needs=seq_files(hdir), cond=both_datasets))
            S.append(Stage(f"gbm_{src}to{dst}_s{s}",
                           cmds=[[PY, "08_window_gbm.py", "--seq", hdir.name, "--seed", s, "--tag", f"gbm_{src}to{dst}_s{s}"]],
                           outputs=[A(f"report_gbm_{src}to{dst}_s{s}.json")], needs=seq_files(hdir), cond=both_datasets))
            S.append(Stage(f"lgbm_{src}to{dst}_s{s}",
                           cmds=[[PY, "07_baseline.py", "--holdout", ho, "--portscan-as", "attack", "--seed", s,
                                  "--tag", f"lgbm_{src}to{dst}_s{s}"]],
                           outputs=[A(f"report_lgbm_{src}to{dst}_s{s}.json")], needs=prep_out, cond=both_datasets))
        S.append(Stage(f"cleanup_{ho}", fn=rm_dir(hdir), after=[n for n in names if not n.startswith("lgbm")],
                       cond=both_datasets))

    s0 = seeds[0]
    S.append(Stage("final_eval", cmds=[[PY, "06_eval.py", "--tag", f"weights_s{s0}"]],
                   outputs=[A("figures", f"confusion_matrix_weights_s{s0}.png")],
                   needs=[A(f"best_model_weights_s{s0}.keras")]))
    return S


def finals():
    return [Stage("final_collect", cmds=[[PY, "collect_results.py"]], always_run=True),
            Stage("final_bundle", cmds=[[PY, "make_bundle.py"]], always_run=True)]


ORDER = []


def consider(stg, per_pass):
    s = st_of(stg.name)
    if not applicable(stg)[0]:
        s["status"], s["note"] = "skipped: not applicable", applicable(stg)[1]
        return
    if complete(stg):
        s["status"] = "done"
        return
    if stg.lazy_for and done_like(stg):
        if (DONE / stg.name).exists():
            s["status"], s["note"] = "done", "temporary data removed after use"
        else:
            s["status"], s["note"] = "skipped: not needed", "temporary data, nothing left that needs it"
        return
    if stg.after:
        waiting = [n for n in stg.after if not done_like(BY_NAME[n])]
        if waiting:
            s["status"], s["note"] = "skipped: waiting", f"waiting for {waiting[0]}"
            return
    missing = [p for p in stg.needs if not p.exists()]
    if missing:
        s["status"], s["note"] = "skipped: missing input", f"missing {missing[0].name}"
        return
    if s.get("fatal"):
        return
    ok = execute(stg, per_pass)
    if not ok and stg.critical:
        raise RuntimeError(f"critical stage '{stg.name}' failed")


def kill_orphans():
    """After a crash or kill -9 of the runner, its child step can keep running. Stop such leftovers
    (only processes of THIS project directory) so a restart never runs a step twice."""
    names = ("01_prepare.py", "04_sequences.py", "05_train.py", "07_baseline.py", "08_window_gbm.py",
             "09_ensemble.py", "06_eval.py", "collect_results.py")
    me = os.getpid()
    n = 0
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            argv = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
            argv = [a.decode(errors="ignore") for a in argv if a]
            if len(argv) < 2 or not os.path.basename(argv[0]).startswith("python"):
                continue
            if os.path.basename(argv[1]) not in names:
                continue
            if os.path.realpath(f"/proc/{d}/cwd") != str(C.ROOT.resolve()):
                continue
            os.kill(int(d), signal.SIGKILL)
            n += 1
        except Exception:
            pass
    if n:
        log(f"stopped {n} leftover process(es) from an earlier interrupted run")


def resource_monitor():
    """Every 60 s append GPU use, GPU memory and RAM to logs/resources.log (check it with `bash run_all.sh gpu`)."""
    import threading

    def mem_gb():
        try:
            m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
            tot, av = m["MemTotal"] / 1048576, m["MemAvailable"] / 1048576
            return f"ram_used={tot - av:.1f}/{tot:.1f}GB"
        except Exception:
            return "ram=?"

    def gpu():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
            u, mu, mt = [x.strip() for x in o.splitlines()[0].split(",")]
            return f"gpu_util={u}% gpu_mem={mu}/{mt}MB"
        except Exception:
            return "gpu=n/a"

    def loop():
        while True:
            try:
                cur = json.loads(CURRENT.read_text()).get("stage", "-") if CURRENT.exists() else "-"
                with open(C.ROOT / "logs" / "resources.log", "a") as fh:
                    fh.write(f"[{now()}] {cur:<22} {gpu()} {mem_gb()}\n")
            except Exception:
                pass
            time.sleep(60)
    threading.Thread(target=loop, daemon=True).start()


def lock():
    if PIDFILE.exists():
        try:
            pid = int(PIDFILE.read_text())
            if pid != os.getpid() and Path(f"/proc/{pid}/cmdline").exists() \
                    and "run_all" in Path(f"/proc/{pid}/cmdline").read_text():
                print(f"already running (pid {pid}). Use: bash run_all.sh status")
                sys.exit(0)
        except Exception:                                    # noqa: BLE001
            pass
    PIDFILE.write_text(str(os.getpid()))


def print_status():
    if not STATE_D:
        print("nothing has run yet.")
        return
    counts = {}
    for n, s in STATE_D.items():
        k = s["status"].split(":")[0]
        counts[k] = counts.get(k, 0) + 1
    print("stages:", counts)
    if FINISHED.exists():
        print(FINISHED.read_text().strip())
    elif CURRENT.exists():
        cur = json.loads(CURRENT.read_text())
        print(f"running: {cur['stage']} (since {cur['since']})")
        print("last log lines:\n" + "\n".join(tail(HERE / cur["log"], 1500).splitlines()[-4:]))
    bad = [n for n, s in STATE_D.items() if s["status"] == "FAILED"]
    if bad:
        print("FAILED so far:", ", ".join(bad))
    deg = [n for n, s in STATE_D.items() if "degraded" in s.get("note", "")]
    if deg:
        print("ran with reduced memory settings:", ", ".join(deg))


def main():
    global ORDER
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--normalize-data", action="store_true")
    ap.add_argument("--inject-failure", action="append", default=[], help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.status:
        print_status()
        return 0
    if args.fresh:
        shutil.rmtree(PROG, ignore_errors=True)
        print("progress cleared (data and results are untouched).")
        return 0
    if args.normalize_data:
        normalize_data()
        return 0
    for spec in args.inject_failure:
        p = spec.split(":")
        INJECT[p[0]] = {"left": int(p[1]), "mem": len(p) > 2 and p[2] == "mem"}

    os.environ.pop("IDS_WILL_DOWNLOAD", None)   # the launcher's hint must not weaken the real check
    lock()
    kill_orphans()
    (C.ROOT / 'logs').mkdir(exist_ok=True)
    resource_monitor()
    FINISHED.unlink(missing_ok=True)
    log("=" * 70)
    log(f"run_all start | seeds {SEEDS} | data root {C.DATA_ROOT} | python {sys.version.split()[0]}")

    stages = build(SEEDS)
    fin = finals()
    BY_NAME.update({s.name: s for s in stages + fin})
    ORDER = [s.name for s in stages + fin]
    for n in ORDER:
        st_of(n)
    for v in STATE_D.values():                         # a re-run is a request to try again
        v["fatal"] = False
        if v["status"] == "running":
            v["status"] = "pending"

    # ---- fast, early, loud failures --------------------------------------------------
    normalize_data()
    try:
        download_2018()
    except Exception as e:                                   # noqa: BLE001
        log(f"2018 download: {e}")
    rc, out = run_cmd([PY, "preflight.py", "env"], "preflight_env", dict(os.environ), 1800)
    if rc != 0:
        log("PREFLIGHT FAILED - nothing was started. Fix the items marked [FAIL] in logs/preflight_env.log:")
        for line in out.splitlines():
            if "[FAIL]" in line or "FAILED" in line:
                log("   " + line.strip())
        FINISHED.write_text(f"[{now()}] STOPPED BY PREFLIGHT - see logs/preflight_env.log")
        return 1

    # ---- main loop ---------------------------------------------------------------------
    for p in range(1, MAX_PASSES + 1):
        log(f"--- pass {p}/{MAX_PASSES}")
        try:
            for stg in stages:
                consider(stg, ATTEMPTS_PER_PASS)
                save_state(ORDER)
        except RuntimeError as e:
            log(f"ABORT of this pass: {e}. Later stages need its output.")
        pending = [x.name for x in stages if not done_like(x) and not st_of(x.name).get("fatal")]
        if any(x.critical and st_of(x.name).get("fatal") for x in stages):
            log("a critical stage failed deterministically (see its log); not retrying")
            break
        if not pending:
            break
        if p < MAX_PASSES:
            log(f"{len(pending)} stage(s) unfinished; sleeping {PASS_SLEEP}s then retrying them")
            time.sleep(PASS_SLEEP)

    # ---- always: tables + bundle ---------------------------------------------------------
    for stg in fin:
        BY_NAME[stg.name] = stg
        execute(stg, 1)
    save_state(ORDER)

    failed = [n for n in ORDER if st_of(n)["status"] == "FAILED"]
    unfinished = [s.name for s in stages if not done_like(s)]
    degraded = [n for n in ORDER if "degraded" in st_of(n).get("note", "")]
    if not failed and not unfinished:
        msg = f"ALL DONE. Results: results_bundle.zip (RESULTS.md inside)."
    else:
        msg = (f"FINISHED WITH PROBLEMS. failed: {failed or '-'}; never ran: "
               f"{[n for n in unfinished if n not in failed] or '-'}. Everything else is complete and bundled "
               f"(results_bundle.zip). Re-run `bash run_all.sh` to retry only the failed steps.")
    if degraded:
        msg += f" Reduced-memory settings were used for: {degraded}."
    FINISHED.write_text(f"[{now()}] {msg}")
    CURRENT.unlink(missing_ok=True)
    log(msg)
    return 0 if not failed and not unfinished else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        try:
            if PIDFILE.exists() and PIDFILE.read_text().strip() == str(os.getpid()):
                PIDFILE.unlink()
        except Exception:                                    # noqa: BLE001
            pass
