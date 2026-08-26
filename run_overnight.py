"""
Unattended pipeline: wait for the running job, then do everything left.

  1. wait for the stage-0 Q&A run to exit
  2. extract Mises_Human_Action.pdf (held back to protect that run's memory)
  3. probe block sizes until one fits in 4 GB of VRAM
  4. stage 1 -- train on book prose        -> adapters/economics_books
  5. stage 2 -- Q&A on top of stage 1      -> adapters/economics_both

Every step appends to logs/overnight.log with timestamps. Any step that fails
is recorded and the script moves on where it safely can, so a failure at 3am
does not silently waste the rest of the night.

    python run_overnight.py [pid_to_wait_for]
"""

import ctypes
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
import subprocess
import sys
import time
from datetime import datetime

ROOT = _ROOT
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG_DIR = os.path.join(ROOT, "logs")
LOG = os.path.join(LOG_DIR, "overnight.log")
MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-3B-Instruct")
BOOKS_ADAPTER = os.path.join(ROOT, "adapters", "economics_books")
BOTH_ADAPTER = os.path.join(ROOT, "adapters", "economics_both")

BLOCK_CANDIDATES = [384, 256, 192]


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def alive(pid):
    """True while the process is still running (Windows)."""
    PROCESS_QUERY = 0x1000
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY, False, int(pid))
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259  # STILL_ACTIVE


def run(name, args, env_extra, log_name):
    """Run a step, streaming its output to its own log file."""
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    path = os.path.join(LOG_DIR, log_name)
    log(f"START {name}  -> logs/{log_name}")
    t0 = time.time()
    with open(path, "w", encoding="utf-8") as out:
        rc = subprocess.call([PY, "-u"] + args, stdout=out,
                             stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    mins = (time.time() - t0) / 60
    if rc == 0:
        log(f"OK    {name}  ({mins:.1f} min)")
    else:
        log(f"FAIL  {name}  exit={rc}  ({mins:.1f} min) -- see logs/{log_name}")
    return rc


def tail(path, n=12):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return [l.rstrip() for l in f.readlines()[-n:]]
    except OSError:
        return []


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log("=" * 60)
    log("overnight pipeline starting")

    # 1 -----------------------------------------------------------------
    if len(sys.argv) > 1:
        pid = int(sys.argv[1])
        if alive(pid):
            log(f"waiting for stage-0 pid {pid} to finish...")
            while alive(pid):
                time.sleep(20)
            log("stage 0 exited")
            time.sleep(20)          # let VRAM/commit settle
        else:
            log(f"stage-0 pid {pid} already finished")

    # 2 -----------------------------------------------------------------
    run("extract Human Action", ["prepare_books.py", "Mises_Human_Action.pdf"],
        {}, "extract_human_action.log")

    # 3 -----------------------------------------------------------------
    block = None
    for candidate in BLOCK_CANDIDATES:
        rc = run(f"VRAM probe block={candidate}", ["train.py"],
                 {"QWEN_MODEL": MODEL, "TRAIN_MODE": "text",
                  "BLOCK_SIZE": candidate, "MAX_STEPS": 2, "LOG_EVERY": 1,
                  "EVAL_FRACTION": 0, "OUTPUT_DIR": BOOKS_ADAPTER + "_probe"},
                 f"probe_{candidate}.log")
        if rc == 0:
            block = candidate
            log(f"block size {candidate} fits")
            break
        log(f"block size {candidate} did not fit, trying smaller")
    if block is None:
        log("ABORT: no block size fit in VRAM")
        return

    # 4 -----------------------------------------------------------------
    rc = run(f"STAGE 1 books (block={block}, 2 epochs)", ["train.py"],
             {"QWEN_MODEL": MODEL, "TRAIN_MODE": "text", "BLOCK_SIZE": block,
              "EPOCHS": 2, "LOG_EVERY": 10, "OUTPUT_DIR": BOOKS_ADAPTER},
             "stage1_books.log")
    if rc != 0:
        log("ABORT: stage 1 failed, not starting stage 2")
        return

    # 5 -----------------------------------------------------------------
    # Lower LR so instruction tuning does not wash out the prose style.
    run("STAGE 2 Q&A on top (1 epoch, lr 1e-4)", ["train.py"],
        {"QWEN_MODEL": MODEL, "TRAIN_MODE": "chat", "EPOCHS": 1,
         "LEARNING_RATE": 1e-4, "LOG_EVERY": 10,
         "RESUME_ADAPTER": BOOKS_ADAPTER, "OUTPUT_DIR": BOTH_ADAPTER},
        "stage2_both.log")

    log("-" * 60)
    log("SUMMARY")
    for name, f in (("stage 1 books", "stage1_books.log"),
                    ("stage 2 both", "stage2_both.log")):
        log(f"  {name}:")
        for line in tail(os.path.join(LOG_DIR, f), 6):
            if line.strip():
                log(f"    {line}")
    log("overnight pipeline done")


if __name__ == "__main__":
    main()
