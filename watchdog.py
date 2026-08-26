"""
Detached watchdog for the overnight pipeline.

It cannot notify anyone directly, so instead it writes a FAIL line into
logs/overnight.log -- the file the session's monitor is already tailing. That
turns "the pipeline died silently" into an event that reaches a human.

Runs as its own OS process so it survives whatever happens to the session.

    python watchdog.py <pid> [stall_seconds]
"""

import ctypes
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
import sys
import time
from datetime import datetime

ROOT = _ROOT
OVERNIGHT_LOG = os.path.join(ROOT, "logs", "overnight.log")
STAGE_LOGS = [os.path.join(ROOT, "logs", n)
              for n in ("stage1_books.log", "stage2_both.log")]

pid = int(sys.argv[1])
STALL = int(sys.argv[2]) if len(sys.argv) > 2 else 1500


def alive(p):
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, p)
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259


def say(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    with open(OVERNIGHT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def finished():
    try:
        with open(OVERNIGHT_LOG, encoding="utf-8", errors="replace") as f:
            return "overnight pipeline done" in f.read()
    except OSError:
        return False


def newest_stage_log_age():
    ages = []
    now = time.time()
    for path in STAGE_LOGS:
        if os.path.exists(path):
            ages.append(now - os.path.getmtime(path))
    return min(ages) if ages else None


while True:
    time.sleep(60)

    if finished():
        break

    if not alive(pid):
        say(f"FAIL watchdog: pipeline process {pid} vanished before finishing")
        break

    age = newest_stage_log_age()
    if age is not None and age > STALL:
        say(f"FAIL watchdog: no training progress for {int(age)}s (hung or OOM)")
        break
