"""Active health check for the multi-seed LTH pipelines.

Catches failure modes that produce NO log output (the dangerous kind):
  - orchestrator process died
  - a chain's Modal app is a zombie (detached, listed alive, but Tasks=0
    because its container was killed, e.g. by a workspace spend limit)
  - a seed has produced no new results.json for hours (silent stall)
  - local GPU idle while a local run is expected

Prints one PROBLEM line per issue and exits 1 if any were found, else exits 0.

Usage: python healthcheck.py [--stall-hours 4]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (67, 68, 69)


def sh(cmd, env=None, timeout=120):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace", env=env, cwd=HERE).stdout or ""
    except Exception as e:
        return f"__ERROR__ {e}"


def running_processes():
    out = sh(["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"])
    if "__ERROR__" in out:
        out = sh(["powershell", "-NoProfile", "-Command",
                  "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
                  "| Select-Object -ExpandProperty CommandLine"])
    return out


def app_task_count(profile, app_id):
    """(alive_state, task_count) for one app id; task_count None if unknown."""
    env = {**os.environ, "MODAL_PROFILE": profile, "PYTHONIOENCODING": "utf-8"}
    out = sh(["modal", "app", "list"], env=env, timeout=180)
    if app_id not in out:
        return False, 0
    idx = out.index(app_id)
    row = out[idx:idx + 400]
    alive = ("ephemeral" in row.lower()) or ("running" in row.lower())
    cells = [c.strip() for c in row.replace("│", "|").split("|")]
    tasks = next((int(c) for c in cells if c.isdigit()), None)
    return alive, tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stall-hours", type=float, default=4.0)
    args = ap.parse_args()

    problems = []
    stall_candidates = {}
    healthy_apps = {s: 0 for s in SEEDS}
    now = time.time()
    procs = running_processes()

    # ---- 1. progress + stall detection ----
    print("== progress ==")
    for seed in SEEDS:
        paths = glob.glob(os.path.join(HERE, f"checkpoints/seed_{seed}/*/**/results.json"),
                          recursive=True)
        newest = max((os.path.getmtime(p) for p in paths), default=0)
        age = (now - newest) / 3600 if newest else None
        per = {}
        for p in paths:
            try:
                per[json.load(open(p))["config"]] = per.get(json.load(open(p))["config"], 0) + 1
            except Exception:
                pass
        age_s = f"{age:.1f}h ago" if age is not None else "never"
        print(f"  seed {seed}: {len(paths)}/68 done, newest result {age_s}  {per}")
        remaining = 68 - len(paths)
        if remaining > 0 and age is not None and age > args.stall_hours:
            # only a real problem if nothing is actually running for this seed;
            # resolved after the liveness pass below
            stall_candidates[seed] = (f"seed {seed}: no new result in {age:.1f}h "
                                      f"({remaining} runs left) and nothing running")

    # ---- 2. orchestrator liveness ----
    print("== processes ==")
    for seed in SEEDS:
        want_modal = f"--seed {seed}"
        has_modal = bool(re.search(r"modal_run_seed\.py\s+--seed\s+" + str(seed), procs))
        done = len(glob.glob(os.path.join(HERE, f"checkpoints/seed_{seed}/*/**/results.json"),
                             recursive=True))
        print(f"  seed {seed}: modal orchestrator {'UP' if has_modal else 'down'}")
        if not has_modal and done < 68:
            problems.append(f"seed {seed}: modal orchestrator NOT RUNNING ({done}/68 done)")
    has_local = bool(re.search(r"run_seed\.py\s+--seed", procs)) and "modal_run_seed" not in procs.split("run_seed.py")[0][-40:]
    print(f"  local run_seed present: {bool(re.search(r'(?<!modal_)run_seed[.]py --seed', procs))}")

    # ---- 3. zombie apps (the silent killer) ----
    print("== modal app liveness ==")
    for seed in SEEDS:
        state_path = os.path.join(HERE, f"checkpoints/seed_{seed}/modal_state.json")
        if not os.path.exists(state_path):
            continue
        try:
            state = json.load(open(state_path))
        except Exception:
            continue
        # a freshly spawned app legitimately reports 0 tasks while Modal queues
        # the container and pulls the image; only treat 0 tasks as dead once the
        # spawn (state-file write) is well in the past
        spawn_age = now - os.path.getmtime(state_path)
        for cfg, e in state.items():
            if e.get("state") != "running" or not e.get("app_id"):
                continue
            alive, tasks = app_task_count(e["profile"], e["app_id"])
            starting = (tasks == 0) and spawn_age < 900
            tag = "OK" if (alive and (tasks is None or tasks > 0)) else \
                  ("starting (0 tasks, spawned %.0fm ago)" % (spawn_age / 60) if starting else
                   "ZOMBIE (0 tasks)" if alive else "app gone")
            print(f"  seed {seed} {cfg} r{e.get('round')} on {e['profile']}: {tag}")
            if starting or (alive and (tasks is None or tasks > 0)):
                healthy_apps[seed] += 1
            if starting:
                continue
            if alive and tasks == 0:
                problems.append(f"seed {seed} {cfg} round {e.get('round')}: ZOMBIE app "
                                f"{e['app_id']} on {e['profile']} (detached, 0 tasks) - "
                                f"stop it so the orchestrator respawns")
            elif not alive:
                problems.append(f"seed {seed} {cfg} round {e.get('round')}: app "
                                f"{e['app_id']} gone from {e['profile']} - awaiting respawn")

    # ---- 4. local GPU ----
    gpu = sh(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
              "--format=csv,noheader"]).strip()
    print(f"== local gpu ==\n  {gpu}")

    for seed, msg in stall_candidates.items():
        if healthy_apps[seed] == 0:
            problems.append(msg)

    print("\n== verdict ==")
    if problems:
        for p in problems:
            print(f"  PROBLEM: {p}")
        raise SystemExit(1)
    print("  all clear")


if __name__ == "__main__":
    main()
