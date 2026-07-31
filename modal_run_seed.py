"""Local orchestrator: run one seed's LTH sweep on Modal, across multiple accounts.

Runs the 4 config chains (6x1, 3x2, 2x3, 1x6) in parallel as detached Modal
jobs, one round at a time per chain. Pruning happens locally (CPU, seconds);
each Modal job trains one round and writes outputs to that workspace's
"phop-lth" volume; this script polls, downloads results into the SAME local
checkpoints/ tree the local harness uses, then advances the chain.

Account rotation: profiles are tried in --profiles order. When a spawn fails
or a detached app dies with credit/billing-looking errors (or twice for any
reason), the profile is marked exhausted and the chain moves to the next one.

Fully resumable: re-run the same command after any interruption. Completed
rounds are skipped via results.json; a round that was mid-training on Modal
is re-spawned (round-level granularity).

Usage:
    python modal_run_seed.py --seed 68 --gpu L4
"""
import argparse
import json
import os
import subprocess
import sys
import time

CONFIGS = ["6x1", "3x2", "2x3", "1x6"]
DEFAULT_PROFILES = ["modalworkspace1", "modalworkspace2", "modalworkspace3",
                    "jushuang", "jushuang2006", "adambanks1", "adambanks2", "adambanks3",
                    "justin-huang", "229a1", "229a2", "229a3",
                    "cs229aa1", "cs229aa2", "cs229aa3", "gangsterkhan1162",
                    "adad1", "adad2", "adad3", "regionitems"]
CREDIT_PATTERNS = ("credit", "balance", "payment", "billing", "spending limit",
                   "insufficient", "quota", "exceeded")

HERE = os.path.dirname(os.path.abspath(__file__))


def modal_cmd(args_list, profile, gpu, timeout=None, capture=True):
    env = {**os.environ, "MODAL_PROFILE": profile, "PHOP_GPU": gpu,
           "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(["modal"] + args_list, env=env, cwd=HERE, timeout=timeout,
                          capture_output=capture, text=True, encoding="utf-8",
                          errors="replace")


def looks_like_credit_failure(text):
    t = (text or "").lower()
    return any(p in t for p in CREDIT_PATTERNS)


class Chain:
    def __init__(self, config, seed, ckpt_root, rounds, steps, profile_idx):
        self.config = config
        self.seed = seed
        self.ckpt_root = ckpt_root
        self.rounds = rounds
        self.steps = steps
        self.profile_idx = profile_idx
        self.state = "idle"        # idle | running | done
        self.round = None
        self.spawned_at = None
        self.fail_count = 0
        self.app_id = None         # modal app id of this chain's in-flight job

    # ---- local paths (identical layout to run_seed.py) ----
    def base_dir(self):
        return os.path.join(self.ckpt_root, f"seed_{self.seed}", self.config)

    def run_dir(self, r):
        return self.base_dir() if r == 0 else os.path.join(self.base_dir(), "lth", f"round_{r}")

    def next_round(self):
        for r in range(0, self.rounds + 1):
            if not os.path.exists(os.path.join(self.run_dir(r), "results.json")):
                return r
        return None

    def vol_dir(self, r):
        return f"seed_{self.seed}/{self.config}/round_{r}"


def prune_local(chain, r, prune_frac):
    from run_seed import prune_round
    run_dir = chain.run_dir(r)
    if os.path.exists(os.path.join(run_dir, "init.pt")) and \
       os.path.exists(os.path.join(run_dir, "mask.pt")):
        return
    prune_round(chain.config, r, chain.base_dir(), prune_frac)


def spawn_round(chain, r, profile, gpu):
    """Returns (ok, output, app_id). app_id is the detached modal app running
    this round, parsed from the `modal run` banner."""
    import re
    args = ["run", "--detach", "modal_phop.py::main",
            "--config", chain.config, "--round-num", str(r),
            "--seed", str(chain.seed), "--steps", str(chain.steps)]
    if r > 0:
        args += ["--init-path", os.path.join(chain.run_dir(r), "init.pt"),
                 "--mask-path", os.path.join(chain.run_dir(r), "mask.pt")]
    res = modal_cmd(args, profile, gpu, timeout=900)
    out = (res.stdout or "") + (res.stderr or "")
    if res.returncode != 0 or "SPAWNED" not in out:
        return False, out, None
    m = re.search(r"(ap-[A-Za-z0-9]+)", out)
    return True, out, (m.group(1) if m else None)


def app_id_running(profile, gpu, app_id):
    """True if this SPECIFIC app still has a live container.

    A detached app whose task was killed (e.g. workspace spend limit) stays
    listed as "ephemeral (detached)" with Tasks=0 forever. Treating that as
    alive silently stalls the chain, so require a non-zero task count.
    """
    if not app_id:
        return None
    res = modal_cmd(["app", "list"], profile, gpu, timeout=300)
    if res.returncode != 0:
        return None
    out = res.stdout or ""
    if app_id not in out:
        return False
    # the app's row may wrap over several lines; the task count is the first
    # bare integer in the cells following the state
    idx = out.index(app_id)
    row = out[idx:idx + 400]
    state_alive = ("ephemeral" in row.lower()) or ("running" in row.lower())
    if not state_alive:
        return False
    cells = [c.strip() for c in row.replace("│", "|").split("|")]
    for c in cells:
        if c.isdigit():
            return int(c) > 0
    return state_alive


def load_state(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(path, chains, profiles):
    state = {c.config: {"round": c.round, "state": c.state, "app_id": c.app_id,
                        "profile": profiles[c.profile_idx]} for c in chains}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def try_download(chain, r, profile, gpu):
    """Returns True when the round's outputs were fetched into the local tree."""
    run_dir = chain.run_dir(r)
    os.makedirs(run_dir, exist_ok=True)
    vol = chain.vol_dir(r)

    # results.json existing on the volume marks completion
    res = modal_cmd(["volume", "get", "--force", "phop-lth",
                     f"{vol}/results.json", os.path.join(run_dir, "results.json.dl")],
                    profile, gpu, timeout=300)
    if res.returncode != 0:
        return False

    files = [("final.pt", "final.pt")]
    if r == 0:
        files.append(("lth_rewind_1000.pt", None))  # goes to base_dir for pruning
    for remote_name, _ in files:
        dest = os.path.join(run_dir if remote_name != "lth_rewind_1000.pt" else chain.base_dir(),
                            remote_name)
        res = modal_cmd(["volume", "get", "--force", "phop-lth",
                         f"{vol}/{remote_name}", dest], profile, gpu, timeout=600)
        if res.returncode != 0:
            print(f"  [{chain.config}] WARNING: failed to download {remote_name}: {res.stderr}")
            return False

    os.replace(os.path.join(run_dir, "results.json.dl"),
               os.path.join(run_dir, "results.json"))
    return True


def app_running(profile, gpu):
    res = modal_cmd(["app", "list"], profile, gpu, timeout=300)
    if res.returncode != 0:
        return None  # unknown
    return ("phop-lth" in (res.stdout or "")) and \
           (("running" in res.stdout.lower()) or ("ephemeral" in res.stdout.lower()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=68)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--prune-frac", type=float, default=0.2)
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--poll-seconds", type=int, default=180)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    profiles = args.profiles.split(",")
    exhausted = set()
    configs = args.configs.split(",")
    # stagger initial profiles so chains start on different accounts
    chains = [Chain(cfg, args.seed, args.ckpt_root, args.rounds, args.steps,
                    profile_idx=i % len(profiles)) for i, cfg in enumerate(configs)]

    def chain_profile(chain):
        n = len(profiles)
        for k in range(n):
            idx = (chain.profile_idx + k) % n
            if profiles[idx] not in exhausted:
                chain.profile_idx = idx
                return profiles[idx]
        return None

    print(f"[orchestrator] seed {args.seed}, gpu {args.gpu}, "
          f"{len(configs)} chains x {args.rounds + 1} rounds, profiles: {profiles}")

    # recover in-flight jobs from the previous orchestrator via the state file
    state_path = os.path.join(args.ckpt_root, f"seed_{args.seed}", "modal_state.json")
    prev = load_state(state_path)
    for chain in chains:
        entry = prev.get(chain.config)
        if not entry or entry.get("state") != "running" or not entry.get("app_id"):
            continue
        r = chain.next_round()
        if r is None or r != entry.get("round"):
            continue
        profile = entry.get("profile")
        if profile in profiles and app_id_running(profile, args.gpu, entry["app_id"]):
            chain.profile_idx = profiles.index(profile)
            chain.round, chain.state = r, "running"
            chain.app_id, chain.spawned_at = entry["app_id"], time.time()
            print(f"[{chain.config}] recovered in-flight round {r} "
                  f"({entry['app_id']} on {profile})")

    while True:
        all_done = True
        for chain in chains:
            if chain.state == "done":
                continue
            all_done = False
            profile = chain_profile(chain)
            if profile is None:
                print(f"[{chain.config}] ALL PROFILES EXHAUSTED — chain stalled. "
                      f"Finish remaining rounds locally with run_seed.py.")
                chain.state = "done"  # stop trying
                continue

            if chain.state == "idle":
                r = chain.next_round()
                if r is None:
                    print(f"[{chain.config}] chain complete!")
                    chain.state = "done"
                    continue
                # restart-safety: the round may already be finished on the volume
                if try_download(chain, r, profile, args.gpu):
                    print(f"[{chain.config}] round {r} found complete on volume ({profile})")
                    continue
                if r > 0:
                    prune_local(chain, r, args.prune_frac)
                ok, out, app_id = spawn_round(chain, r, profile, args.gpu)
                if ok:
                    chain.round, chain.state = r, "running"
                    chain.app_id = app_id
                    chain.spawned_at, chain.fail_count = time.time(), 0
                    save_state(state_path, chains, profiles)
                    print(f"[{chain.config}] round {r} SPAWNED on {profile} "
                          f"({args.gpu}, {app_id})")
                else:
                    chain.fail_count += 1
                    tag = "credit-like" if looks_like_credit_failure(out) else "error"
                    print(f"[{chain.config}] spawn failed on {profile} ({tag}):\n{out[-500:]}")
                    if looks_like_credit_failure(out) or chain.fail_count >= 2:
                        exhausted.add(profile)
                        chain.fail_count = 0
                        print(f"[orchestrator] marking profile {profile} exhausted "
                              f"({len(exhausted)}/{len(profiles)})")

            elif chain.state == "running":
                if try_download(chain, chain.round, profile, args.gpu):
                    res = json.load(open(os.path.join(chain.run_dir(chain.round), "results.json")))
                    accs = ", ".join(f"p{p}={a:.3f}" for p, a in
                                     sorted(res["accuracy"].items(), key=lambda kv: int(kv[0])))
                    print(f"[{chain.config}] round {chain.round} DONE "
                          f"({res['wall_hours']:.1f}h on {res.get('gpu', '?')}): {accs}")
                    chain.state, chain.app_id = "idle", None
                    save_state(state_path, chains, profiles)
                elif time.time() - chain.spawned_at > 600:
                    running = app_id_running(profile, args.gpu, chain.app_id)
                    if running is False:
                        chain.fail_count += 1
                        print(f"[{chain.config}] round {chain.round} app died on {profile} "
                              f"(fail {chain.fail_count})")
                        if chain.fail_count >= 2:
                            exhausted.add(profile)
                            chain.fail_count = 0
                            print(f"[orchestrator] marking profile {profile} exhausted "
                                  f"({len(exhausted)}/{len(profiles)})")
                        chain.state, chain.app_id = "idle", None  # respawn (possibly on next profile)
                        save_state(state_path, chains, profiles)

        if all_done:
            break
        time.sleep(args.poll_seconds)

    print("[orchestrator] finished. Run `python make_table.py` for the aggregate table.")


if __name__ == "__main__":
    main()
