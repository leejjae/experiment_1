#!/usr/bin/env python
"""Drive the 12 WebArena-Lite streams: 4 sites x 3 arms.

Structure follows section 5.2.1 of the paper: within a stream the tasks run one after
another, and each task gets L=5 *consecutive* trials with the memory that has accumulated
so far. Streams are independent (disjoint memory roots and result dirs) so they run in
parallel; episodes inside a stream never do, because that ordering is what "continual"
means here.

    python run_streams.py --dry-run
    python run_streams.py --arms a b c --sites shopping reddit
    python run_streams.py --resume            # skip tasks that already have results

Arms (the diagonal pairing validated on ALFWorld):
    a  verbalized confidence + p     + w*A_hat   (the authors' default)
    b  token logprobs        + log p + w*A_hat   (scale-matched to the KL closed form)
    c  no memory                                 (baseline)

Retrieval r=0.8 / k=10 stay as the authors hardcoded them in jitrl_agent.py; they are not
reachable from the CLI and are deliberately left alone.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
PYTHON = sys.executable

ARMS = {
    "a": ["--logit_mode", "verbalized", "--update_rule", "prob"],
    "b": ["--logit_mode", "token", "--update_rule", "logprob"],
    "c": ["--disable_memory", "True"],
}

COMMON = [
    "--config_dir", "config_files_lite",
    "--llm_model", "qwen3-32b",
    "--llm_eval", "gpt-oss-20b",
    "--llm_extract", "gpt-oss-20b",
    "--gamma", "0.95",
    "--top_actions", "3",
    "--llm_temperature", "0.7",
    "--max_obs_chars", "24000",
    "--memory_scope", "site",
    "--headless", "True",
]


def tasks_by_site(config_dir):
    grouped = defaultdict(list)
    for path in Path(config_dir).glob("*.json"):
        if not path.stem.isdigit():
            continue
        conf = json.loads(path.read_text())
        grouped[conf["sites"][0]].append(conf["task_id"])
    return {site: sorted(ids) for site, ids in sorted(grouped.items())}


def already_done(result_dir, task_id, attempts):
    marker = Path(result_dir) / f"cumulative_results_walite_{task_id}_x{attempts}.json"
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text())
    except json.JSONDecodeError:
        return False
    # Only skip a task whose every attempt is on record. A partial file means the stream
    # died mid-task, and re-running it is cheaper than reasoning about a hole in Final SR.
    runs = {row.get("run") for row in data.get("detailed_results", [])}
    return runs == set(range(1, attempts + 1))


def run_stream(site, arm, task_ids, args):
    tag = f"{site}_{arm}"
    result_dir = HERE / args.results_root / tag
    output_path = HERE / args.output_root / f"arm_{arm}"
    log_path = HERE / args.log_root / f"{tag}.log"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    if args.sim_log:
        env["JITRL_SIM_LOG"] = str(HERE / args.log_root / f"{tag}.simlog.jsonl")

    with open(log_path, "a") as log:
        log.write(f"\n{'=' * 78}\nstream {tag}: {len(task_ids)} tasks x {args.attempts} "
                  f"attempts, started {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 78}\n")
        log.flush()
        for task_id in task_ids:
            if args.resume and already_done(result_dir, task_id, args.attempts):
                log.write(f"[skip] task {task_id} already complete\n")
                log.flush()
                continue
            cmd = [PYTHON, str(HERE / "run.py"),
                   "--task_name", f"walite.{task_id}",
                   "--repeat_runs", str(args.attempts),
                   "--max_steps", str(args.max_steps),
                   "--result_dir", str(result_dir),
                   "--output_path", str(output_path),
                   *COMMON, *ARMS[arm]]
            log.write(f"\n[task {task_id}] {' '.join(cmd)}\n")
            log.flush()
            started = time.time()
            # A stream must survive a task that crashes OR runs long: one bad task must not
            # cost the rest of the continual sequence. subprocess.call raises TimeoutExpired
            # (it does not just return a code), so the timeout has to be caught explicitly --
            # letting it propagate is what killed the reddit streams on the first pilot.
            try:
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                     cwd=str(HERE), timeout=args.task_timeout)
            except subprocess.TimeoutExpired:
                rc = "TIMEOUT"
            except Exception as e:                      # noqa: BLE001 - keep the stream alive
                rc = f"EXC:{type(e).__name__}: {e}"
            log.write(f"[task {task_id}] rc={rc} in {(time.time() - started) / 60:.1f} min\n")
            log.flush()
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["a", "b", "c"], choices=list(ARMS))
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--attempts", type=int, default=5, help="L in the paper")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--config-dir", default="config_files_lite")
    ap.add_argument("--results-root", default="results/streams")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--log-root", default="logs/streams")
    ap.add_argument("--task-timeout", type=int, default=28800,
                    help="seconds before a single task (all its attempts) is abandoned. One reddit task -- 5 attempts x 30 steps over an 84k-char page -- ran past the old 2h default.")
    ap.add_argument("--sim-log", action="store_true",
                    help="record the retrieval similarity distribution per call")
    ap.add_argument("--limit-tasks", type=int, default=0,
                    help="use only the first N tasks of each site (pilot runs)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    grouped = tasks_by_site(HERE / args.config_dir)
    sites = args.sites or list(grouped)
    missing = [s for s in sites if s not in grouped]
    if missing:
        raise SystemExit(f"unknown site(s): {missing}. available: {list(grouped)}")

    if args.limit_tasks:
        grouped = {s_: ids[:args.limit_tasks] for s_, ids in grouped.items()}
    streams = [(site, arm, grouped[site]) for site in sites for arm in args.arms]
    total_eps = sum(len(ids) for _, _, ids in streams) * args.attempts

    print(f"{len(streams)} streams, {total_eps} episodes "
          f"({args.attempts} attempts x {args.max_steps} max steps)")
    for site, arm, ids in streams:
        print(f"  {site:<16} arm {arm}  {len(ids):>3} tasks -> {len(ids) * args.attempts:>4} episodes")
    if args.dry_run:
        return

    procs = []
    for site, arm, ids in streams:
        p = subprocess.Popen(
            [sys.executable, __file__, "--_single", site, arm,
             "--arms", arm, "--attempts", str(args.attempts),
             "--max-steps", str(args.max_steps), "--config-dir", args.config_dir,
             "--results-root", args.results_root, "--output-root", args.output_root,
             "--log-root", args.log_root, "--task-timeout", str(args.task_timeout),
             "--limit-tasks", str(args.limit_tasks)]
            + (["--resume"] if args.resume else []) + (["--sim-log"] if args.sim_log else []),
            cwd=str(HERE))
        procs.append((f"{site}_{arm}", p))
        print(f"  launched {site}_{arm} (pid {p.pid})")

    failed = []
    for tag, p in procs:
        if p.wait() != 0:
            failed.append(tag)
    print(f"\nall streams finished. failed: {failed or 'none'}")


if __name__ == "__main__":
    # A child invocation runs exactly one stream in-process.
    if "--_single" in sys.argv:
        i = sys.argv.index("--_single")
        site_arg, arm_arg = sys.argv[i + 1], sys.argv[i + 2]
        del sys.argv[i:i + 3]
        ap = argparse.ArgumentParser()
        ap.add_argument("--arms", nargs="*", default=["a"])
        ap.add_argument("--attempts", type=int, default=5)
        ap.add_argument("--max-steps", type=int, default=30)
        ap.add_argument("--config-dir", default="config_files_lite")
        ap.add_argument("--results-root", default="results/streams")
        ap.add_argument("--output-root", default="output")
        ap.add_argument("--log-root", default="logs/streams")
        ap.add_argument("--task-timeout", type=int, default=28800)
        ap.add_argument("--limit-tasks", type=int, default=0)
        ap.add_argument("--sim-log", action="store_true")
        ap.add_argument("--resume", action="store_true")
        child_args = ap.parse_args()
        ids = tasks_by_site(HERE / child_args.config_dir)[site_arg]
        if child_args.limit_tasks:
            ids = ids[:child_args.limit_tasks]
        run_stream(site_arg, arm_arg, ids, child_args)
    else:
        main()
