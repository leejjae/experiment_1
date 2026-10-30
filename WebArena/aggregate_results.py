#!/usr/bin/env python
"""Aggregate WebArena-Lite runs into the paper's two headline metrics.

Section 5.2.1 of the JitRL paper reports, over L=5 consecutive trials per task:

    Average Success Rate = (1 / (T * L)) * sum_i sum_j  y_ij
      -- averaged over every attempt; reflects overall learning efficiency.

    Final Success Rate   = (1 / T) * sum_i y_iL
      -- the last attempt only; reflects converged capability.

Final is the reason early stopping must stay off: a task that succeeds on attempt 2
and then skips 3-5 has no y_i5 at all, and the metric is simply not computable for it.
This script refuses to report Final over incomplete tasks rather than silently
substituting the last attempt that happened to run.

Input: the per-task JSON that run.py writes,
    <result_dir>/cumulative_results_<task>_x<L>.json
with detailed_results = [{run: 1..L, task_name, success}, ...].

Usage:
    python aggregate_results.py results/arm_a --config-dir config_files_lite
    python aggregate_results.py results/arm_a results/arm_b --labels arm_a arm_b
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict


def wilson_ci(successes, total, z=1.96):
    """Wilson score interval -- behaves sanely at 0% and 100%, unlike the normal approx."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_task_site_map(config_dir):
    """task_id -> primary site, from the generated lite configs."""
    site_of = {}
    for path in glob.glob(os.path.join(config_dir, "*.json")):
        name = os.path.basename(path)
        if not re.fullmatch(r"\d+\.json", name):
            continue
        with open(path) as f:
            conf = json.load(f)
        sites = conf.get("sites") or []
        site_of[str(conf["task_id"])] = sites[0] if sites else "unknown"
    return site_of


def load_runs(result_dir):
    """task_id -> {attempt index: bool}, from run.py's per-task summaries."""
    per_task = {}
    pattern = os.path.join(result_dir, "**", "cumulative_results_*.json")
    for path in glob.glob(pattern, recursive=True):
        with open(path) as f:
            data = json.load(f)
        task_name = data.get("task_name", "")
        task_id = task_name.rsplit(".", 1)[-1]
        attempts = {}
        for row in data.get("detailed_results", []):
            # An errored attempt is recorded with success False; that is a failed attempt,
            # not a missing one, so it counts in both metrics.
            attempts[int(row["run"])] = bool(row.get("success"))
        if attempts:
            per_task[task_id] = attempts
    return per_task


def summarize(per_task, site_of, n_attempts):
    rows = defaultdict(lambda: {"avg_succ": 0, "avg_tot": 0, "fin_succ": 0, "fin_tot": 0,
                                "incomplete": [], "by_attempt": defaultdict(lambda: [0, 0])})
    for task_id, attempts in sorted(per_task.items(), key=lambda kv: int(kv[0])):
        site = site_of.get(task_id, "unknown")
        for bucket in (rows[site], rows["ALL"]):
            for j, ok in attempts.items():
                bucket["avg_tot"] += 1
                bucket["avg_succ"] += int(ok)
                cell = bucket["by_attempt"][j]
                cell[1] += 1
                cell[0] += int(ok)
            if n_attempts in attempts:
                bucket["fin_tot"] += 1
                bucket["fin_succ"] += int(attempts[n_attempts])
            else:
                bucket["incomplete"].append(task_id)
    return rows


def print_report(label, rows, n_attempts):
    print(f"\n{'=' * 78}\n{label}   (L={n_attempts})\n{'=' * 78}")
    header = f"{'site':<16} {'n':>4} {'Average SR':>22} {'Final SR':>22}"
    print(header)
    print("-" * len(header))
    for site in sorted(rows, key=lambda s: (s == "ALL", s)):
        r = rows[site]
        n_tasks = r["fin_tot"] + len(r["incomplete"])
        a_lo, a_hi = wilson_ci(r["avg_succ"], r["avg_tot"])
        avg = f"{r['avg_succ'] / r['avg_tot']:.1%} [{a_lo:.2f},{a_hi:.2f}]" if r["avg_tot"] else "n/a"
        if r["incomplete"]:
            fin = f"n/a ({len(r['incomplete'])} incomplete)"
        else:
            f_lo, f_hi = wilson_ci(r["fin_succ"], r["fin_tot"])
            fin = f"{r['fin_succ'] / r['fin_tot']:.1%} [{f_lo:.2f},{f_hi:.2f}]" if r["fin_tot"] else "n/a"
        print(f"{site:<16} {n_tasks:>4} {avg:>22} {fin:>22}")

    # Per-attempt curve: this is where continual learning shows up (or does not).
    all_rows = rows["ALL"]
    print(f"\n{'attempt':<10} {'success rate':>14}   (learning curve)")
    print("-" * 44)
    for j in range(1, n_attempts + 1):
        succ, tot = all_rows["by_attempt"].get(j, [0, 0])
        if tot:
            bar = "#" * int(30 * succ / tot)
            print(f"  j={j:<7} {succ / tot:>13.1%}   {bar}")
        else:
            print(f"  j={j:<7} {'no data':>13}")

    incomplete = rows["ALL"]["incomplete"]
    if incomplete:
        print(f"\nWARNING: {len(incomplete)} task(s) never ran attempt {n_attempts}, so Final SR "
              f"is not computable for them. Re-run with early stopping disabled "
              f"(test_webarena_lite.py --no_early_stop).")
        print(f"  task ids: {', '.join(incomplete[:20])}" + (" ..." if len(incomplete) > 20 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dirs", nargs="+", help="one directory per arm")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--config-dir", default="config_files_lite")
    ap.add_argument("--attempts", type=int, default=5, help="L, the number of trials per task")
    ap.add_argument("--csv", default=None, help="also write one row per (arm, task, attempt)")
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(d.rstrip("/")) for d in args.result_dirs]
    if len(labels) != len(args.result_dirs):
        raise SystemExit("--labels must have one entry per result dir")

    site_of = load_task_site_map(args.config_dir)
    if not site_of:
        print(f"[WARN] no task configs found in {args.config_dir}; every task will be 'unknown'")

    csv_rows = []
    for label, result_dir in zip(labels, args.result_dirs):
        per_task = load_runs(result_dir)
        if not per_task:
            print(f"\n[{label}] no cumulative_results_*.json under {result_dir} -- nothing to aggregate")
            continue
        print_report(label, summarize(per_task, site_of, args.attempts), args.attempts)
        for task_id, attempts in per_task.items():
            for j, ok in sorted(attempts.items()):
                csv_rows.append((label, site_of.get(task_id, "unknown"), task_id, j, int(ok)))

    if args.csv and csv_rows:
        with open(args.csv, "w") as f:
            f.write("arm,site,task_id,attempt,success\n")
            for row in csv_rows:
                f.write(",".join(str(x) for x in row) + "\n")
        print(f"\nwrote {len(csv_rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
