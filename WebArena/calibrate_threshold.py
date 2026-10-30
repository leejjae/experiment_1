#!/usr/bin/env python
"""Read the retrieval-similarity distribution collected by JITRL_SIM_LOG and report
the percentiles a --retrieval_threshold should be picked from.

Why this exists: the threshold is not portable across environments. ALFWorld's 0.85 was
calibrated against its own state_key Jaccard distribution; WebArena retrieves on
normalized URLs + trajectory context + task goal, so the distribution is a different
shape entirely. Setting it too high makes retrieval never fire -- and JitRL then behaves
exactly like a memoryless agent, with no error to notice. Setting it too low pulls in
unrelated trajectories and the advantage on top of them becomes noise.

Usage:
    JITRL_SIM_LOG=output/simlog.jsonl python run.py ... --retrieval_threshold 0.0
    python calibrate_threshold.py output/simlog.jsonl
"""

import argparse
import json
import sys
from collections import Counter


def percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    idx = min(int(round(q / 100.0 * (len(sorted_vals) - 1))), len(sorted_vals) - 1)
    return sorted_vals[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_log", help="JSONL file written by JITRL_SIM_LOG")
    ap.add_argument("--top-k", type=int, default=10,
                    help="retrieval_top_k the run will use; the per-call top-k distribution "
                         "is what actually decides whether retrieval fires (default: 10)")
    args = ap.parse_args()

    calls = []
    with open(args.sim_log) as f:
        for line in f:
            line = line.strip()
            if line:
                calls.append(json.loads(line))

    if not calls:
        sys.exit(f"{args.sim_log} is empty -- retrieval never ran. Check that memory is "
                 f"enabled and that at least one episode has already been stored.")

    all_sims = sorted(s for c in calls for s in c["similarities"])
    # The k-th best candidate per call is the real decision boundary: a threshold above it
    # leaves that call with fewer than k trajectories.
    kth = sorted(sorted(c["similarities"])[-args.top_k] if len(c["similarities"]) >= args.top_k
                 else min(c["similarities"]) for c in calls)
    best = sorted(max(c["similarities"]) for c in calls)

    print(f"retrieval calls        : {len(calls)}")
    print(f"candidates per call    : mean {sum(len(c['similarities']) for c in calls)/len(calls):.1f}"
          f"  min {min(len(c['similarities']) for c in calls)}"
          f"  max {max(len(c['similarities']) for c in calls)}")
    print(f"total pairs            : {len(all_sims)}")
    print()

    header = f"{'percentile':>10} {'all pairs':>12} {'best/call':>12} {f'{args.top_k}th/call':>12}"
    print(header)
    print("-" * len(header))
    for q in (5, 25, 50, 75, 90, 95, 99):
        print(f"{q:>9}% {percentile(all_sims, q):>12.4f} "
              f"{percentile(best, q):>12.4f} {percentile(kth, q):>12.4f}")
    print()

    # How often would retrieval fire at each candidate threshold?
    print("threshold  calls firing  (fraction of retrieval calls with >=1 trajectory)")
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        firing = sum(1 for b in best if b >= t)
        bar = "#" * int(40 * firing / len(best))
        print(f"   {t:<5.2f}  {firing:>5}/{len(best)} {firing/len(best):>6.1%}  {bar}")
    print()

    degenerate = Counter()
    for c in calls:
        if len(set(c["similarities"])) == 1:
            degenerate["identical"] += 1
    if degenerate["identical"]:
        print(f"WARNING: {degenerate['identical']}/{len(calls)} calls had every candidate at the "
              f"same similarity. That is the ALFWorld p75=1.000 failure mode -- the key has no "
              f"discriminative power, and no threshold will fix it.")

    print("How to read this: pick a threshold where retrieval fires on most calls but is not "
          "firing on everything. If the 'all pairs' and 'best/call' columns are nearly equal, "
          "the key is not discriminating and the threshold is not the thing to tune.")


if __name__ == "__main__":
    main()
