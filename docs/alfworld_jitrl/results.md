# ALFWorld: memory-on JitRL with judge-only rewards

Run 2026-10-22, 370/370 trials. Code: [`experiments/alfworld_jitrl/`](../../experiments/alfworld_jitrl/). Run: `experiments/alfworld_jitrl/runs/20261022_eq25_logit/` (`plan.json` and per-trial results; it was produced by an earlier supervisor with the same agent, judge and protocol, so its JSON fields differ slightly from those `run.py` writes).

## Setup

- **Protocol**: valid_seen 74 tasks × 5 trials, original order, seed = 42 + index·101 + trial, Qwen3-32B-AWQ actor at T = 0.7, 30 steps. Memory is fresh and empty per room and persists across the tasks and trials of that room, so each room is one sequential stream.
- **Rewards (judge-only)**: after each trial, gpt-oss-20b (T = 0.3, reasoning effort medium) scores every step on the seven-level rubric of `ALFWorld/src/utils.py` (−10, −6, −3, 0, +3, +6, +10). The trajectory it sees has no environment reward and no success line. Scores × 0.1 are the step rewards; γ = 0.95, no success bonus or failure penalty. The environment outcome never reaches the actor, the state tracker, the judge or the memory.
- **Retrieval**: the WebArena upstream rule (same location, task 1-gram Jaccard > 0.27, similarity 0.7 · history + 0.3 · task, k = 3).
- **Advantage**: Eq. 25 of the JitRL paper, A / (max over the candidate set of |A| + ε), instead of the code's division by the largest positive advantage.
- **Update**: score = log p + w · Ã (the paper's logit update, Eq. 10), with the code's episode weight w = min(1 + ep/50 · 0.5, 1.5); the paper gives no value for β.

## Memory-on JitRL variants

Only the last row's run is included in this repository; the others are listed for reference.

| Variant | Avg | Final | Kitchen | Living room | Bathroom | Bedroom | Choices changed by memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| Initial port (code normalization, p + w·A) | 48.89 | 48.21 | 24.0 | 48.6 | 64.0 | 59.0 | 61% |
| + upstream retrieval k = 3 | 52.43 | 49.46 | 27.0 | 55.7 | 57.0 | 70.0 | 53% |
| + Eq. 25 normalization, p + w·Ã | 55.00 | 65.36 | 35.0 | 60.0 | 59.0 | 66.0 | 49% |
| **+ Eq. 25 normalization, log p + w·Ã** | **65.68** | 62.86 | 43.0 | 65.7 | 78.0 | 76.0 | 34% |
| JitRL, memory off | 69.46 | 73.39 | 40.0 | 72.9 | 85.0 | 80.0 | 0% |

Per room the cells are Average success over all 5 trials. Task-paired differences (± SE):
- Eq. 25 + logit vs upstream retrieval: +13.5 ± 2.7 pp.
- Eq. 25 + logit vs memory off: −3.5 ± 2.9 pp.

## Interpretation

- Of the two places where the code differs from the paper, the **combination space** matters: adding the advantage to log p instead of p is worth +13.5 pp, and changing only the denominator is not distinguishable from noise (+2.4 ± 3.1 pp).
- In log space, actions that memory injects but the actor never proposed have p = 0 → log(1e-6) ≈ −13.8 and are never executed (14% of executed actions with the upstream-retrieval variant, 0% here). Part of the gain may come from this blocking and part from the smaller effect of the advantage in log space; this run cannot separate them.
- Even the best variant does **not** improve over trials (69 / 66 / 69 / 62 / 62%) and stays slightly below memory off. Likely causes are the quality of the judge-only rewards (failed episodes still get positive returns) and the unspecified β.

## Running it

```bash
scripts/serve_vllm.sh jitrl          # 2 × ≥ 32 GB; or `jitrl-tp` on 2 × 24 GB
RUN_DIR=runs_new/jitrl python experiments/alfworld_jitrl/launch.py
```

Expect `[Judge] attempt 1 incomplete` lines now and then: when the first 6000-token budget truncates the response, the judge is asked again with a larger budget. No score is ever filled in. A stream that dies mid-trial restores the memory snapshot taken before that trial and reruns it from the same seed.
