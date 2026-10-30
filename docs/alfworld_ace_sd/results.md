# ALFWorld: ACE + self-distillation (β sweep)

Completed 2026-10-23. 5 β values × 370 trials = 1,850 trials, no restarts or crashes. Code: [`experiments/alfworld_ace_sd/`](../../experiments/alfworld_ace_sd/). Runs: `experiments/alfworld_ace_sd/runs/20261022/beta_<β>/<room>/results/`.

## Setup

Every role uses the same frozen Qwen3-32B-AWQ. No judge score, reward or success signal is given to any role.

- **Generator**: the memory-free JitRL prompt proposes 3 candidate actions (T = 0.7). The base policy p and the teacher q score the same candidates, in the same order, with 1-token label logprobs. The teacher prompt is the same prompt plus the full playbook block. The executed action is argmax log p + β(log q − log p). When the playbook is empty, q = p.
- **Reflector / Curator**: the no-ground-truth Reflector and Curator prompts of ACE AppWorld (commit 928e868), rewritten for ALFWorld actions and observations. Field names and structure are kept; the examples are format-only placeholders. Both roles run at T = 0, once at the end of every trial. The Curator only ADDs bullets; IDs are assigned by ACE's `apply_curator_operations` (`shr` / `vc` / `misc`).
- **Protocol**: ALFWorld valid_seen, 74 tasks (kitchen 20 / living room 14 / bathroom 20 / bedroom 20), original order, 5 consecutive trials per task, seed = 42 + index·101 + trial, 30 steps. The playbook starts from three empty sections per room and accumulates across trials and tasks of that room.
- Caveat: no reward is given, but the Reflector can infer success from an early end of the trajectory and the last observation. This is the same condition as the judge-only JitRL baseline.

## Results

| Arm | Avg (room mean) | Final | Kitchen | Living room | Bathroom | Bedroom | trial 1→5 |
|---|---:|---:|---:|---:|---:|---:|---|
| ACE-SD β = 0.25 | 72.75 | 73.93 | 37.0 | 80.0 | 89.0 | 85.0 | 74 / 68 / 76 / 70 / 73 |
| ACE-SD β = 0.5 | 73.32 | 76.61 | 54.0 | 74.3 | 84.0 | 81.0 | 68 / 73 / 76 / 73 / 77 |
| ACE-SD β = 0.75 | 71.86 | 68.57 | 51.0 | 71.4 | 87.0 | 78.0 | 76 / 74 / 70 / 70 / 69 |
| **ACE-SD β = 1.0** | **75.39** | **80.89** | 68.0 | 78.6 | 83.0 | 72.0 | 73 / 73 / 72 / 77 / 81 |
| ACE-SD β = 1.5 | 73.11 | 73.39 | 66.0 | 71.4 | 85.0 | 70.0 | 73 / 77 / 72 / 72 / 73 |
| JitRL (initial port, prob update) * | 48.89 | 48.21 | 24.0 | 48.6 | 64.0 | 59.0 | 54 / 36 / 50 / 55 / 49 |
| JitRL (Eq. 25 normalization + logit) * | 65.68 | 62.86 | 43.0 | 65.7 | 78.0 | 76.0 | 69 / 66 / 69 / 62 / 62 |
| JitRL, memory off | 69.46 | 73.39 | 40.0 | 72.9 | 85.0 | 80.0 | 68 / 64 / 73 / 69 / 73 |
| ReAct (2-shot) | 78.00 | 81.61 | 81.0 | 70.0 | 77.0 | 84.0 | 80 / 74 / 78 / 78 / 82 |

\* Reported for reference; these two runs are not included in this repository.

Per room, Average / Final (each cell is "Average / Final", Final = 5th trial). ACE-SD has no training environment, so there is one row per room.

| Test env | Tasks | β = 0.25 | β = 0.5 | β = 0.75 | β = 1.0 | β = 1.5 | JitRL * | JitRL Eq. 25 + logit * | Memory off | ReAct (2-shot) |
|---|---:|---|---|---|---|---|---|---|---|---|
| Kitchen | 20 | 37.00 / 30.00 | 54.00 / 65.00 | 51.00 / 55.00 | **68.00 / 80.00** | 66.00 / 70.00 | 24.00 / 25.00 | 43.00 / 25.00 | 40.00 / 35.00 | 81.00 / 90.00 |
| Living room | 14 | 80.00 / 85.71 | 74.29 / 71.43 | 71.43 / 64.29 | 78.57 / 78.57 | 71.43 / 78.57 | 48.57 / 42.86 | 65.71 / 71.43 | 72.86 / 78.57 | 70.00 / 71.43 |
| Bathroom | 20 | 89.00 / 90.00 | 84.00 / 90.00 | 87.00 / 80.00 | 83.00 / 80.00 | 85.00 / 80.00 | 64.00 / 65.00 | 78.00 / 85.00 | 85.00 / 95.00 | 77.00 / 75.00 |
| Bedroom | 20 | 85.00 / 90.00 | 81.00 / 80.00 | 78.00 / 75.00 | 72.00 / 85.00 | 70.00 / 65.00 | 59.00 / 60.00 | 76.00 / 70.00 | 80.00 / 85.00 | 84.00 / 90.00 |
| Mean | - | 72.75 / 73.93 | 73.32 / 76.61 | 71.86 / 68.57 | **75.39 / 80.89** | 73.11 / 73.39 | 48.89 / 48.21 | 65.68 / 62.86 | 69.46 / 73.39 | 78.00 / 81.61 |

| β | Steps where the teacher changed the choice | vs memory off (task-paired) | Final bullets (K / L / Ba / Be) |
|---|---:|---:|---|
| 0.25 | 4.4% | +3.0 ± 2.5 pp | 84 / 41 / 63 / 56 |
| 0.5 | 10.4% | +4.1 ± 2.9 pp | 81 / 55 / 61 / 69 |
| 0.75 | 14.7% | +2.7 ± 3.0 pp | 82 / 39 / 103 / 64 |
| 1.0 | 25.6% | +5.9 ± 3.7 pp | 73 / 36 / 52 / 73 |
| 1.5 | 33.5% | +4.1 ± 3.7 pp | 74 / 38 / 41 / 62 |

The label-logprob floor was never used, and the Curator produced no JSON errors.

## Interpretation

- **Every β beats the JitRL variants.** ACE-SD is 6–10 pp above the best JitRL variant (Eq. 25 + logit, 65.68) and 3–6 pp above memory off. The gain over memory off is about the size of its standard error (2.5–3.7), so no single β is clearly significant on its own.
- **Kitchen improves clearly with β**: 37 → 54 → 51 → 68 → 66, against 40 for memory off. Kitchen was the weakest room for the earlier methods, and it improves as the teacher gets more weight, which suggests that the playbook really helps. Bedroom goes the other way (85 → 70).
- **Success rises over trials** for β = 1.0 (73 → 81) and β = 0.5 (68 → 77), a pattern none of the JitRL variants showed.
- **ACE-SD is still below 2-shot ReAct (78.00).** The Final of β = 1.0 (80.89) is close to the ReAct Final (81.61).

## Caveat: a confound in the comparison

Even at small β, ACE-SD selects actions differently from memory-off JitRL. Memory off executes the `best_action` field of the proposal JSON (sampled at T = 0.7). ACE-SD **rescores** the same candidates and takes the argmax (for β → 0, the argmax of the base scores). At β = 0.25 the teacher changes only 4.4% of the choices, yet the gain is +3.0 pp, so part of the gain may come from the **rescoring argmax itself** rather than the playbook. Separating the two needs a **β = 0 control**. Trial 1 of the iterative self-distillation run is exactly that control ([iterative.md](iterative.md)): it matches memory off (−0.2 ± 5.3 pp), so rescoring alone does not explain the gain.
