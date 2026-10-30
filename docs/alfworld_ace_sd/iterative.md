# ALFWorld: iterative self-distillation (π_{k+1} ∝ π_k (q_k/π_1)^β)

Code: [`agent_iter.py`](../../experiments/alfworld_ace_sd/agent_iter.py), [`run_iter.py`](../../experiments/alfworld_ace_sd/run_iter.py), [`launch_iter.py`](../../experiments/alfworld_ace_sd/launch_iter.py), [`launch_fixedfb.py`](../../experiments/alfworld_ace_sd/launch_fixedfb.py). Runs: `experiments/alfworld_ace_sd/runs/20261029_iter/beta_<β>/` and `experiments/alfworld_ace_sd/runs/20261029_fixedfb/<rule>/`. Every run has 370 trials with no crashes; the β = 1 run took 1 h 38 min on 2 × RTX 5090.

## Setup

- π_1 is the base policy. After trial k, the Reflector writes feedback f_k from that trial's trajectory alone. Its playbook field is `(empty)`, and no Curator is used.
- q_j(a|s) = π_1(a | s, f_j): the same prompt with f_j added, scoring the same candidates in the same order with 1-token label logprobs.
- At trial k the agent executes the argmax of `log π_k = log π_1 + β Σ_{j<k} (log q_j − log π_1)` (the **sum** rule).
- Feedback is **reset per task**: every task starts from π_1 and uses only its own f_1..f_{k−1}. Trial 1 is therefore the argmax of the base rescoring, which is the **β = 0 control** asked for in the [ACE-SD results](results.md).
- Proposal (JitRL prompt, 3 candidates, T = 0.7), protocol (valid_seen 74 tasks × 5, seed = 42 + index·101 + trial, 30 steps) and the no-reward condition are the same as in ACE-SD.

## Results, β = 1

Per room, Average / Final (Final = 5th trial).

| Test env | Tasks | Iterative SD (β = 1) | ACE-SD β = 1 | Memory off | ReAct (0-shot) |
|---|---:|---|---|---|---|
| Kitchen | 20 | 64.00 / 70.00 | 68.00 / 80.00 | 40.00 / 35.00 | 44.00 / 50.00 |
| Living room | 14 | 75.71 / 78.57 | 78.57 / 78.57 | 72.86 / 78.57 | 67.14 / 64.29 |
| Bathroom | 20 | 86.00 / 85.00 | 83.00 / 80.00 | 85.00 / 95.00 | 50.00 / 60.00 |
| Bedroom | 20 | 80.00 / 85.00 | 72.00 / 85.00 | 80.00 / 85.00 | 70.00 / 55.00 |
| Mean | - | **76.43 / 79.64** | 75.39 / 80.89 | 69.46 / 73.39 | 57.79 / 57.32 |

Room-mean success rate per trial, and the share of steps where the teacher changed the base choice:

| Trial | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Iterative SD | 66.6 | **80.9** | 77.1 | 77.9 | 79.6 |
| ACE-SD β = 1 | 73.4 | 73.4 | 71.6 | 77.7 | 80.9 |
| Memory off | 68 | 64 | 73 | 69 | 73 |
| ReAct (0-shot) | 52.3 | 58.4 | 63.6 | 57.3 | 57.3 |
| Iterative SD: choice changed | 0% | 27.7% | 45.6% | 52.7% | 54.1% |
| Iterative SD: mean steps | 17.8 | 13.9 | 14.9 | 15.1 | 14.2 |

Kitchen success per trial: 30 → 75 → 80 → 65 → 70.

Task-paired differences (room mean, ± SE):
- vs memory off: −0.2 ± 5.3 pp at trial 1, +8.8 ± 3.7 pp over trials 2–5, +7.0 ± 3.5 pp overall.
- vs ACE-SD β = 1: +1.0 ± 3.0 pp overall.

## Interpretation

- **The rescoring argmax alone has no effect.** Trial 1 (the β = 0 control) equals memory off at the same seeds (−0.2 ± 5.3 pp). ACE-SD's gain over memory off therefore comes from the feedback, not from rescoring.
- **Most of the gain comes from the first piece of feedback.** Success goes from 66.6 to 80.9 between trials 1 and 2 (Kitchen 30 → 75) and then stays at 77–80. The trial 2–5 mean (78.9) is 21 pp above 0-shot ReAct (57.79).
- **Multiplying in more feedback does not help.** At β = 1 the trial-k policy is q_1⋯q_{k−1}/π_1^{k−2}. As teachers accumulate, the share of changed choices grows from 28% to 54% while success stays flat. Between consecutive trials there are 28 failure→success and 18 success→failure transitions, so earlier feedback also changes behaviour on tasks that were already solved.
- **Overall it is on par with ACE-SD β = 1** (+1.0 ± 3.0 pp), but the learning curves differ. ACE-SD improves gradually as the room-level playbook grows (73 → 81), while iterative SD jumps once from same-task feedback. Iterative SD uses information about re-solving the same task, and ACE-SD uses information that transfers across tasks, so the trial-2 jump of iterative SD does not indicate generalization.
- By task type (trial 1 → mean of trials 2–5), the object-processing tasks gain most: clean 50 → 85, cool 40 → 70, heat 40 → 75. The feedback directly fixes failures of the form "skipped the processing step".

## Why does sum plateau? Over-accumulation (offline, β = 1 traces)

Using the stored per-step log p and log q_j, we recomputed what other combination rules would have chosen among the same candidates. Script: [`analysis/overacc_offline.py`](../../analysis/overacc_offline.py).

- **There is decision-level over-accumulation.** At β = 1 the score is Σ log q_j − (k−2) log p, so from k ≥ 3 on, actions the base dislikes get a bonus.
  - Steps choosing an action that no teacher ranks first: 17.4% at k = 3, 21.5% at k = 4, 25.4% at k = 5.
  - About 70% of those are the base's lowest-ranked action, and the teachers give them a mean probability of only 0.07–0.18.
  - Teachers agree with each other on the argmax 76–82% of the time.
- **It does not spoil tasks that were already solved.** The success→failure rate between consecutive trials is 8.0%, half that of memory off (16.8%) and 0-shot ReAct (16.6%), which get no feedback.
- **The plateau comes from tasks that keep failing, and over-accumulation is concentrated there.**
  - 7 tasks are never solved in trials 2–5; 6 of them are pick_two, and all 7 also failed at trial 1.
  - Memory off solves the same tasks 43% of the time (0-shot ReAct 9%), so they are not impossible.
  - On these 7 tasks, 36–43% of the choices are actions no teacher ranks first, twice the rate on the other tasks (13–21%). The section on direct evidence below shows that 76% of such choices actually happen at steps where the teachers are **unanimous**, and 97.5% are caused by the −(k−2) log p term. They are not caused by teacher disagreement.
- **Limitation**: offline analysis alone cannot settle the causal direction. The β = 0.5 run and the fixed-feedback rule comparison below test it.

## β = 0.5

Same code, tasks and seeds with β = 0.5. Run: `runs/20261029_iter/beta_0.5/`.

| Test env | Iterative SD β = 0.5 | Iterative SD β = 1 | ACE-SD β = 1 | Memory off | ReAct (0-shot) |
|---|---|---|---|---|---|
| Kitchen | 66.00 / 80.00 | 64.00 / 70.00 | 68.00 / 80.00 | 40.00 / 35.00 | 44.00 / 50.00 |
| Living room | 82.86 / 85.71 | 75.71 / 78.57 | 78.57 / 78.57 | 72.86 / 78.57 | 67.14 / 64.29 |
| Bathroom | 86.00 / 90.00 | 86.00 / 85.00 | 83.00 / 80.00 | 85.00 / 95.00 | 50.00 / 60.00 |
| Bedroom | 90.00 / 85.00 | 80.00 / 85.00 | 72.00 / 85.00 | 80.00 / 85.00 | 70.00 / 55.00 |
| Mean | **81.21 / 85.18** | 76.43 / 79.64 | 75.39 / 80.89 | 69.46 / 73.39 | 57.79 / 57.32 |

| Trial | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| β = 0.5 SR | 70.4 | 83.9 | 82.7 | 83.9 | 85.2 |
| β = 1 SR | 66.6 | 80.9 | 77.1 | 77.9 | 79.6 |
| β = 0.5: choice no teacher ranks first | - | (11.7%)* | 0.3% | 2.8% | 8.4% |
| β = 1: choice no teacher ranks first | - | 0.1% | 17.4% | 21.5% | 25.4% |

\* At k = 2, β = 0.5 interpolates between p and q. This 11.7% is the base's top choice surviving, not a choice the base dislikes.

- **β = 0.5 does not plateau**; it keeps rising to 85.2 after trial 2. The over-accumulation indicator appears later and smaller, as predicted: the coefficient on log p is 1 − 0.5(k−1), which is 0 at k = 3 (equal to the teacher mean) and −1 at k = 5.
- The trial 2–5 success→failure rate is also 8.0%. Tasks never solved drop from 7 to 3 (5 pick_two tasks become solvable).
- Task-paired difference (β = 0.5 − β = 1): +5.0 ± 3.0 pp over trials 2–5, +4.8 ± 2.9 pp overall. However, trial 1 differs by +3.8 ± 5.3 pp even though both runs use the same policy there, so much of the difference may be noise. The result **points in the direction** of the over-accumulation hypothesis but is not decisive on its own.

## Combination rules with the same feedback (β = 1)

We reused the f_k stored by the β = 1 sum run (the Reflector is not called again) and changed only the combination rule. With r_j = log q_j − log π_1, the score is log π_1 + β·R:

- sum: R = Σ r_j (the original β = 1 run)
- replace: R = r_{k−1}
- mean: R = mean of r_j
- EMA: R_1 = r_1, R_j = 0.5·R_{j−1} + 0.5·r_j

Code: `combine` in `agent_iter.py`, `run_iter.py --rule --feedback_from`, `launch_fixedfb.py`.

| Test env | sum | replace | mean | EMA (λ = 0.5) | Memory off | ReAct (0-shot) |
|---|---|---|---|---|---|---|
| Kitchen | 64.00 / 70.00 | 68.00 / 75.00 | 66.00 / 65.00 | 65.00 / 80.00 | 40.00 / 35.00 | 44.00 / 50.00 |
| Living room | 75.71 / 78.57 | 78.57 / 78.57 | 87.14 / 100.00 | 84.29 / 92.86 | 72.86 / 78.57 | 67.14 / 64.29 |
| Bathroom | 86.00 / 85.00 | 86.00 / 80.00 | 91.00 / 95.00 | 88.00 / 90.00 | 85.00 / 95.00 | 50.00 / 60.00 |
| Bedroom | 80.00 / 85.00 | 86.00 / 95.00 | 91.00 / 100.00 | 89.00 / 95.00 | 80.00 / 85.00 | 70.00 / 55.00 |
| Mean | 76.43 / 79.64 | 79.64 / 82.14 | **83.79 / 90.00** | 81.57 / 89.46 | 69.46 / 73.39 | 57.79 / 57.32 |

| Rule | SR trial 1 → 5 | SR trials 3–5 | vs sum, trials 3–5 (paired) | Choice no teacher ranks first (k = 3/4/5) | Success→failure | Tasks failed in all of trials 2–5 |
|---|---|---:|---:|---|---:|---:|
| sum | 66.6 / 80.9 / 77.1 / 77.9 / 79.6 | 78.2 | - | 17.4 / 21.5 / 25.4% | 8.0% | 7 |
| replace | 68.6 / 80.2 / 84.6 / 82.7 / 82.1 | 83.2 | +4.9 ± 3.0 | 0.0 / 0.1 / 0.0% | 9.8% | 3 |
| mean | 72.1 / 82.1 / 88.2 / 86.4 / 90.0 | **88.2** | **+10.0 ± 3.2** | 1.1 / 0.4 / 0.2% | 8.2% | 2 |
| EMA | 65.4 / 79.1 / 85.2 / 88.8 / 89.5 | 87.8 | +9.6 ± 3.4 | 0.4 / 0.3 / 0.1% | 7.7% | 2 |

- Noise reference: in trials 1–2 all four rules define the same policy, yet they differ from sum by −1.5 to +3.4 pp (SE 2.5–3.5).
- **The sum plateau is caused by the combination rule.** With identical feedback text, mean and EMA are about 10 pp above sum in trials 3–5 (SE ≈ 3.3, outside the noise range) and keep improving up to trial 5 (90.0 / 89.5). Choices that no teacher ranks first essentially disappear. The feedback was generated from sum's own trajectories, so the comparison, if anything, favours sum.
- **The latest feedback alone is not enough.** replace beats sum but is 5.1 ± 2.4 pp below mean, so averaging over trials helps. mean and EMA are indistinguishable (−0.4 ± 2.5 pp).
- Kitchen barely changes across rules (64–68); the gains come from Living room, Bathroom and Bedroom.

## Direct evidence of over-accumulation (offline)

Computed from the stored per-step log p and log q_j. Script: [`analysis/overacc_evidence.py`](../../analysis/overacc_evidence.py). The sum score is `(1 − β(k−1))·log p + β·Σ log q_j`; below, c = 1 − β(k−1) denotes the coefficient on log p.

**1. Unanimity violations**: among steps where the base and every teacher rank the same action first, the share where a different action is chosen.

| Run | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---:|---:|---:|---:|
| sum β = 1 | 0.1% (c = 0) | 19.2% (c = −1) | 27.9% (c = −2) | 32.3% (c = −3) |
| sum β = 0.5 | 0.0% (c = +0.5) | 0.0% (c = 0) | 2.4% (c = −0.5) | 9.7% (c = −1) |
| mean / EMA / replace (β = 1) | 0.0% | 0.0% | 0.0% | 0.0–0.2% |

- Violations appear only once c turns negative, and grow as c becomes more negative.
- The mean / EMA / replace scores are convex combinations of the log q_j, so a unanimous teacher choice is always selected; the measured rate is indeed 0.
- With c < 0, actions with low log p, i.e. actions the base dislikes, receive a bonus that can overturn the agreement of all teachers.

**2. Attribution** (sum β = 1, k ≥ 3, 3,228 steps)
- 689 steps (21.3%) choose an action that no teacher ranks first.
- 97.5% of them would not be chosen by the teacher product alone (Σ log q_j, without the base term): the −(k−2) log p term makes the choice.
- 75.6% of them are unanimity violations as defined above, and 71.8% are the base's lowest-ranked action. The median log p of the chosen action is −6.44, against −0.21 for normal choices.
- Example: kitchen task 58, trial 4, step 3. The base and all three teachers rank `go to drawer 1` first with probability 0.86–0.98. Sum picks `go to cabinet 17` (log p = −6.19) because of the −2·log p = +12.4 bonus.

**3. These choices are bad actions.** 43.3% of them repeat an earlier action, against 23.6% for normal choices, so revisits and loops increase. All candidates are valid actions, so "Nothing happens" is 0 in both groups. The first deviating choice comes early (median step 3).

**4. Link to trial outcomes.** For the same task, trial and feedback text, the mean − sum success difference, split by how many deviating choices sum made in the first 10 steps (trials 3–5):

| Deviating choices in first 10 steps | n | sum SR | mean SR | Difference | Trial-1 difference (same policy, noise reference) |
|---|---:|---:|---:|---:|---:|
| 0 | 110 | 93.6 | 99.1 | +5.5 | +3.7 |
| 1 | 39 | 84.6 | 87.2 | +2.6 | −2.5 |
| 2+ | 73 | 52.1 | 71.2 | **+19.2** | +12.3 |

- The sum − mean gap is concentrated in trials with many deviating choices.
- However, this group consists of harder tasks (trial-1 SR 41–53%). Subtracting the trial-1 noise difference leaves only +6.9 pp, so the trial-level link **points the right way but is confounded with difficulty**.

**Summary**
- The mechanism is confirmed directly: the −(k−2) log p term changes the decisions, overturning unanimous teacher choices and increasing repeated actions (1–3).
- The causal evidence that this mechanism produces the plateau is the intervention above: with the same feedback, changing only the rule gives mean and EMA +10 pp in trials 3–5.
- A more direct test would replay a sum trajectory up to its first unanimity violation, take the agreed action at that single step, and continue. This needs GPUs and has not been run.

## Open follow-ups

- Run mean / EMA on-policy instead of with fixed feedback, so each rule receives feedback from its own trajectories.
- Investigate why Kitchen plateaus regardless of the rule.
