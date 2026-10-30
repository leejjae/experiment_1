# ALFWorld: Reflexion / EvoTest / AWM (no environment reward, LLM judge)

3 methods × 370 trials, no crashes. Code: [`methods.py`](../../experiments/alfworld_baselines/methods.py), [`run_baseline.py`](../../experiments/alfworld_baselines/run_baseline.py), [`launch.py`](../../experiments/alfworld_baselines/launch.py), prompts in [`prompts/`](../../experiments/alfworld_baselines/prompts/). Runs: `experiments/alfworld_baselines/runs/20261029/<method>/<room>/results/`.

## Setup

- Same protocol as iterative self-distillation: valid_seen 74 tasks × 5 trials, original order, seed = 42 + index·101 + trial, Qwen3-32B-AWQ, actor T = 0.7, 30 steps.
- The actor is the memory-free JitRL agent (`enable_cross_mem=False`), the same as the "memory off" arm. Each method only produces the text placed in the actor's `guiding_prompt`.
- **No environment success signal reaches any role.** Where a method needs success, the same Qwen3-32B judges the trajectory from the logprobs of a single YES/NO token (success if p > 0.5).

| Method | Memory scope | Behaviour |
|---|---|---|
| Reflexion | Per task | After a trial judged failed, writes a new plan with the official ALFWorld reflection prompt and its 2 few-shot examples. The last 3 plans are the actor's memory. |
| EvoTest | Per task | A tree of guiding prompts. Trial 1 uses the default prompt; later trials pick a parent by UCB (c = 1.0, d = 0.8), and the Evolver (T = 0.7) rewrites it. Once a node is judged successful, the best node is frozen (freeze_on_win). |
| AWM (online) | Across tasks within a room | Induces workflows from trajectories judged successful and updates the room's workflow set. |

From EvoTest we omitted the `extract_state` code generation and the score-delta memory: ALFWorld has no intermediate score, and the actor prompt already contains a state summary.

## Results

Per room, Average / Final (Final = 5th trial).

| Test env | Tasks | Reflexion | EvoTest | AWM | Iterative SD mean (β = 1) | Iterative SD EMA (β = 1) | Iterative SD sum β = 0.5 | Memory off | ReAct (0-shot) |
|---|---:|---|---|---|---|---|---|---|---|
| Kitchen | 20 | 60.00 / 70.00 | 56.00 / 65.00 | 43.00 / 55.00 | 66.00 / 65.00 | 65.00 / 80.00 | 66.00 / 80.00 | 40.00 / 35.00 | 44.00 / 50.00 |
| Living room | 14 | 70.00 / 78.57 | 75.71 / 71.43 | 61.43 / 57.14 | 87.14 / 100.00 | 84.29 / 92.86 | 82.86 / 85.71 | 72.86 / 78.57 | 67.14 / 64.29 |
| Bathroom | 20 | 83.00 / 80.00 | 89.00 / 95.00 | 85.00 / 80.00 | 91.00 / 95.00 | 88.00 / 90.00 | 86.00 / 90.00 | 85.00 / 95.00 | 50.00 / 60.00 |
| Bedroom | 20 | 84.00 / 90.00 | 89.00 / 95.00 | 78.00 / 85.00 | 91.00 / 100.00 | 89.00 / 95.00 | 90.00 / 85.00 | 80.00 / 85.00 | 70.00 / 55.00 |
| Mean | - | 74.25 / 79.64 | 77.43 / 81.61 | 66.86 / 69.29 | **83.79 / 90.00** | 81.57 / 89.46 | 81.21 / 85.18 | 69.46 / 73.39 | 57.79 / 57.32 |

Room-mean success rate per trial, and the success→failure rate between consecutive trials:

| | Trial 1 | 2 | 3 | 4 | 5 | Success→failure |
|---|---:|---:|---:|---:|---:|---:|
| Reflexion | 64.3 | 73.6 | 79.6 | 74.1 | 79.6 | 10.1% |
| EvoTest | 73.9 | 71.1 | 79.1 | 81.4 | 81.6 | 12.8% |
| AWM | 63.6 | 67.3 | 65.5 | 68.6 | 69.3 | 10.7% |
| Iterative SD mean | 72.1 | 82.1 | 88.2 | 86.4 | 90.0 | 8.2% |
| Memory off | 66.8 | 65.2 | 73.4 | 68.6 | 73.4 | 16.8% |
| ReAct (0-shot) | 52.3 | 58.4 | 63.6 | 57.3 | 57.3 | 16.6% |

Task-paired differences (all 5 trials, room mean, ± SE):

| | vs memory off | vs iterative SD mean | vs ReAct 0-shot |
|---|---:|---:|---:|
| Reflexion | +4.8 ± 2.8 | −9.5 ± 2.7 | +16.5 ± 3.7 |
| EvoTest | +8.0 ± 2.4 | −6.4 ± 2.7 | +19.6 ± 3.7 |
| AWM | −2.6 ± 2.8 | −16.9 ± 3.1 | +9.1 ± 3.8 |

For reference, iterative SD mean is +14.3 ± 2.4 over memory off.

Success rate by task type (all 5 trials, %):

| | look | simple | clean | cool | heat | pick_two |
|---|---:|---:|---:|---:|---:|---:|
| Reflexion | 73 | 94 | 70 | 60 | 60 | 59 |
| EvoTest | 84 | 95 | 72 | 44 | 64 | 65 |
| AWM | 78 | 91 | 72 | 24 | 56 | 42 |
| Iterative SD mean | 89 | 98 | 86 | 68 | 68 | 68 |
| Memory off | 78 | 92 | 60 | 36 | 56 | 52 |

## Judge quality (these runs, against the true outcome)

| | Accuracy | Precision of "success" | Recall of true success | Recall of true failure | FP / FN |
|---|---:|---:|---:|---:|---|
| Reflexion | 0.895 | 0.984 | 0.873 | 0.957 | 4 / 35 |
| EvoTest | 0.892 | 0.996 | 0.864 | 0.988 | 1 / 39 |
| AWM | 0.895 | 0.986 | 0.855 | 0.975 | 3 / 36 |

This matches the pre-run check on 150 stored trajectories (0.88). Almost all errors are false negatives (a success judged as failure). As a result, Reflexion also writes reflections for tasks it already solved, EvoTest freezes later than it could, and AWM misses some induction opportunities. False positives are rare, so failed trajectories seldom enter the workflows or get frozen.

## Method behaviour

- **Reflexion**: 125 reflections over 370 trials. The trial 2–5 mean (76.7) is above trial 1 (64.3), but the curve is uneven (it drops at trial 4).
- **EvoTest**: 74 root, 92 evolve and 204 freeze trials. Frozen trials succeed 86.3% of the time; trials right after an evolve step, 62.0%. Trial 1 is already high (73.9): the default prompt ("Explore systematically and examine objects…") alone beats memory off at trial 1 (66.8).
- **AWM**: workflows were induced 44 (kitchen), 41 (living room), 86 (bathroom) and 45 (bedroom) times. The final sets contain 10–18 workflows and 4.4k–8.8k characters (mean guide length 4.6k characters, versus 0.4k for Reflexion and 0.3k for EvoTest). On the first half of each room's tasks AWM reaches 76.8% (memory off 74.6%); on the second half, 57.8% (memory off 63.8%). The accumulated workflows hurt more as they grow.

## Interpretation

- **All three baselines are below iterative SD mean / EMA.** Even the best, EvoTest, is 6.4 pp (Average) and 8.4 pp (Final) below iterative SD mean (paired −6.4 ± 2.7). Reflexion, which uses the same kind of same-task feedback, is 9.5 pp below. The same information (text derived from this task's failures) works better when combined as a mean log-ratio over 1-token scores than when pasted into the prompt.
- **Reflexion and EvoTest beat memory off** by +4.8 and +8.0 pp. Both gain most in Kitchen (40 → 60 / 56) and are equal or slightly better in Bathroom and Bedroom.
- **AWM is equal to or slightly below memory off.** As the room-level workflow set grows, performance on later tasks drops. Cool (24%) and pick_two (42%) in particular fall below memory off: generic procedures induced from other tasks appear to override task-specific steps (cooling, the second object). We used neither the offline setting of the original paper nor a cap on the number of workflows, so this is not an upper bound for AWM.
- **Differences in the comparison**: iterative SD mean / EMA reuse the fixed feedback from the β = 1 sum run, whereas the baselines are on-policy (their text comes from their own trajectories). The baselines also go through the LLM judge, while iterative SD needs no success decision (it writes a reflection after every trial). On-policy iterative SD sum at β = 0.5 (81.21 / 85.18) is also above all three baselines.
