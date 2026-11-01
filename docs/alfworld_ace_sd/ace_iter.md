# Iterative SD with ACE playbook teachers (mean / EMA)

This experiment combines the two self-distillation variants of this repository:

| | teacher context | number of teachers at trial k | combination |
|---|---|---|---|
| ACE-SD (`run.py`) | current room playbook M | 1 | `log p + β(log q − log p)` |
| Iterative SD (`run_iter.py`, `launch_fixedfb.py`) | per-task reflection f_j | k − 1 (reflections of trials 1..k−1) | `log p + β·R`, R = sum / replace / mean / EMA of r_j |
| **Iterative ACE (this, `run_ace_iter.py`)** | **room playbook snapshots M_1..M_k** | **k** | `log p + β·R`, R = **mean** or **EMA** of r_j |

The mean / EMA rows in the README table (marked †) are the reflection-teacher variant with fixed feedback.
This experiment has not been run yet; it replaces the reflection f_j with the ACE playbook.

## Method

- **Actor, proposals and scoring.** Same as ACE-SD (`agent.py`). The memory-free JitRL proposal call (T = 0.7) gives up to 3 candidates. The same Qwen3-32B-AWQ then scores them with 1-token label logprobs: once without context (log p), and once per teacher with the playbook in a `TEACHER_BLOCK` (log q_j).
- **Playbook.** One playbook per room per split. It starts from the three empty ACE sections and is updated after every trial by the ACE Reflector and Curator (T = 0, `prompts/reflector.txt`, `prompts/curator.txt`). It is on-policy: the trajectory comes from this run's own actor. The Reflector sees only the task and the ACTION / OBSERVATION trajectory. No role ever sees the reward, the `won` flag or a judge. This is identical to ACE-SD.
- **Teachers.** For trial k (1..5) of a task, M_j is the room playbook just before trial j of the same task, for j = 1..k. All k snapshots are fixed for the whole trial.
  - Trial 1 has one teacher, M_1, so it is exactly ACE-SD β = 1.
  - A snapshot without bullets gives log q = log p (r = 0), as in ACE-SD.
  - Identical snapshots are scored once (cache). The scores are unchanged by the cache.
- **Combination** (`agent_iter.combine`), with r_j = log q_j − log p per candidate:
  - mean: R = (1/k) Σ_j r_j. The effective β stays 1, so there is no over-accumulation.
  - EMA: R_1 = r_1, R_j = (1 − λ) R_{j−1} + λ r_j, with λ = 0.5.
  - The action is argmax_a [log p(a) + β R(a)], with β = 1.

## Task cohorts (one per machine)

| split | plan file | tasks (kitchen / living_room / bathroom / bedroom) | trials per rule | trials total (mean + EMA) | AlfredTWEnv mode |
|---|---|---|---|---|---|
| valid_seen | `data/alfworld_valid_seen_74.json` | 74 (20 / 14 / 20 / 20) | 370 | 740 | eval_in_distribution |
| valid_unseen | `data/alfworld_valid_unseen_59.json` | 59 (16 / 11 / 16 / 16) | 295 | 590 | eval_out_of_distribution |

- **valid_seen** is the cohort of every result in the README.
- **valid_unseen** is the valid_unseen cohort of the earlier JitRL comparison: same games, order and `index` values. It is not the full 134-game split.
- **Protocol (both splits).** Original order, 5 consecutive trials per task, seed = 42 + index·101 + trial, 30 steps.

## Running on two machines

Each machine runs one split: 8 streams = {mean, EMA} × 4 rooms. A room is one sequential stream, because its playbook depends on all earlier trials. The rooms and rules run in parallel.

### 0. Get the code (a machine that already has a clone)

```bash
cd experiment_1
git status            # must show no modified tracked files; untracked run folders are fine
git pull              # adds this experiment; existing run folders are not touched
```

On a new machine, run `git clone https://github.com/leejjae/experiment_1.git` and follow the Installation section of the README. The ALFWorld data must contain both `json_2.1.1/valid_seen` and `json_2.1.1/valid_unseen`; `alfworld-download` fetches both. The runner checks that every game file of the split exists before it starts.

### 1. Start the actor server(s)

```bash
# in the vLLM environment, or set VLLM_PYTHON=/path/to/vllm-venv/bin/python
bash scripts/serve_vllm.sh split   # 2 x >= 32 GB: one server per GPU on :8000 and :8002
bash scripts/serve_vllm.sh tp      # 2 x 24 GB: one tensor-parallel server on :8000 -> use VLLM_PORTS=8000 below
curl -s localhost:8000/v1/models   # wait until each port answers (a few minutes)
```

### 2. Launch

Machine A (valid_seen):

```bash
export ALFWORLD_DATA=~/.cache/alfworld
SPLIT=valid_seen RUN_DIR=runs_new/ace_iter \
  nohup python experiments/alfworld_ace_sd/launch_ace_iter.py > launch_ace_iter_valid_seen.log 2>&1 &
```

Machine B (valid_unseen):

```bash
export ALFWORLD_DATA=~/.cache/alfworld
SPLIT=valid_unseen RUN_DIR=runs_new/ace_iter \
  nohup python experiments/alfworld_ace_sd/launch_ace_iter.py > launch_ace_iter_valid_unseen.log 2>&1 &
```

- Prefix `VLLM_PORTS=8000` with `tp`.
- `SPLIT` is required.
- The 8 streams are spread over the ports round-robin.

Output layout:

```
RUN_DIR/<split>/manifest.json                         split, rules, beta, lam, ports, trial targets, source hashes
RUN_DIR/<split>/logs/<rule>_<room>.log                one line per committed trial (COMMITTED ...), tracebacks
RUN_DIR/<split>/<rule>/beta_1/<room>/results/task_XXXX_run_YY.json   one file per trial
```

### 3. Monitor

```bash
tail -f launch_ace_iter_<split>.log                                  # START / RECOVERY / GIVE_UP / COMPLETE
tail -n 2 runs_new/ace_iter/<split>/logs/*.log                       # latest committed trial per stream
python analysis/agg_ace_iter.py runs_new/ace_iter                    # progress + Average / Final so far
grep -l Traceback runs_new/ace_iter/<split>/logs/*.log               # streams that crashed at least once
```

- **Crashes.** A crashed stream is restarted automatically, at most 3 times. It resumes after its last committed trial and rebuilds the teacher snapshots of the current task from the committed `playbook_before` values.
- **Restarting the launcher.** If the launcher itself stops (reboot, kill), start the same command again with the same `SPLIT` and `RUN_DIR`. It refuses to start if any source file or the plan changed since the manifest was written. Never edit the code in the middle of a run.

### 4. After COMPLETE

```bash
python analysis/agg_ace_iter.py runs_new/ace_iter --check      # must report 0 problems for every stream
```

Then stop the vLLM servers. To put both splits in one place, copy `RUN_DIR/<split>` from one machine into the other's `RUN_DIR`, for example `rsync -a machineB:experiment_1/runs_new/ace_iter/valid_unseen runs_new/ace_iter/`. Then run the aggregation once over both.

### Expected time

The valid_seen ACE-SD β = 1 run took 6–9 h per room with 20 streams sharing two RTX 5090 servers. This experiment uses 8 streams and adds up to 4 extra scoring calls per step on trials 2–5. Identical snapshots reuse one call. Plan for roughly 6–10 h for valid_seen on 2 × RTX 5090, and about 20 % less for valid_unseen (fewer tasks). On 2 × RTX 4090 with `tp`, expect 1.3–2× longer. The longest stream is kitchen (100 or 80 sequential trials).

## Per-trial JSON

On top of the ACE-SD fields (`steps`, `success`, `playbook_before`, `playbook_after`, `reflection`, `curator_raw`, `operations`, `counters`, ...):

- `split`: valid_seen or valid_unseen.
- `rule`, `lam`, `beta`: the combination rule and its parameters.
- `n_teachers`: k, the number of teacher snapshots for this trial.
- `steps[i].log_p`: base label logprobs of the candidates.
- `steps[i].log_q`: one list per teacher, in the order M_1..M_k.
- `steps[i].scores`: log p + β R.
- `steps[i].chosen`: index of the executed candidate.
- `steps[i].base_choice`: argmax of log p.
- `counters.n_teacher_changed`: steps where the teachers changed the choice.

`success` (from the environment) is written for evaluation only and is never part of any prompt.

## Pilot (valid_seen, kitchen, 2 tasks × 3 trials, both rules)

- 185 steps.
- 0 score and 0 argmax mismatches.
- n_teachers = k at every trial; the playbook chain was unbroken.
- Trial 1 equals ACE-SD β = 1.
- A kill / resume at trial 3 restored all three snapshots.
- A dry start on valid_unseen loaded the 134-game out-of-distribution split and the 59-task plan.
