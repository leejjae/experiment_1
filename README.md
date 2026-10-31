# JitRL on ALFWorld: self-distillation vs. test-time learning baselines

This repository ports [JitRL (Just-in-Time Reinforcement Learning)](https://github.com/liushiliushi/JitRL) to ALFWorld and compares the following methods under one protocol:

- **ACE + self-distillation (ACE-SD)**: a room-level ACE playbook serves as teacher context; candidate actions are scored as `log p + β(log q − log p)` with 1-token label logprobs.
- **Iterative self-distillation**: per task, the reflection f_k written after trial k defines a teacher q_k = π₁(a|s, f_k), and actions are chosen by `log π_k = log π₁ + β·R_k`. We compare four ways to combine the log-ratios into R_k: sum, replace, mean and EMA.
- **Baselines**: Reflexion, EvoTest, AWM (online), ReAct (0-shot and 2-shot), JitRL with memory off, and memory-on JitRL with judge-only rewards (Eq. 25 normalization + logit update, the best memory-on JitRL variant).

**No condition gives the environment reward or success signal to any role.** Baselines that need a success decision ask the same LLM to judge the trajectory; memory-on JitRL takes its rewards from a step-level LLM judge (gpt-oss-20b).

> The standalone entry point `ALFWorld/main.py` is the plain JitRL port and **does** feed the environment reward and `won` flag to the memory. It is not used for any reported result. For memory-on JitRL under this repository's protocol, use `experiments/alfworld_jitrl/`.

The repository also contains the WebArena JitRL code (a modified version of upstream).

## Results (ALFWorld valid_seen, 74 tasks × 5 trials)

Success rate per room, **Average / Final** (Final = 5th trial). The actor is Qwen3-32B-AWQ in every condition (T = 0.7, 30 steps).

| Method | Kitchen (20) | Living room (14) | Bathroom (20) | Bedroom (20) | Mean |
|---|---|---|---|---|---|
| ReAct (0-shot) | 44.00 / 50.00 | 67.14 / 64.29 | 50.00 / 60.00 | 70.00 / 55.00 | 57.79 / 57.32 |
| JitRL, memory off | 40.00 / 35.00 | 72.86 / 78.57 | 85.00 / 95.00 | 80.00 / 85.00 | 69.46 / 73.39 |
| JitRL, memory on (Eq. 25 + logit, judge-only) | 43.00 / 25.00 | 65.71 / 71.43 | 78.00 / 85.00 | 76.00 / 70.00 | 65.68 / 62.86 |
| Reflexion | 60.00 / 70.00 | 70.00 / 78.57 | 83.00 / 80.00 | 84.00 / 90.00 | 74.25 / 79.64 |
| EvoTest | 56.00 / 65.00 | 75.71 / 71.43 | 89.00 / 95.00 | 89.00 / 95.00 | 77.43 / 81.61 |
| AWM (online) | 43.00 / 55.00 | 61.43 / 57.14 | 85.00 / 80.00 | 78.00 / 85.00 | 66.86 / 69.29 |
| ACE-SD β = 1 | 68.00 / 80.00 | 78.57 / 78.57 | 83.00 / 80.00 | 72.00 / 85.00 | 75.39 / 80.89 |
| Iterative SD, sum, β = 1 | 64.00 / 70.00 | 75.71 / 78.57 | 86.00 / 85.00 | 80.00 / 85.00 | 76.43 / 79.64 |
| Iterative SD, sum, β = 0.5 | 66.00 / 80.00 | 82.86 / 85.71 | 86.00 / 90.00 | 90.00 / 85.00 | 81.21 / 85.18 |
| Iterative SD, replace, β = 1 † | 68.00 / 75.00 | 78.57 / 78.57 | 86.00 / 80.00 | 86.00 / 95.00 | 79.64 / 82.14 |
| Iterative SD, **mean**, β = 1 † | 66.00 / 65.00 | 87.14 / 100.00 | 91.00 / 95.00 | 91.00 / 100.00 | **83.79 / 90.00** |
| Iterative SD, EMA, β = 1 † | 65.00 / 80.00 | 84.29 / 92.86 | 88.00 / 90.00 | 89.00 / 95.00 | 81.57 / 89.46 |

† Reuses the reflection texts of the sum β = 1 run and changes only the combination rule.

`python analysis/export_trials.py` regenerates the full table (including the ACE-SD β sweep and 2-shot ReAct). Analyses:

- [ACE-SD β sweep](docs/alfworld_ace_sd/results.md)
- [Iterative SD, combination rules, over-accumulation analysis](docs/alfworld_ace_sd/iterative.md)
- [Reflexion / EvoTest / AWM](docs/alfworld_baselines/results.md)
- [Memory-on JitRL (Eq. 25 + logit, judge-only)](docs/alfworld_jitrl/results.md)

## Layout

```
ALFWorld/                 JitRL agent and environment wrapper for ALFWorld (src/), standalone entry point main.py
data/alfworld_valid_seen_74.json   the 74 tasks, their order, seeds and step budget (gamefile paths are relative to $ALFWORLD_DATA)
experiments/
  alfworld_react/         ReAct 0/2-shot, JitRL memory off
  alfworld_ace_sd/        ACE-SD (run.py, launch.py), iterative SD (run_iter.py, launch_iter.py),
                          fixed-feedback rule comparison (launch_fixedfb.py), playbook viewer (show_playbook.py)
  alfworld_baselines/     Reflexion / EvoTest / AWM (methods.py, run_baseline.py, launch.py)
  alfworld_jitrl/         memory-on JitRL, judge-only rewards: Eq. 25 + logit (agent.py), upstream retrieval k=3,
                          step judge (judge.py), one stream per room (run.py, launch.py)
  alfworld_no_feedback/   helper that strips the environment outcome from the policy input (without_outcome)
  */runs/                 per-trial JSON of the reported runs (trajectories, scores, playbooks, reflections)
analysis/                 tables and analyses (read the stored runs only; no GPU needed except judge_acc.py)
results/alfworld_trials.csv        per-trial success of every condition
scripts/serve_vllm.sh     starts the vLLM server(s)
WebArena/                 JitRL for WebArena (modified upstream)
```

## Installation

```bash
# 1) Agent environment (Python 3.11)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
alfworld-download                      # defaults to ~/.cache/alfworld
export ALFWORLD_DATA=~/.cache/alfworld

# 2) LLM server environment (a separate venv is recommended): vLLM 0.25.1, Qwen/Qwen3-32B-AWQ
pip install vllm==0.25.1
```

### LLM server

| GPUs | Command | Ports for the launchers |
|---|---|---|
| 2 × ≥ 32 GB (e.g. 2 × RTX 5090, used for the reported runs) | `scripts/serve_vllm.sh split` (one server per GPU, :8000 / :8002) | default |
| 2 × 24 GB (e.g. 2 × RTX 4090) | `scripts/serve_vllm.sh tp` (one tensor-parallel server, :8000) | `VLLM_PORTS=8000` |
| memory-on JitRL, 2 × ≥ 32 GB | `scripts/serve_vllm.sh jitrl` (actor :8000 on GPU 0, judge gpt-oss-20b :8001 on GPU 1; the reported setup) | default |
| memory-on JitRL, 2 × 24 GB | `scripts/serve_vllm.sh jitrl-tp` (both models tensor-parallel over both GPUs, actor :8000, judge :8001) | default |

On a single 24 GB GPU the weights (about 19 GB) leave almost no room for the KV cache, so use `tp`. The `jitrl-tp` memory split (actor 0.56, judge 0.36 of each GPU) has not been run on RTX 4090s. Check that both servers start and that their logs report a KV cache larger than the max model length; actor requests stay under 2.5k tokens and judge requests under about 20k.

The protocol (prompts, seeds, temperatures) is identical in all setups. Generations are not bit-identical across GPUs and batching, so reproduction is statistical.

## Reproducing the experiments

Each launcher runs several streams and restarts a crashed stream from its last committed trial. **Point `RUN_DIR` to a new folder** so that the reported runs are not overwritten. A launcher refuses to start when the source hashes differ from those recorded in an existing run folder.

```bash
export ALFWORLD_DATA=~/.cache/alfworld
# ReAct / memory off (per arm and room)
python experiments/alfworld_react/run.py --help

# ACE-SD β sweep
RUN_DIR=runs_new/ace_sd python experiments/alfworld_ace_sd/launch.py

# Iterative SD (sum), β = 1 or 0.5
RUN_DIR=runs_new/iter python experiments/alfworld_ace_sd/launch_iter.py 1
RUN_DIR=runs_new/iter python experiments/alfworld_ace_sd/launch_iter.py 0.5

# replace / mean / EMA with the same feedback (FEEDBACK_DIR = the sum β = 1 run)
RUN_DIR=runs_new/fixedfb FEEDBACK_DIR=runs_new/iter/beta_1 python experiments/alfworld_ace_sd/launch_fixedfb.py

# Reflexion / EvoTest / AWM
RUN_DIR=runs_new/baselines python experiments/alfworld_baselines/launch.py

# Memory-on JitRL, Eq. 25 + logit, judge-only (needs the judge server: serve_vllm.sh jitrl | jitrl-tp)
RUN_DIR=runs_new/jitrl python experiments/alfworld_jitrl/launch.py
python experiments/alfworld_jitrl/run.py --room kitchen --out runs_new/jitrl_pilot --limit_tasks 1 --repeats 2   # pilot

# One method, one room (pilot)
python experiments/alfworld_baselines/run_baseline.py --method evotest --room kitchen --limit_tasks 2 --trials 3 --out runs_new/pilot
```

Wall-clock time on 2 × RTX 5090: about 1 h 40 min for one iterative SD run (370 trials), and about 4 h for the three baselines (AWM is the bottleneck because it runs sequentially within a room). Expect roughly 1.3–2× longer on 2 × RTX 4090.

Inspecting the results:

```bash
python analysis/export_trials.py           # full table + results/alfworld_trials.csv
python analysis/agg_baselines.py           # baseline comparison, paired differences, judge accuracy
python analysis/overacc_evidence.py        # over-accumulation evidence for the sum rule
python experiments/alfworld_ace_sd/show_playbook.py --task 58 --room kitchen --beta 1   # playbook changes per trial
```

## WebArena

The WebArena code is [upstream JitRL](https://github.com/liushiliushi/JitRL) plus:

- support for OpenAI-compatible local servers such as vLLM; set a base URL per model with `LLM_BASE_URLS='{"qwen3-32b": "http://localhost:8000/v1"}'`
- WebArena Lite config generation (`generate_lite_configs.py`, `config_files_lite/`)
- multi-stream runs (`run_streams.py`, `run_stream.sh`)
- a similarity log for calibrating the retrieval threshold (`JITRL_SIM_LOG`, `calibrate_threshold.py`)

```bash
cd WebArena
pip install browsergym-core browsergym-experiments openai tiktoken numpy langchain pillow
# set WA_SHOPPING, WA_SHOPPING_ADMIN, WA_REDDIT, WA_GITLAB, WA_MAP, WA_WIKIPEDIA in .env
python test_webarena_lite.py --tasks 0 --repeat 5 --model gpt-4o
python test_webarena_lite.py --tasks 0 --repeat 10 --disable_memory   # memory off
```

## Credits

- JitRL: Li et al., *Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates*, arXiv:2601.18510. Original source of the WebArena code and of the ALFWorld agent design.
- ACE ([ace-agent/ace-appworld](https://github.com/ace-agent/ace-appworld), Apache-2.0): `experiments/alfworld_ace_sd/playbook.py` vendors functions from commit 928e868; the Reflector and Curator prompts are adapted to ALFWorld.
- Reflexion ([noahshinn/reflexion](https://github.com/noahshinn/reflexion), MIT): the ALFWorld reflection prompt and few-shot examples (`experiments/alfworld_baselines/prompts/reflexion_few_shot_examples.txt`).
- EvoTest (He et al.): UCB selection and the Evolver follow [yf-he/EvoTest](https://github.com/yf-he/EvoTest). We omit the `extract_state` code generation and the score-delta memory.
- AWM (Wang et al., *Agent Workflow Memory*): the online setting, reimplemented with ALFWorld prompts.
- ReAct (Yao et al.): the 2-shot examples are the official ALFWorld demos.

If you use JitRL, please cite the original paper:

```bibtex
@article{li2026just,
  title={Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates},
  author={Li, Yibo and Lin, Zijie and Deng, Ailin and Zhang, Xuan and He, Yufei and Ji, Shuo and Cao, Tri and Hooi, Bryan},
  journal={arXiv preprint arXiv:2601.18510},
  year={2026}
}
```
