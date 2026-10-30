# JitRL on ALFWorld: self-distillation과 test-time learning baseline 비교

[JitRL (Just-in-Time Reinforcement Learning)](https://github.com/liushiliushi/JitRL)을 ALFWorld로 옮기고, 그 위에서 다음을 같은 프로토콜로 비교한 코드와 결과입니다.

- **ACE + self-distillation (ACE-SD)**: 방 단위로 누적되는 ACE playbook을 teacher context로 쓰고, 1-token label logprob으로 `log p + β(log q − log p)`를 채점합니다.
- **반복 self-distillation**: 태스크마다 trial k의 reflection f_k로 teacher q_k = π₁(a|s, f_k)를 만들고, `log π_k = log π₁ + β·R_k`로 행동을 고릅니다. R_k의 결합 규칙은 sum / replace / mean / EMA를 비교합니다.
- **Baseline**: Reflexion, EvoTest, AWM(online), ReAct(0-shot / 2-shot), 메모리를 끈 JitRL.

모든 조건에서 **환경의 reward나 성공 여부를 어떤 역할에도 주지 않습니다.** 성공 판정이 필요한 baseline은 같은 LLM이 trajectory를 보고 판정합니다.

WebArena용 JitRL 코드(upstream 기반 수정본)도 함께 들어 있습니다.

## 결과 요약 (ALFWorld valid_seen 74태스크 × 5 trial)

방별 성공률 **Average / Final**(Final = 5번째 trial)입니다. 모든 조건의 actor는 Qwen3-32B-AWQ(T = 0.7, 30 step)입니다.

| 방법 | Kitchen (20) | Living room (14) | Bathroom (20) | Bedroom (20) | 평균 |
|---|---|---|---|---|---|
| ReAct (0-shot) | 44.00 / 50.00 | 67.14 / 64.29 | 50.00 / 60.00 | 70.00 / 55.00 | 57.79 / 57.32 |
| JitRL 메모리 끔 | 40.00 / 35.00 | 72.86 / 78.57 | 85.00 / 95.00 | 80.00 / 85.00 | 69.46 / 73.39 |
| Reflexion | 60.00 / 70.00 | 70.00 / 78.57 | 83.00 / 80.00 | 84.00 / 90.00 | 74.25 / 79.64 |
| EvoTest | 56.00 / 65.00 | 75.71 / 71.43 | 89.00 / 95.00 | 89.00 / 95.00 | 77.43 / 81.61 |
| AWM (online) | 43.00 / 55.00 | 61.43 / 57.14 | 85.00 / 80.00 | 78.00 / 85.00 | 66.86 / 69.29 |
| ACE-SD β=1 | 68.00 / 80.00 | 78.57 / 78.57 | 83.00 / 80.00 | 72.00 / 85.00 | 75.39 / 80.89 |
| 반복 SD sum β=1 | 64.00 / 70.00 | 75.71 / 78.57 | 86.00 / 85.00 | 80.00 / 85.00 | 76.43 / 79.64 |
| 반복 SD sum β=0.5 | 66.00 / 80.00 | 82.86 / 85.71 | 86.00 / 90.00 | 90.00 / 85.00 | 81.21 / 85.18 |
| 반복 SD replace β=1 † | 68.00 / 75.00 | 78.57 / 78.57 | 86.00 / 80.00 | 86.00 / 95.00 | 79.64 / 82.14 |
| 반복 SD **mean** β=1 † | 66.00 / 65.00 | 87.14 / 100.00 | 91.00 / 95.00 | 91.00 / 100.00 | **83.79 / 90.00** |
| 반복 SD EMA β=1 † | 65.00 / 80.00 | 84.29 / 92.86 | 88.00 / 90.00 | 89.00 / 95.00 | 81.57 / 89.46 |

† sum β=1 run의 reflection 텍스트를 고정해 다시 쓰고 결합 규칙만 바꾼 결과입니다.

전체 표(ACE-SD β sweep, ReAct 2-shot 포함)는 `python analysis/export_trials.py`로 다시 만들 수 있습니다. 분석은 [docs/](docs/)에 있습니다(한국어).

- [ACE-SD β sweep](docs/alfworld_ace_sd/results.md)
- [반복 SD, 결합 규칙 비교, 과누적 분석](docs/alfworld_ace_sd/iterative.md)
- [Reflexion / EvoTest / AWM](docs/alfworld_baselines/results.md)

## 구조

```
ALFWorld/                 ALFWorld용 JitRL 에이전트와 환경 래퍼 (src/), 단독 실행 진입점 main.py
data/alfworld_valid_seen_74.json   태스크 74개, 순서, seed, step 수 (gamefile은 $ALFWORLD_DATA 기준 상대 경로)
experiments/
  alfworld_react/         ReAct 0/2-shot, JitRL 메모리 끔
  alfworld_ace_sd/        ACE-SD (run.py, launch.py), 반복 SD (run_iter.py, launch_iter.py),
                          고정 피드백 규칙 비교 (launch_fixedfb.py), playbook 보기 (show_playbook.py)
  alfworld_baselines/     Reflexion / EvoTest / AWM (methods.py, run_baseline.py, launch.py)
  alfworld_no_feedback/   환경 결과를 정책 입력에서 지우는 헬퍼 (without_outcome)
  */runs/                 발표한 결과의 trial별 JSON (trajectory, 점수, playbook·reflection 포함)
analysis/                 표와 분석 스크립트 (raw run만 읽으므로 GPU 불필요, judge_acc.py만 서버 필요)
results/alfworld_trials.csv        모든 조건의 trial별 성공 여부
scripts/serve_vllm.sh     vLLM 서버 실행
WebArena/                 WebArena용 JitRL (upstream 수정본)
```

## 설치

```bash
# 1) 에이전트 환경 (Python 3.11)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
alfworld-download                      # 기본 위치 ~/.cache/alfworld
export ALFWORLD_DATA=~/.cache/alfworld

# 2) LLM 서버 환경 (별도 venv 권장): vLLM 0.25.1, Qwen/Qwen3-32B-AWQ
pip install vllm==0.25.1
```

### LLM 서버

| GPU | 실행 | launcher에 줄 포트 |
|---|---|---|
| 32 GB 이상 × 2 (예: RTX 5090 × 2, 원 실험 환경) | `scripts/serve_vllm.sh split` (GPU당 서버 1개, :8000 / :8002) | 기본값 |
| 24 GB × 2 (예: RTX 4090 × 2) | `scripts/serve_vllm.sh tp` (tensor parallel 서버 1개, :8000) | `VLLM_PORTS=8000` |

24 GB GPU 한 장에는 가중치(약 19 GB)를 올리고 나면 KV cache가 거의 남지 않으므로 `tp`를 쓰세요. 프롬프트, seed, 온도 등 프로토콜은 두 구성에서 같습니다. 다만 GPU와 배칭이 다르면 생성 결과가 bit 단위로 같지는 않으므로, 재현은 통계적 의미의 재현입니다.

## 실험 재현

각 launcher는 stream을 여러 개 띄우고, 죽은 stream은 마지막 trial부터 이어서 다시 시작합니다. 발표한 run 폴더를 덮어쓰지 않도록 **새 run은 `RUN_DIR`로 다른 폴더를 지정**하세요(소스 hash가 다르면 launcher가 시작을 거부합니다).

```bash
export ALFWORLD_DATA=~/.cache/alfworld
# ReAct / 메모리 끔 (arm별, 방별)
python experiments/alfworld_react/run.py --help

# ACE-SD β sweep
RUN_DIR=runs_new/ace_sd python experiments/alfworld_ace_sd/launch.py

# 반복 SD (sum), β = 1 또는 0.5
RUN_DIR=runs_new/iter python experiments/alfworld_ace_sd/launch_iter.py 1
RUN_DIR=runs_new/iter python experiments/alfworld_ace_sd/launch_iter.py 0.5

# 같은 피드백으로 replace / mean / EMA (FEEDBACK_DIR = sum β=1 run)
RUN_DIR=runs_new/fixedfb FEEDBACK_DIR=runs_new/iter/beta_1 python experiments/alfworld_ace_sd/launch_fixedfb.py

# Reflexion / EvoTest / AWM
RUN_DIR=runs_new/baselines python experiments/alfworld_baselines/launch.py

# 한 조건, 한 방만 (pilot)
python experiments/alfworld_baselines/run_baseline.py --method evotest --room kitchen --limit_tasks 2 --trials 3 --out runs_new/pilot
```

참고 소요 시간(RTX 5090 × 2): 반복 SD 370 trial 약 1시간 40분, baseline 3종 약 4시간(AWM은 방 안에서 순차 실행이라 가장 깁니다). RTX 4090 × 2에서는 대략 1.3–2배로 예상합니다.

결과 보기:

```bash
python analysis/export_trials.py           # 전체 표 + results/alfworld_trials.csv
python analysis/agg_baselines.py           # baseline 비교, paired 차이, judge 정확도
python analysis/overacc_evidence.py        # sum 규칙의 과누적 증거
python experiments/alfworld_ace_sd/show_playbook.py --task 58 --room kitchen --beta 1   # trial별 playbook 변화
```

## WebArena

[upstream JitRL](https://github.com/liushiliushi/JitRL)의 WebArena 코드에 다음을 더한 것입니다: OpenAI 호환 로컬 서버(vLLM) 지원(`LLM_BASE_URLS='{"qwen3-32b": "http://localhost:8000/v1"}'`처럼 모델별 주소 지정), WebArena Lite 설정 생성(`generate_lite_configs.py`, `config_files_lite/`), 여러 stream 실행(`run_streams.py`, `run_stream.sh`), retrieval threshold 보정용 유사도 로그(`JITRL_SIM_LOG`, `calibrate_threshold.py`).

```bash
cd WebArena
pip install browsergym-core browsergym-experiments openai tiktoken numpy langchain pillow
# .env 에 WA_SHOPPING, WA_SHOPPING_ADMIN, WA_REDDIT, WA_GITLAB, WA_MAP, WA_WIKIPEDIA URL 설정
python test_webarena_lite.py --tasks 0 --repeat 5 --model gpt-4o
python test_webarena_lite.py --tasks 0 --repeat 10 --disable_memory   # 메모리 끔
```

## 출처

- JitRL: Li et al., *Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates*, arXiv:2601.18510. WebArena 코드와 ALFWorld 에이전트 설계의 원본입니다.
- ACE ([ace-agent/ace-appworld](https://github.com/ace-agent/ace-appworld), Apache-2.0): `experiments/alfworld_ace_sd/playbook.py`는 commit 928e868의 함수를 가져온 것이고, Reflector/Curator 프롬프트는 ALFWorld에 맞게 고친 것입니다.
- Reflexion ([noahshinn/reflexion](https://github.com/noahshinn/reflexion), MIT): ALFWorld reflection 프롬프트와 few-shot 예시(`experiments/alfworld_baselines/prompts/reflexion_few_shot_examples.txt`).
- EvoTest (He et al.): UCB 선택과 Evolver 구성은 [yf-he/EvoTest](https://github.com/yf-he/EvoTest)를 따랐고, `extract_state` 코드 생성과 점수 변화 메모리는 뺐습니다.
- AWM (Wang et al., *Agent Workflow Memory*): online 설정을 ALFWorld 프롬프트로 다시 구현했습니다.
- ReAct (Yao et al.): 2-shot 예시는 공식 ALFWorld demo입니다.

JitRL을 쓰신다면 원 논문을 인용해 주세요.

```bibtex
@article{li2026just,
  title={Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates},
  author={Li, Yibo and Lin, Zijie and Deng, Ailin and Zhang, Xuan and He, Yufei and Ji, Shuo and Cao, Tri and Hooi, Bryan},
  journal={arXiv preprint arXiv:2601.18510},
  year={2026}
}
```
