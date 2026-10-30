# ALFWorld: Reflexion / EvoTest / AWM (환경 reward 없음, LLM judge)

2026-10-29 17:57 UTC 완료. 3 × 370 trial, Traceback 0. 코드: [`methods.py`](../../experiments/alfworld_baselines/methods.py), [`run_baseline.py`](../../experiments/alfworld_baselines/run_baseline.py), [`launch.py`](../../experiments/alfworld_baselines/launch.py), 프롬프트 [`prompts/`](../../experiments/alfworld_baselines/prompts/). run: `experiments/alfworld_baselines/runs/20261029/<method>/<room>/results/`.

## 설정

- 프로토콜은 반복 SD와 같다: valid_seen 74태스크 × 5 trial, 원래 순서, seed = 42 + index·101 + trial, Qwen3-32B-AWQ, actor T = 0.7, 30 step.
- actor는 "메모리 끔" arm과 같은 JitRL 에이전트(`enable_cross_mem=False`)다. 각 방법은 actor의 `guiding_prompt`에 들어갈 텍스트만 만든다.
- **환경의 성공 여부는 어떤 역할에도 주지 않는다.** 성공 판정이 필요한 곳은 같은 Qwen3-32B가 trajectory를 보고 YES/NO 1-token logprob으로 판정한다(p > 0.5면 성공).

| 방법 | 메모리 범위 | 동작 |
|---|---|---|
| Reflexion | 태스크별 | 실패로 판정된 trial 뒤에 공식 ALFWorld reflection 프롬프트와 few-shot 2개로 새 계획 작성. 최근 3개를 actor 메모리로 사용 |
| EvoTest | 태스크별 | guiding prompt 트리. trial 1은 기본 prompt, 이후 UCB(c = 1.0, d = 0.8)로 부모를 골라 Evolver(T = 0.7)가 다시 씀. 성공 판정이 나오면 최고 node로 고정(freeze_on_win) |
| AWM (online) | 방 안에서 태스크 간 누적 | 성공으로 판정된 trajectory에서 workflow를 유도해 방의 workflow 집합을 갱신 |

EvoTest에서 `extract_state` 코드 생성과 점수 변화량 메모리는 뺐다. ALFWorld에는 중간 점수가 없고, 상태 요약은 이미 actor 프롬프트에 들어간다.

## 결과

방별 Average / Final(Final = 5번째 trial).

| Test Env | 태스크 | Reflexion | EvoTest | AWM | 반복 SD mean (β=1) | 반복 SD EMA (β=1) | 반복 SD sum β=0.5 | 메모리 끔 | ReAct (0-shot) |
|---|---:|---|---|---|---|---|---|---|---|
| Kitchen | 20 | 60.00 / 70.00 | 56.00 / 65.00 | 43.00 / 55.00 | 66.00 / 65.00 | 65.00 / 80.00 | 66.00 / 80.00 | 40.00 / 35.00 | 44.00 / 50.00 |
| Living room | 14 | 70.00 / 78.57 | 75.71 / 71.43 | 61.43 / 57.14 | 87.14 / 100.00 | 84.29 / 92.86 | 82.86 / 85.71 | 72.86 / 78.57 | 67.14 / 64.29 |
| Bathroom | 20 | 83.00 / 80.00 | 89.00 / 95.00 | 85.00 / 80.00 | 91.00 / 95.00 | 88.00 / 90.00 | 86.00 / 90.00 | 85.00 / 95.00 | 50.00 / 60.00 |
| Bedroom | 20 | 84.00 / 90.00 | 89.00 / 95.00 | 78.00 / 85.00 | 91.00 / 100.00 | 89.00 / 95.00 | 90.00 / 85.00 | 80.00 / 85.00 | 70.00 / 55.00 |
| 평균 | - | 74.25 / 79.64 | 77.43 / 81.61 | 66.86 / 69.29 | **83.79 / 90.00** | 81.57 / 89.46 | 81.21 / 85.18 | 69.46 / 73.39 | 57.79 / 57.32 |

trial별 방 평균 SR과 연속 trial의 성공→실패 비율:

| | trial 1 | 2 | 3 | 4 | 5 | 성공→실패 |
|---|---:|---:|---:|---:|---:|---:|
| Reflexion | 64.3 | 73.6 | 79.6 | 74.1 | 79.6 | 10.1% |
| EvoTest | 73.9 | 71.1 | 79.1 | 81.4 | 81.6 | 12.8% |
| AWM | 63.6 | 67.3 | 65.5 | 68.6 | 69.3 | 10.7% |
| 반복 SD mean | 72.1 | 82.1 | 88.2 | 86.4 | 90.0 | 8.2% |
| 메모리 끔 | 66.8 | 65.2 | 73.4 | 68.6 | 73.4 | 16.8% |
| ReAct (0-shot) | 52.3 | 58.4 | 63.6 | 57.3 | 57.3 | 16.6% |

태스크 단위 paired 차이(전체 5 trial, 방 평균, ± SE):

| | 메모리 끔 대비 | 반복 SD mean 대비 | ReAct 0-shot 대비 |
|---|---:|---:|---:|
| Reflexion | +4.8 ± 2.8 | −9.5 ± 2.7 | +16.5 ± 3.7 |
| EvoTest | +8.0 ± 2.4 | −6.4 ± 2.7 | +19.6 ± 3.7 |
| AWM | −2.6 ± 2.8 | −16.9 ± 3.1 | +9.1 ± 3.8 |

참고로 반복 SD mean은 메모리 끔 대비 +14.3 ± 2.4다.

task type별 SR(5 trial 전체, %):

| | look | simple | clean | cool | heat | pick_two |
|---|---:|---:|---:|---:|---:|---:|
| Reflexion | 73 | 94 | 70 | 60 | 60 | 59 |
| EvoTest | 84 | 95 | 72 | 44 | 64 | 65 |
| AWM | 78 | 91 | 72 | 24 | 56 | 42 |
| 반복 SD mean | 89 | 98 | 86 | 68 | 68 | 68 |
| 메모리 끔 | 78 | 92 | 60 | 36 | 56 | 52 |

## judge 품질 (이번 run, 실제 성공 여부와 대조)

| | 정확도 | 성공 판정 precision | 실제 성공 recall | 실제 실패 recall | FP / FN |
|---|---:|---:|---:|---:|---|
| Reflexion | 0.895 | 0.984 | 0.873 | 0.957 | 4 / 35 |
| EvoTest | 0.892 | 0.996 | 0.864 | 0.988 | 1 / 39 |
| AWM | 0.895 | 0.986 | 0.855 | 0.975 | 3 / 36 |

사전 점검(저장된 150개, 0.88)과 비슷하다. 오류는 대부분 FN(성공을 실패로 판정)이다. 그래서 Reflexion은 이미 성공한 태스크에도 reflection을 쓰고, EvoTest는 freeze가 늦어지며, AWM은 유도 기회를 일부 놓친다. 반대로 FP는 거의 없어서 잘못된 trajectory가 workflow나 freeze로 들어가는 경우는 드물다.

## 방법별 동작

- **Reflexion**: reflection 125개(370 trial 중). trial 2–5 평균 76.7로 trial 1(64.3)보다 오르지만, trial 4에서 떨어지는 등 고르지 않다.
- **EvoTest**: root 74, evolve 92, freeze 204 trial. freeze된 trial의 SR은 86.3%, evolve 직후 trial은 62.0%다. trial 1이 73.9로 높은데, 기본 prompt("Explore systematically and examine objects…")만으로도 메모리 끔 trial 1(66.8)보다 높다.
- **AWM**: 방별 유도 횟수는 kitchen 44, living room 41, bathroom 86, bedroom 45다. 최종 workflow는 10–18개, 4.4k–8.8k자다(평균 guide 4.6k자, Reflexion 0.4k, EvoTest 0.3k). 방 안의 앞쪽 절반 태스크는 76.8%(메모리 끔 74.6%)이고, 뒤쪽 절반은 57.8%(메모리 끔 63.8%)다. workflow가 쌓일수록 오히려 손해다.

## 해석

- **세 baseline 모두 반복 SD mean / EMA보다 낮다.** 가장 좋은 EvoTest도 반복 SD mean보다 Average 6.4%p, Final 8.4%p 낮다(paired −6.4 ± 2.7). 같은 태스크 피드백을 쓰는 Reflexion과 비교하면 차이는 9.5%p다. 같은 종류의 정보(그 태스크의 실패에서 나온 텍스트)라도, 프롬프트에 붙여 넣는 것보다 1-token 채점에서 평균 log-ratio로 결합하는 쪽이 낫다.
- **Reflexion과 EvoTest는 메모리 끔보다 높다.** 각각 +4.8, +8.0%p. 두 방법 모두 Kitchen에서 크게 오르고(40 → 60 / 56), Bathroom·Bedroom은 비슷하거나 약간 높다.
- **AWM은 메모리 끔과 차이가 없거나 약간 낮다.** 방 단위로 쌓인 workflow가 길어지면서 뒤쪽 태스크에서 성능이 떨어진다. 특히 cool(24%)과 pick_two(42%)가 메모리 끔보다 낮다. 다른 태스크에서 유도한 일반 절차가 이 태스크의 세부 단계(냉각, 두 번째 물체)를 덮는 것으로 보인다. 원 논문의 offline 설정이나 workflow 수 제한은 쓰지 않았으므로 AWM의 상한은 아니다.
- **비교 조건 차이**: 반복 SD mean / EMA는 β = 1 sum run에서 나온 피드백을 고정해서 쓴 결과다. baseline들은 자기 trajectory에서 나온 텍스트를 쓰는 on-policy다. 그리고 baseline은 LLM judge를 거치지만 반복 SD는 성공 판정이 필요 없다(실패 여부와 관계없이 매 trial reflection을 쓴다). 반복 SD β = 0.5(on-policy sum)도 81.21 / 85.18로 세 baseline보다 높다.
