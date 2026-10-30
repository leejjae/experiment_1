#!/usr/bin/env bash
# One stream = one (site, arm) pair. Tasks run in order, each repeated L consecutive times
# (paper Sec. 5.2.1: "each task receives L consecutive episode trials").
set -uo pipefail
TASK_ID="$1"; TAG="$2"; L="${3:-2}"; STEPS="${4:-30}"
set -a; . ./env_setup.txt; set +a
exec "${PYTHON:-python}" run.py \
  --task_name "walite.${TASK_ID}" --config_dir config_files_lite --repeat_runs "$L" \
  --llm_model qwen3-32b --llm_eval gpt-oss-20b --llm_extract gpt-oss-20b \
  --logit_mode verbalized --update_rule prob \
  --gamma 0.95 --retrieval_threshold 0.1 --retrieval_top_k 10 --top_actions 3 \
  --llm_temperature 0.7 --max_steps "$STEPS" --max_obs_chars 24000 \
  --memory_scope site --headless True \
  --result_dir "results/${TAG}"
