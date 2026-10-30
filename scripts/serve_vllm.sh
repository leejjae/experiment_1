#!/usr/bin/env bash
# Start the Qwen3-32B-AWQ server(s) that every ALFWorld experiment talks to.
#
#   scripts/serve_vllm.sh split   # one server per GPU on :8000 and :8002 (needs >= 32 GB per GPU, e.g. RTX 5090 x2)
#   scripts/serve_vllm.sh tp      # one tensor-parallel server on :8000 over 2 GPUs (e.g. RTX 4090 x2)
#
# With `tp`, run the launchers with VLLM_PORTS=8000.
set -euo pipefail
MODE=${1:-split}
COMMON=(--model Qwen/Qwen3-32B-AWQ --served-model-name qwen3-32b --max-model-len 32768
        --max-logprobs 20 --enable-prefix-caching)
export VLLM_USE_FLASHINFER_SAMPLER=0
case "$MODE" in
  split)
    for g in 0 1; do
      CUDA_VISIBLE_DEVICES=$g nohup python -m vllm.entrypoints.openai.api_server "${COMMON[@]}" \
        --gpu-memory-utilization 0.94 --max-num-seqs 16 --port $((8000 + 2 * g)) > vllm_$g.log 2>&1 &
    done ;;
  tp)
    CUDA_VISIBLE_DEVICES=0,1 nohup python -m vllm.entrypoints.openai.api_server "${COMMON[@]}" \
      --tensor-parallel-size 2 --gpu-memory-utilization 0.94 --max-num-seqs 32 --port 8000 > vllm_tp.log 2>&1 & ;;
  *) echo "usage: $0 split|tp" >&2; exit 1 ;;
esac
