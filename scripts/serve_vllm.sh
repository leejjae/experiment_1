#!/usr/bin/env bash
# Start the vLLM servers the ALFWorld experiments talk to.
#
#   scripts/serve_vllm.sh split      # Qwen3-32B-AWQ, one server per GPU on :8000 and :8002 (>= 32 GB per GPU, e.g. RTX 5090 x2)
#   scripts/serve_vllm.sh tp         # Qwen3-32B-AWQ, one tensor-parallel server on :8000 over 2 GPUs (e.g. RTX 4090 x2)
#   scripts/serve_vllm.sh jitrl      # memory-on JitRL: actor Qwen3-32B-AWQ on GPU 0 (:8000) + judge gpt-oss-20b on GPU 1 (:8001)
#                                    #   (the reported setup, >= 32 GB per GPU)
#   scripts/serve_vllm.sh jitrl-tp   # memory-on JitRL on 2 x 24 GB: both models tensor-parallel over both GPUs,
#                                    #   actor :8000 (max length 16k) + judge :8001. Check the KV-cache lines in the logs.
#
# With `tp`, run the SD / baseline launchers with VLLM_PORTS=8000.
set -euo pipefail
MODE=${1:-split}
ACTOR=(--model Qwen/Qwen3-32B-AWQ --served-model-name qwen3-32b --max-logprobs 20 --enable-prefix-caching)
JUDGE=(--model openai/gpt-oss-20b --served-model-name gpt-oss-20b --enable-prefix-caching)
# VLLM_PYTHON: the interpreter of the vLLM environment, if `python` on PATH is a different venv.
SERVE=("${VLLM_PYTHON:-python}" -m vllm.entrypoints.openai.api_server)
export VLLM_USE_FLASHINFER_SAMPLER=0
case "$MODE" in
  split)
    for g in 0 1; do
      CUDA_VISIBLE_DEVICES=$g nohup "${SERVE[@]}" "${ACTOR[@]}" --max-model-len 32768 \
        --gpu-memory-utilization 0.94 --max-num-seqs 16 --port $((8000 + 2 * g)) > vllm_$g.log 2>&1 &
    done ;;
  tp)
    CUDA_VISIBLE_DEVICES=0,1 nohup "${SERVE[@]}" "${ACTOR[@]}" --max-model-len 32768 \
      --tensor-parallel-size 2 --gpu-memory-utilization 0.94 --max-num-seqs 32 --port 8000 > vllm_tp.log 2>&1 & ;;
  jitrl)
    CUDA_VISIBLE_DEVICES=0 nohup "${SERVE[@]}" "${ACTOR[@]}" --max-model-len 32768 \
      --gpu-memory-utilization 0.94 --max-num-seqs 16 --port 8000 > vllm_actor.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 nohup "${SERVE[@]}" "${JUDGE[@]}" --max-model-len 32768 \
      --gpu-memory-utilization 0.90 --max-num-seqs 8 --port 8001 > vllm_judge.log 2>&1 & ;;
  jitrl-tp)
    # gpu-memory-utilization is a per-server budget, so the two fractions must sum below 1.
    # Start the actor first and wait until it is up, so the two do not profile memory at the same time.
    CUDA_VISIBLE_DEVICES=0,1 nohup "${SERVE[@]}" "${ACTOR[@]}" --max-model-len 16384 --tensor-parallel-size 2 \
      --gpu-memory-utilization 0.56 --max-num-seqs 8 --port 8000 > vllm_actor.log 2>&1 &
    until curl -sf localhost:8000/v1/models > /dev/null; do sleep 10; done
    CUDA_VISIBLE_DEVICES=0,1 nohup "${SERVE[@]}" "${JUDGE[@]}" --max-model-len 24576 --tensor-parallel-size 2 \
      --gpu-memory-utilization 0.36 --max-num-seqs 4 --port 8001 > vllm_judge.log 2>&1 & ;;
  *) echo "usage: $0 split|tp|jitrl|jitrl-tp" >&2; exit 1 ;;
esac
