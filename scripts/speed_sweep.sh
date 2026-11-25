#!/bin/bash
# speed_sweep.sh <tag> [c ...] - the throughput sweep from the README.
# One server (started with launch_inkling.sh, then warmed up with warmup_variants.py)
# stays up; only the client's --max-concurrency changes between points.
# Results go to $OUT_DIR/<tag>/; summarize them with summarize_speed.py.
#
# Environment variables:
#   SHAREGPT   path to ShareGPT_V3_unfiltered_cleaned_split.json (required, not included)
#   MODEL_DIR  local copy of the checkpoint, used only for its tokenizer (required)
#   OUT_DIR    where to write results (default: ./results)
#   BASE_URL   server root (default: http://localhost:8000)
#   CLIENT_IMAGE  image whose `vllm bench serve` is used (default: the one in the README,
#                 vLLM 0.19.1)
#
# Our runs also waited for the machine to cool down between points when it got hot
# (GPU >= 84 C); that wait was never triggered and is not part of this script.
set -u
TAG=${1:?usage: speed_sweep.sh <tag> [c ...]}; shift
CS=${*:-1 2 4 8 16 32}
: "${SHAREGPT:?set SHAREGPT}" "${MODEL_DIR:?set MODEL_DIR}"
BASE_URL=${BASE_URL:-http://localhost:8000}
CLIENT_IMAGE=${CLIENT_IMAGE:-vllm/vllm-openai:gemma4-cu130}
OUT=$(realpath -m "${OUT_DIR:-./results}/$TAG")
mkdir -p "$OUT"
METRICS='^vllm:(spec_decode|prefix_cache|num_preemptions|generation_tokens_total|prompt_tokens_total)'
for c in $CS; do
  curl -s "$BASE_URL/metrics" | grep -E "$METRICS" > "$OUT/metrics_before_c$c.txt"
  docker run --rm --network host \
    -v "$(realpath "$SHAREGPT"):/datasets/sharegpt.json:ro" \
    -v "$(realpath "$MODEL_DIR"):/tokenizer:ro" \
    -v "$OUT:/results" \
    --entrypoint vllm "$CLIENT_IMAGE" \
    bench serve \
      --backend openai-chat --base-url "$BASE_URL" \
      --endpoint /v1/chat/completions \
      --model inkling-2-mini --tokenizer /tokenizer --trust-remote-code \
      --dataset-name sharegpt \
      --dataset-path /datasets/sharegpt.json \
      --num-prompts 200 --max-concurrency "$c" \
      --save-result --result-dir /results --result-filename "inkling2mini-$TAG-c$c.json" \
      > "$OUT/client_c$c.log" 2>&1
  echo "rc=$? c=$c" >> "$OUT/progress.txt"
  curl -s "$BASE_URL/metrics" | grep -E "$METRICS" > "$OUT/metrics_after_c$c.txt"
done
