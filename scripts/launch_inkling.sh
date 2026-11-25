#!/bin/bash
# launch_inkling.sh <off|on> - start Inkling 2 Mini on DGX Spark with a vLLM nightly.
#   off: no speculative decoding / on: DSpark, 7 draft tokens
# Stop it with: docker stop inkling2mini   (--rm removes the container)
#
# Environment variables:
#   MODEL_DIR  local copy of the checkpoint; if unset, vLLM downloads
#              thinkingmachines/Inkling-2-Mini into $STATE_DIR/cache
#   STATE_DIR  host directory for the caches (default: ./inkling-state). The Triton
#              cache lives in $STATE_DIR/triton-cache, so that kernels compiled on
#              GB10 survive container restarts.
#   HOST, PORT listen address (default: 127.0.0.1:8000)
#   IMAGE      vLLM image (default: the nightly used in the README)
#
# The server serves the model as "inkling-2-mini". Our runs used MODEL_DIR; the
# download path has not been exercised.
set -eu
MODE=${1:?usage: launch_inkling.sh <off|on>}
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-647a76866b4559a74e27d1c6b55fbffe85c07ce4}
REVISION=b85c3cfa615fc413cc9af3379f491b963f3b586a
STATE_DIR=$(realpath -m "${STATE_DIR:-./inkling-state}")
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}

SPEC=()
case "$MODE" in
  on) SPEC=(--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}') ;;
  off) ;;
  *) echo "usage: launch_inkling.sh <off|on>" >&2; exit 2 ;;
esac

DOCKER_OPTS=()
if [ -n "${MODEL_DIR:-}" ]; then
  MODEL_DIR=$(realpath "$MODEL_DIR")
  DOCKER_OPTS=(-v "$MODEL_DIR:/models/Inkling-2-Mini:ro" -e HF_HUB_OFFLINE=1)
  MODEL=(/models/Inkling-2-Mini)
else
  MODEL=(thinkingmachines/Inkling-2-Mini --revision "$REVISION")
fi
mkdir -p "$STATE_DIR/cache" "$STATE_DIR/triton-cache"

docker run -d --rm --name inkling2mini --gpus all --network host --ipc host \
  -v "$STATE_DIR/cache:/root/.cache" \
  -v "$STATE_DIR/triton-cache:/root/.triton" \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  "${DOCKER_OPTS[@]}" \
  "$IMAGE" "${MODEL[@]}" \
  --served-model-name inkling-2-mini \
  --host "$HOST" --port "$PORT" \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.75 \
  --max-num-seqs 64 \
  --kv-cache-dtype auto \
  --no-enable-prefix-caching \
  --reasoning-parser inkling \
  "${SPEC[@]}"
echo "started; follow the log with: docker logs -f inkling2mini"
