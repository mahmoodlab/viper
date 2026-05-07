#!/usr/bin/env bash
# Evaluate a self-served open-weight model behind a vLLM endpoint.
# Usage: VLLM_URL=http://localhost:8000/v1 MODEL=Qwen/Qwen3.5-27B examples/eval_vllm.sh
set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:?Set MODEL to the served model name (e.g. Qwen/Qwen3.5-27B)}"
JUDGE_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is required for the LLM judge}"

# Quick health check on the vLLM endpoint.
if ! curl -sf -o /dev/null "${VLLM_URL%/v1}/v1/models"; then
    echo "vLLM endpoint at ${VLLM_URL} did not respond. Is the server running?" >&2
    exit 2
fi

viper-eval \
    --model "${MODEL}" \
    --api-base "${VLLM_URL}" \
    --api-key "dummy" \
    --judge-api-key "${JUDGE_API_KEY}" \
    --output ./eval_logs

echo
echo "✅ Done. results.json written to eval_logs/${MODEL//\//--}/<timestamp>/"
