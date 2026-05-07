#!/usr/bin/env bash
# Evaluate a Claude model via Anthropic's OpenAI-compatible adapter.
# Anthropic exposes /v1/openai for direct OpenAI Chat Completions interop.
set -euo pipefail

MODEL="${MODEL:-claude-sonnet-4-6}"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "Set ANTHROPIC_API_KEY (Anthropic console -> API Keys) first." >&2
    exit 1
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Set OPENAI_API_KEY for the LLM judge." >&2
    exit 1
fi

viper-eval \
    --model "${MODEL}" \
    --api-base "https://api.anthropic.com/v1/openai" \
    --api-key "${ANTHROPIC_API_KEY}" \
    --judge-api-key "${OPENAI_API_KEY}" \
    --output ./eval_logs

echo
echo "✅ Done. results.json written to eval_logs/${MODEL//\//--}/<timestamp>/"
