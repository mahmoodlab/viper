#!/usr/bin/env bash
# Full VIPER evaluation against an OpenAI model.
# Default: gpt-4o-mini (~$5, ~25 min). Override via the MODEL env var.
set -euo pipefail

MODEL="${MODEL:-gpt-4o-mini}"
JUDGE="${JUDGE:-gpt-5.4}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Set OPENAI_API_KEY first." >&2
    exit 1
fi

viper-eval \
    --model "${MODEL}" \
    --judge-model "${JUDGE}" \
    --output ./eval_logs

echo
echo "✅ Done. results.json written to eval_logs/${MODEL//\//--}/<timestamp>/"
