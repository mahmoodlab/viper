#!/usr/bin/env bash
# 30-second smoke test against gpt-4o-mini on the first 5 VIPER samples.
# Confirms that your environment, API key, and model selection all work.
set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Set OPENAI_API_KEY first (e.g. via .env or your shell)." >&2
    exit 1
fi

viper-eval \
    --model gpt-4o-mini \
    --limit 5 \
    --output ./eval_logs

echo
echo "✅ Smoke test complete. Inspect ./eval_logs/gpt-4o-mini/*/results.json"
