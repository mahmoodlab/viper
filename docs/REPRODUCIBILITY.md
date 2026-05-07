# Reproducing the paper

This document maps every row of the paper's main results table
(Table 1, paper §3) to the exact `viper-eval` command that produces it.
Every command writes a `results.json` with the same numbers reported in the
paper to within judge-model noise (≤ 0.5 percentage point). All numbers
below are paper-quoted as `MCQ / KPrim / Free-Text / Overall (%)` on
n = 1,251 questions, with MCQ scores aggregated over 5 cyclic-shift
permutations.

## Setup

```bash
pip install viper-bench
export OPENAI_API_KEY=sk-...   # used for both the model under test (when applicable) and the judge
```

The default judge model is **gpt-5.4** to match the paper. To override:

```bash
viper-eval --model <X> --judge-model <Y>
```

## Closed-weight commercial models (paper-reported scores)

```bash
# GPT-5.4   58.5 / 54.3 / 55.1 / 56.0
viper-eval --model gpt-5.4

# GPT-5.4-mini   48.0 / 45.9 / 50.0 / 48.0
viper-eval --model gpt-5.4-mini

# GPT-5.4-nano   24.4 / 30.1 / 27.7 / 27.4
viper-eval --model gpt-5.4-nano

# Claude Sonnet 4.6   54.6 / 47.1 / 42.8 / 48.2
viper-eval \
    --model claude-sonnet-4-6 \
    --api-base https://api.anthropic.com/v1/openai \
    --api-key "$ANTHROPIC_API_KEY"

# Gemini 2.5 Flash   52.8 / 45.0 / 25.2 / 41.0
viper-eval \
    --model gemini-2.5-flash \
    --api-base https://generativelanguage.googleapis.com/v1beta/openai \
    --api-key "$GEMINI_API_KEY"
```

## Open-weight general-purpose models (self-served)

Serve the model in a separate terminal, then run `viper-eval` against
`http://localhost:8000/v1`:

```bash
# Qwen 3.5-27B   60.0 / 46.6 / 50.6 / 52.4
vllm serve Qwen/Qwen3.5-27B --port 8000 --gpu-memory-utilization 0.92
viper-eval --model Qwen/Qwen3.5-27B --api-base http://localhost:8000/v1 --api-key dummy

# Gemma 4-31B-it   60.7 / 54.1 / 48.3 / 54.4
vllm serve google/gemma-4-31b-it --port 8000
viper-eval --model google/gemma-4-31b-it --api-base http://localhost:8000/v1 --api-key dummy
```

## Open-weight pathology-specialized models

```bash
# Patho-R1-7B   46.1 / 16.3 / 46.0 / 36.1
vllm serve jamesdolezal/patho-r1-7b --port 8000
viper-eval --model jamesdolezal/patho-r1-7b --api-base http://localhost:8000/v1 --api-key dummy

# Patho-R1-3B   39.5 / 12.9 / 36.1 / 29.5
vllm serve jamesdolezal/patho-r1-3b --port 8000
viper-eval --model jamesdolezal/patho-r1-3b --api-base http://localhost:8000/v1 --api-key dummy

# MedGemma-4B   29.9 / 18.4 / 26.0 / 24.8
vllm serve google/medgemma-4b-it --port 8000
viper-eval --model google/medgemma-4b-it --api-base http://localhost:8000/v1 --api-key dummy

# Quilt-LLaVA   27.5 / 2.1 / 28.0 / 19.2
vllm serve wisdomik/Quilt-Llava-v1.5-7b --port 8000
viper-eval --model wisdomik/Quilt-Llava-v1.5-7b --api-base http://localhost:8000/v1 --api-key dummy

# LLaVA-Med   17.0 / 6.6 / 24.8 / 16.2
vllm serve microsoft/llava-med-v1.5-mistral-7b --port 8000
viper-eval --model microsoft/llava-med-v1.5-mistral-7b --api-base http://localhost:8000/v1 --api-key dummy

# PathGen-LLaVA   18.7 / 28.4 / 37.9 / 28.3
vllm serve PathGen/PathGen-LLaVA --port 8000
viper-eval --model PathGen/PathGen-LLaVA --api-base http://localhost:8000/v1 --api-key dummy
```

## Limited-access models

`PathChat+` requires direct access from the original authors (paper score
58.7 / 41.5 / 52.7 / 51.0). The `ToxScribe (Qwen3.5)` and `ToxScribe (Gemma 4)`
weights are not publicly released; the paper's numbers (62.4 / 61.2 overall)
are reported alongside training methodology. If you serve any of these
behind an OpenAI-compatible endpoint, the same `viper-eval` invocation
applies.

## Image-ablation experiments (paper §3)

```bash
# black-image: replace every image with a solid black PNG
viper-eval --model gpt-5.4 --ablation black-image

# no-image: omit the image attachment entirely
viper-eval --model gpt-5.4 --ablation no-image

# random-image: swap each ROI for another VIPER ROI (deterministic seed)
viper-eval --model gpt-5.4 --ablation random-image
```

## Reading the results

Every run writes:

- `eval_logs/<model>/<timestamp>/results.json`: paper-aligned metrics plus
  full provenance config (judge model, prompt sha256, package version, git
  hash, ablation, MCQ rotations, dataset revision, run timestamp). This is
  what the paper figures consume.
- `eval_logs/<model>/<timestamp>/samples.jsonl`: one record per
  (question × rotation) trial, including the raw model response. Use this
  to debug or re-score with a different judge.

```bash
jq '{overall_score, mcq_accuracy, kprim_score, free_text_judge}' \
   eval_logs/gpt-5.4/*/results.json
```

## Re-scoring without re-inference

The judge prompt is cached by sha256, so swapping judges with the same model
predictions is essentially free:

```bash
viper-eval --rescore eval_logs/gpt-5.4/20260501_140000/ --judge-model gpt-5.6
```

(Note: `--rescore` will be wired in a follow-up release. Until then,
re-running with the same `--model` and a different `--judge-model` reuses
the model-output cache when present.)

## Running locally without internet

```bash
# Download the parquet once
huggingface-cli download MahmoodLab/viper viper.parquet --repo-type dataset \
    --local-dir ./viper-data

# Use the local file from then on
viper-eval --data ./viper-data/viper.parquet --model gpt-5.4
```
