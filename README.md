# VIPER

> An expert-curated benchmark for vision-language models in veterinary pathology.

[![Dataset](https://img.shields.io/badge/Dataset-MahmoodLab%2Fviper-yellow.svg)](https://huggingface.co/datasets/MahmoodLab/viper)
[![License](https://img.shields.io/badge/license-CC--BY--NC--ND%204.0-lightgrey.svg)](LICENSE)
[![CI](https://github.com/mahmoodlab/viper/actions/workflows/ci.yml/badge.svg)](https://github.com/mahmoodlab/viper/actions/workflows/ci.yml)

VIPER is the first expert-curated benchmark for evaluating vision-language models on rodent toxicologic pathology, a domain that anchors preclinical drug-safety assessment but has been almost entirely absent from the pathology-VLM evaluation ecosystem. The benchmark contains **1,251 questions over 419 H&E-stained rat histology images**, spans **9 organs**, and was authored and validated by board-certified veterinary pathologists (ECVP). 

This repository is the official evaluation suite. A single command runs any vision-language model that speaks the OpenAI Chat Completions API (GPT, Claude, Gemini, vLLM, SGLang, llama.cpp, and friends) against the full benchmark and reproduces the paper's scoring exactly.

This project was developed by the [Mahmood Lab](https://faisal.ai/) at Harvard Medical School and Brigham and Women's Hospital, with veterinary pathology partners at COMPATH (University of Bern), UC Davis, the University of Augsburg, TU Dresden, UT MD Anderson Cancer Center, and the University of Lausanne. Funded in part by NIH NIGMS R35GM138216.

> Weishaupt, de Brot, Asin, Grau-Roma, Reitsam, Song, Bang, Le, Kather, Mahmood, Jaume.
> **VIPER: An Expert-Curated Benchmark for Vision-Language Models in Veterinary Pathology.**
> *NeurIPS Datasets and Benchmarks*, 2026.

### Key Features

- **Veterinary-pathology specific.** 1,251 questions over 419 H&E rat ROIs from 9 organs, all authored by board-certified veterinary pathologists (ECVP).
- **Visually grounded.** Every question was anchored in visible morphology and adversarially filtered against image-free guessability.
- **Three question formats.** Multiple-choice (MCQ, 5 options × 5 cyclic rotations), KPrim (4 true/false statements), and free-text with rubric-guided LLM-judge scoring.
- **Run anywhere.** One CLI, one OpenAI-compatible endpoint, no PyTorch installation needed.
- **Paper-aligned scoring.** MCQ accuracy is the mean over 5 cyclic-shift permutations of the answer order; KPrim uses the ETH half-point rule; free-text uses the calibrated 0.7·accuracy + 0.3·completeness LLM judge.
- **Reproducibility.** Every `results.json` carries a SHA-256 fingerprint of the judge prompt, the package version, and the git hash of the eval suite.

### 1. Installation

```bash
git clone https://github.com/mahmoodlab/viper.git
cd viper
uv sync --all-extras
```

VIPER has zero GPU dependencies. Every model is reached over the OpenAI Chat Completions API. To evaluate an open-weight model, serve it with [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang) and pass the local URL via `--api-base`.

### 2. Quickstart

```bash
export OPENAI_API_KEY=sk-...

# 30-second smoke test on 5 questions
viper-eval --model gpt-5.4-mini --limit 5

# Full benchmark, 1,251 questions, 5 MCQ rotations
viper-eval --model gpt-5.4-mini
```

Results land at `eval_logs/<model>/<timestamp>/`:

```
eval_logs/gpt-5.4-mini/<timestamp>/
├── results.json   # paper-aligned metrics + full provenance config
└── samples.jsonl  # one record per (question × rotation) trial
```

```bash
jq '{overall_score, mcq_accuracy, kprim_score, free_text_judge}' \
    eval_logs/gpt-5.4-mini/*/results.json
```

### 3. Evaluating your own model

Step 1. Serve your model behind any OpenAI-compatible endpoint:

```bash
vllm serve my-org/my-vlm --port 8000
```

Step 2. Point `viper-eval` at it:

```bash
viper-eval \
    --model my-org/my-vlm \
    --api-base http://localhost:8000/v1 \
    --api-key dummy
```

Step 3. Read `results.json`. The numbers are directly comparable to the table below.

For working end-to-end examples, see [`examples/`](examples).

### Reproducing the paper

| Model | Domain | MCQ | KPrim | Free-Text | Overall |
| :-- | :-- | --: | --: | --: | --: |
| **ToxScribe (Qwen3.5)** | Veterinary | **67.1** | 61.8 | **58.3** | **62.4** |
| **ToxScribe (Gemma 4)** | Veterinary | 65.2 | **64.1** | 54.3 | 61.2 |
| GPT-5.4 | General | 58.5 | 54.3 | 55.1 | 56.0 |
| Gemma 4 | General | 60.7 | 54.1 | 48.3 | 54.4 |
| Qwen 3.5-27B | General | 60.0 | 46.6 | 50.6 | 52.4 |
| PathChat+ | Human path. | 58.7 | 41.5 | 52.7 | 51.0 |
| Claude Sonnet 4.6 | General | 54.6 | 47.1 | 42.8 | 48.2 |
| GPT-5.4-mini | General | 48.0 | 45.9 | 50.0 | 48.0 |
| Gemini 2.5 Flash | General | 52.8 | 45.0 | 25.2 | 41.0 |
| Patho-R1-7B | Human path. | 46.1 | 16.3 | 46.0 | 36.1 |
| Patho-R1-3B | Human path. | 39.5 | 12.9 | 36.1 | 29.5 |
| PathGen-LLaVA | Human path. | 18.7 | 28.4 | 37.9 | 28.3 |
| GPT-5.4-nano | General | 24.4 | 30.1 | 27.7 | 27.4 |
| MedGemma-4B | Human path. | 29.9 | 18.4 | 26.0 | 24.8 |
| Quilt-LLaVA | Human path. | 27.5 | 2.1 | 28.0 | 19.2 |
| LLaVA-Med | Human path. | 17.0 | 6.6 | 24.8 | 16.2 |

All numbers are mean accuracy (%) on n=1,251 questions. MCQ scores are means across 5 cyclic-shift rotations. The full table with 95% bootstrap confidence intervals is provided in the paper. 

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the exact `viper-eval` calls that reproduces each row.

### 4. The CLI

`viper-eval --help` shows the full surface. The options most users will reach for:

| Flag | Purpose |
| :-- | :-- |
| `--model NAME` | The model name sent in `/v1/chat/completions` requests. |
| `--api-base URL` | OpenAI-compatible endpoint. Defaults to api.openai.com. |
| `--api-key KEY` | API key. Defaults to `$OPENAI_API_KEY`. |
| `--data PATH` / `--hf-dataset NAME` | Use a local parquet, or pull from the Hub. |
| `--mcq-rotations N` | Cyclic shifts per MCQ. Paper default: 5. |
| `--ablation {none, black-image, no-image, random-image}` | Image ablation for paper §3 sanity checks. |
| `--limit N` | Smoke test on the first N samples. |
| `--judge-model NAME` | LLM judge for free-text scoring. Default: `gpt-5.4`. |
| `--output DIR` | Where to write results. Default: `eval_logs/`. |

### Dataset

VIPER is hosted on the Hugging Face Hub at [`MahmoodLab/viper`](https://huggingface.co/datasets/MahmoodLab/viper). Machine-readable metadata in [Croissant](https://mlcommons.org/working-groups/data/croissant/) format is in [`croissant.json`](croissant.json) and on the Hub. The full datasheet is in [`docs/DATASHEET.md`](docs/DATASHEET.md).

```python
from datasets import load_dataset

ds = load_dataset("MahmoodLab/viper")["test"]
sample = ds[0]
sample["image"]          # PIL.Image.Image (1024 × 1024 RGB)
sample["question_type"]  # "mcq" | "kprim" | "free_text"
sample["organ"]          # one of 9 organ slugs
sample["category"]       # one of 7 paper categories
```

| Column | Type |
| :-- | :-- |
| `image` | `Image` (1024 × 1024 H&E RGB ROI) |
| `image_id` | content-hashed identifier |
| `question`, `answer`, `choices` | the question + reference answer |
| `synonyms`, `scoring_rubric` | optional free-text grading aids |
| `organ`, `category`, `magnification`, `source` | metadata |

### Citation

```bibtex
@inproceedings{weishaupt2026viper,
  title     = {VIPER: An Expert-Curated Benchmark for Vision-Language Models in Veterinary Pathology},
  author    = {Weishaupt, Luca and de Brot, Simone and Asin, Javier and Grau-Roma, Lloren\c{c} and Reitsam, Nic and Song, Andrew H. and Bang, Dongmin and Le, Long Phi and Kather, Jakob Nikolas and Mahmood, Faisal and Jaume, Guillaume},
  year      = {2026}
}
```

### License

Code and data are released under [CC BY-NC-ND 4.0](LICENSE). TG-GATEs is released under CC BY-SA 2.1 JP and MMO under CC BY-NC 4.0. See [`docs/DATASHEET.md`](docs/DATASHEET.md) for the full licensing discussion.

### Contact

Questions, errata, or contributions: [issues](https://github.com/mahmoodlab/viper/issues), or email [Faisal Mahmood](mailto:faisalmahmood@bwh.harvard.edu) and [Guillaume Jaume](mailto:guillaume.jaume@unil.ch).
