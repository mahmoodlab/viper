"""VIPER, an expert-curated benchmark for vision-language models in veterinary pathology.

Public API:
    load_viper(...):       load the benchmark as a ``datasets.Dataset``.
    run_inference(...):    async inference loop against an OpenAI-compatible endpoint.
    score(...):            compute MCQ / KPrim / free-text scores from inference items.

CLI:
    viper-eval --model <name> [--api-base URL] [...]
"""

from __future__ import annotations

__version__ = "1.0.0"

from viper.data import load_viper
from viper.inference import InferenceItem, run_inference
from viper.scoring import score

__all__ = [
    "__version__",
    "InferenceItem",
    "load_viper",
    "run_inference",
    "score",
]
