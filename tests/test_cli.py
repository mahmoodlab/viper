"""End-to-end CLI test with a mocked OpenAI client.

Verifies that ``viper-eval`` parses arguments, drives the inference loop, and
writes a well-formed ``results.json`` without ever hitting a real API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _scripted_chat_response(prompt_text: str) -> str:
    """Deterministic fake responses based on the prompt text.

    - MCQ prompts → respond with "(A)" so MCQ scoring sees a valid letter.
    - KPrim prompts → respond with "True" for every per-statement query.
    - Free-text prompts → respond with a stock answer.
    - Judge prompts → respond with a JSON object the runner can parse.
    """
    if "Answer with the letter" in prompt_text:
        return "The answer is (A)."
    if "Is the following statement true or false" in prompt_text:
        return "True."
    if "Answer the question accurately" in prompt_text:
        return "There is centrilobular hepatocellular necrosis."
    if "veterinary pathologist evaluating answers" in prompt_text:
        return '{"diagnostic_accuracy": 7, "completeness": 6}'
    return "OK"


@pytest.fixture
def mocked_openai(monkeypatch):
    """Patch ``viper.inference._Client.chat`` to a deterministic stub."""
    from viper import inference

    async def fake_chat(self, *, model, messages, temperature, max_tokens, response_format=None):
        # Pull the prompt text out of the message payload.
        if isinstance(messages[0]["content"], list):
            prompt_text = messages[0]["content"][0]["text"]
        else:
            prompt_text = messages[0]["content"]
        return _scripted_chat_response(prompt_text)

    monkeypatch.setattr(inference._Client, "chat", fake_chat)
    return None


def test_cli_smoke(mini_dataset, tmp_path: Path, mocked_openai, monkeypatch):
    """``viper-eval`` runs end-to-end on the mini fixture and writes results.json."""
    # Have load_viper return our in-memory fixture instead of hitting the Hub.
    from viper import cli as cli_mod

    monkeypatch.setattr(cli_mod, "load_viper", lambda **kwargs: mini_dataset)

    out_root = tmp_path / "eval_logs"
    rc = cli_mod.main(
        [
            "--model",
            "fake-model",
            "--api-key",
            "EMPTY",
            "--output",
            str(out_root),
            "--mcq-rotations",
            "2",
            "--quiet",
            "--judge-cache",
            str(tmp_path / "_judge_cache"),
        ]
    )
    assert rc == 0

    runs = sorted((out_root / "fake-model").glob("*"))
    assert len(runs) == 1
    run = runs[0]
    results = json.loads((run / "results.json").read_text())

    assert "overall_score" in results
    assert "config" in results
    assert results["config"]["model"] == "fake-model"
    assert results["config"]["mcq_rotations"] == 2
    # Each base MCQ produces ``--mcq-rotations`` trials in samples.jsonl, so the
    # ``num_samples`` reflects per-trial counts: 2 base MCQs × 2 rotations = 4.
    assert results["num_samples"] == {"mcq": 4, "kprim": 2, "free_text": 2}
    # Stub returns "(A)" for MCQ. Across 5-option rotations the gold letter
    # shifts, so the per-question hit rate isn't deterministic; just verify
    # the metric is in [0, 1].
    assert 0.0 <= results["overall_score"] <= 1.0
    assert 0.0 <= results["mcq_accuracy"] <= 1.0
    # KPrim stub answers True for every statement. Both fixture KPrim items
    # have exactly 2/4 True ground truths → ETH rule maps to 0.0.
    assert results["kprim_score"] == pytest.approx(0.0)
    # Free-text judge with stub: 0.7*7 + 0.3*6 = 6.7 → 0.67
    assert results["free_text_judge"] == pytest.approx(0.67, abs=1e-2)


def test_cli_help_renders(capsys):
    from viper.cli import _build_parser

    parser = _build_parser()
    help_text = parser.format_help()
    assert "viper-eval" in help_text
    assert "--mcq-rotations" in help_text
    assert "--ablation" in help_text
