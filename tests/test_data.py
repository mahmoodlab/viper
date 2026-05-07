"""Schema and integrity tests for the VIPER dataset loader."""

from __future__ import annotations

from PIL import Image

from viper.data import (
    EXPECTED_COLUMNS,
    EXPECTED_QUESTION_TYPES,
    EXPECTED_ROW_COUNT,
)


def test_mini_dataset_has_canonical_columns(mini_dataset):
    cols = set(mini_dataset.column_names)
    assert EXPECTED_COLUMNS.issubset(cols), f"missing: {EXPECTED_COLUMNS - cols}"


def test_mini_dataset_question_types_are_canonical(mini_dataset):
    seen = set(mini_dataset.unique("question_type"))
    assert seen.issubset(EXPECTED_QUESTION_TYPES)


def test_mini_dataset_images_decode(mini_dataset):
    sample = mini_dataset[0]
    img = sample["image"]
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_mini_dataset_kprim_answers_are_lists_of_4(mini_dataset):
    """Every KPrim row stores its ground truth as a JSON list of 4 booleans."""
    import json

    for row in mini_dataset:
        if row["question_type"] != "kprim":
            continue
        gt = json.loads(row["answer"])
        assert isinstance(gt, list)
        assert len(gt) == 4
        assert all(isinstance(x, bool) for x in gt)
        assert len(row["choices"]) == 4


def test_mini_dataset_mcq_has_5_choices(mini_dataset):
    for row in mini_dataset:
        if row["question_type"] != "mcq":
            continue
        assert len(row["choices"]) == 5
        assert row["answer"] in {"A", "B", "C", "D", "E"}


def test_mini_dataset_free_text_has_no_choices(mini_dataset):
    for row in mini_dataset:
        if row["question_type"] != "free_text":
            continue
        assert row["choices"] == []
        assert row["answer"]


def test_no_internal_terms_in_mini_dataset(mini_dataset):
    """The mini fixture must itself be free of contaminated terms."""
    import re

    deny_patterns = [
        re.compile(p, re.IGNORECASE)
        for p in (r"\btremont\b", r"\bjnj\b", r"\bsealsync\b", r"\bgs://")
    ]
    text_cols = [
        "question",
        "answer",
        "choices",
        "synonyms",
        "scoring_rubric",
        "organ",
        "category",
        "image_id",
    ]
    for row in mini_dataset:
        for col in text_cols:
            value = row[col]
            if value is None:
                continue
            text = " ".join(value) if isinstance(value, list) else str(value)
            for pat in deny_patterns:
                assert not pat.search(text), f"{col}: {text!r} matched {pat.pattern}"


def test_expected_row_count_constant():
    """Document the published count so a future change is intentional."""
    assert EXPECTED_ROW_COUNT == 1251
