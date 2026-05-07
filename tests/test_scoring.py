"""Unit tests for scoring functions."""

from __future__ import annotations

import pytest

from viper.scoring import (
    aggregate_mcq_with_rotations,
    cyclic_shift_choices,
    parse_kprim_ground_truth,
    parse_mcq_response,
    parse_tf_response,
    random_letter,
    rouge_l,
    score,
    score_kprim,
)

# ---------------------------------------------------------------------------
#  MCQ extraction
# ---------------------------------------------------------------------------


CHOICES = ["necrosis", "fibrosis", "inflammation", "normal", "steatosis"]


@pytest.mark.parametrize(
    "response,expected",
    [
        ("(A)", "A"),
        ("The answer is (B).", "B"),
        ("C.", "C"),
        ("My answer: D", "D"),
        ("I think the correct option is E here", "E"),
        ("After analysis, the histology shows necrosis", "A"),  # content match
        # Both "fibrosis" (B) and "inflammation" (C) appear; "last match wins" → C.
        ("The lesion shows fibrosis with mild inflammation", "C"),
        ("hard to say", None),
        ("", None),
    ],
)
def test_mcq_extraction(response: str, expected: str | None):
    assert parse_mcq_response(response, CHOICES) == expected


def test_mcq_extraction_lowercase_letters():
    assert parse_mcq_response("the answer is (c)", CHOICES) == "C"


def test_mcq_extraction_picks_last():
    """When multiple letters appear, the last occurrence (final answer) wins."""
    response = "I first thought (A) but actually it's (C)."
    assert parse_mcq_response(response, CHOICES) == "C"


def test_random_letter_in_range():
    for seed in range(20):
        letter = random_letter(CHOICES, seed=seed)
        assert letter in {"A", "B", "C", "D", "E"}


# ---------------------------------------------------------------------------
#  KPrim per-statement TF parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response,expected",
    [
        ("True.", True),
        ("true", True),
        ("Yes, this is consistent with hepatocellular necrosis", True),
        ("Correct.", True),
        ("False, the lesion is not present", False),
        ("False - it is not consistent with the diagnosis", False),
        ("No.", False),
        ("Incorrect.", False),
        ("The statement is correct.", True),
        ("The statement is not correct.", False),
        ("This is consistent with the finding.", True),
        ("This is not consistent with the finding.", False),
        ("", None),
        ("Maybe.", None),
    ],
)
def test_tf_parsing(response: str, expected: bool | None):
    assert parse_tf_response(response) == expected


# ---------------------------------------------------------------------------
#  KPrim ETH half-point scoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "predictions,gt,expected",
    [
        ([True, True, True, True], [True, True, True, True], 1.0),
        ([True, True, True, False], [True, True, True, True], 0.5),
        ([False, True, True, True], [True, True, True, True], 0.5),
        ([True, True, False, False], [True, True, True, True], 0.0),
        ([False, False, False, False], [True, True, True, True], 0.0),
        ([None, True, True, True], [True, True, True, True], 0.5),  # one unparsed
        ([None, None, None, None], [True, True, True, True], 0.0),
    ],
)
def test_kprim_eth_half_point(predictions: list[bool | None], gt: list[bool], expected: float):
    assert score_kprim(predictions, gt) == expected


def test_parse_kprim_ground_truth_json():
    assert parse_kprim_ground_truth("[true, false, true, false]") == [
        True,
        False,
        True,
        False,
    ]


def test_parse_kprim_ground_truth_list():
    assert parse_kprim_ground_truth([True, False, True, False]) == [
        True,
        False,
        True,
        False,
    ]


# ---------------------------------------------------------------------------
#  Cyclic shift (paper §2, 5 rotations per MCQ)
# ---------------------------------------------------------------------------


def test_cyclic_shift_zero_is_identity():
    choices = ["a", "b", "c", "d", "e"]
    rot, gold = cyclic_shift_choices(choices, "C", 0)
    assert rot == choices
    assert gold == "C"


def test_cyclic_shift_one_step():
    choices = ["a", "b", "c", "d", "e"]
    rot, gold = cyclic_shift_choices(choices, "C", 1)
    # Rotated: ['e', 'a', 'b', 'c', 'd']; 'c' was at idx 2 → now idx 3 → "D"
    assert rot == ["e", "a", "b", "c", "d"]
    assert gold == "D"


def test_cyclic_shift_two_step():
    choices = ["a", "b", "c", "d", "e"]
    rot, gold = cyclic_shift_choices(choices, "A", 2)
    assert rot == ["d", "e", "a", "b", "c"]
    assert gold == "C"


def test_cyclic_shift_full_cycle_preserves_correct_answer_set():
    """Across all 5 rotations, the gold letter should hit each position exactly once."""
    choices = ["a", "b", "c", "d", "e"]
    gold_letters = [cyclic_shift_choices(choices, "A", k)[1] for k in range(5)]
    assert sorted(gold_letters) == ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------------------
#  Aggregator (mean over rotations, then mean over base questions)
# ---------------------------------------------------------------------------


def test_aggregate_mcq_rotations_simple():
    groups = {"q1": [1.0, 1.0, 1.0, 1.0, 1.0]}  # 1.0
    assert aggregate_mcq_with_rotations(groups) == 1.0


def test_aggregate_mcq_rotations_partial():
    groups = {"q1": [1.0, 0.0, 1.0, 0.0, 1.0]}  # 0.6
    assert abs(aggregate_mcq_with_rotations(groups) - 0.6) < 1e-9


def test_aggregate_mcq_rotations_multi_question():
    groups = {
        "q1": [1.0, 1.0, 0.0, 0.0, 1.0],  # 0.6
        "q2": [1.0, 1.0, 1.0, 1.0, 1.0],  # 1.0
        "q3": [0.0, 0.0, 0.0, 0.0, 0.0],  # 0.0
    }
    expected = (0.6 + 1.0 + 0.0) / 3
    assert abs(aggregate_mcq_with_rotations(groups) - expected) < 1e-9


def test_aggregate_mcq_rotations_empty():
    assert aggregate_mcq_with_rotations({}) == 0.0


# ---------------------------------------------------------------------------
#  ROUGE-L
# ---------------------------------------------------------------------------


def test_rouge_l_identical():
    assert rouge_l(["the cat sat on the mat"], ["the cat sat on the mat"]) == pytest.approx(1.0)


def test_rouge_l_disjoint_low():
    val = rouge_l(["red blue green"], ["zebra hippo whale"])
    assert val < 0.2


def test_rouge_l_partial():
    val = rouge_l(
        ["the cat sat on the mat"],
        ["the cat is on the mat"],
    )
    assert 0.5 < val < 1.0


# ---------------------------------------------------------------------------
#  End-to-end ``score()``
# ---------------------------------------------------------------------------


def test_score_end_to_end_minimal():
    samples = [
        {"question_type": "mcq", "base_question_id": "q1", "rotation_id": 0, "score": 1.0},
        {"question_type": "mcq", "base_question_id": "q1", "rotation_id": 1, "score": 0.0},
        {"question_type": "mcq", "base_question_id": "q2", "rotation_id": 0, "score": 1.0},
        {"question_type": "kprim", "score": 1.0, "pred": "[]", "ref": "[]"},
        {"question_type": "kprim", "score": 0.5, "pred": "[]", "ref": "[]"},
        {"question_type": "free_text", "pred": "x", "ref": "x", "judge_score": 0.8},
    ]
    metrics = score(samples)
    # MCQ: q1 mean = 0.5, q2 mean = 1.0; overall mean = 0.75
    assert metrics["mcq_accuracy"] == pytest.approx(0.75)
    # KPrim: mean(1.0, 0.5) = 0.75
    assert metrics["kprim_score"] == pytest.approx(0.75)
    # Free-text judge: 0.8
    assert metrics["free_text_judge"] == pytest.approx(0.8)
    # Overall: (0.75 * 3 + 0.75 * 2 + 0.8 * 1) / 6 ≈ 0.7583...
    expected = (0.75 * 3 + 0.75 * 2 + 0.8 * 1) / 6
    assert metrics["overall_score"] == pytest.approx(expected)


def test_score_handles_empty_input():
    metrics = score([])
    assert metrics["overall_score"] == 0.0
    assert metrics["num_samples"] == {"mcq": 0, "kprim": 0, "free_text": 0}


def test_score_groups_mcq_by_image_id_when_no_base_question_id():
    """When ``base_question_id`` is absent, the image_id is used as the group key."""
    samples = [
        {"question_type": "mcq", "image_id": "img1", "rotation_id": 0, "score": 1.0},
        {"question_type": "mcq", "image_id": "img1", "rotation_id": 1, "score": 0.0},
    ]
    metrics = score(samples)
    assert metrics["mcq_accuracy"] == pytest.approx(0.5)
    assert metrics["mcq_num_base_questions"] == 1
