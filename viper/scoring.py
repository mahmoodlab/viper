"""Scoring functions for VIPER.

Computes the four numbers reported in the paper (Table 1):

- ``mcq_accuracy``: mean over 5 cyclic-shift permutations of the 5-option order.
- ``kprim_score``: ETH half-point rule (4/4 -> 1.0, 3/4 -> 0.5, <=2/4 -> 0.0).
- ``free_text_rouge``: surface-form overlap with the reference answer (ROUGE-L F).
- ``free_text_judge``: two-axis 0.7*accuracy + 0.3*completeness, by an LLM judge.

The composite ``overall_score`` is the sample-count-weighted mean of the per-type
scores (MCQ + KPrim + free-text-judge), matching the paper.
"""

from __future__ import annotations

import json
import logging
import random as _random
import re
import string
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Text normalisation
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalise(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = (text or "").strip().lower().translate(_PUNCT_TABLE)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
#  MCQ extraction (MMMU-style cascade with optional LLM fallback)
# ---------------------------------------------------------------------------


def _all_choice_letters(n: int) -> list[str]:
    return [chr(65 + i) for i in range(n)]


def parse_mcq_response(response: str, choices: list[str]) -> str | None:
    """Extract the chosen option letter from a free-form model response.

    Cascade (in order): bracketed letters ``(A)``, space-padded ``" A "``,
    period-suffixed ``A.``, content match against the choice text. When
    multiple candidates match, the *last* occurrence wins, so the parser
    captures the model's final answer after any chain-of-thought.

    This function is pure-regex / pure-text. When extraction is genuinely
    ambiguous it returns ``None`` so the caller can decide whether to fall
    back to an LLM extractor or pick a random letter.

    Args:
        response: Free-form text emitted by the model.
        choices: Ordered list of MCQ option strings.

    Returns:
        Uppercase letter (``A``..``chr(64+len(choices))``) or ``None``.
    """
    if not choices:
        return None

    letters = _all_choice_letters(len(choices))
    index2ans = {chr(65 + i): c for i, c in enumerate(choices)}

    clean = (response or "").strip().strip(",.!?;:'\"")
    clean = " " + clean + " "

    candidates: list[str] = []
    ans_with_brack = False
    index_ans = True

    # Step 1: (A), (B), …
    for letter in letters:
        if f"({letter})" in clean or f"({letter.lower()})" in clean:
            candidates.append(letter)
            ans_with_brack = True

    # Step 2: " A ", " B ", …
    if not candidates:
        for letter in letters:
            if f" {letter} " in clean or f" {letter.lower()} " in clean:
                candidates.append(letter)

    # Step 3: A., B., …
    if not candidates:
        for letter in letters:
            if f"{letter}." in clean or f"{letter.lower()}." in clean:
                candidates.append(letter)

    # Step 4: content match (response > 5 tokens)
    if not candidates and len((response or "").split()) > 5:
        for letter, ans_text in index2ans.items():
            if ans_text and ans_text.lower() in (response or "").lower():
                candidates.append(letter)
                index_ans = False

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Disambiguate: pick the LAST occurrence in the response.
    starts: list[int] = []
    for can in candidates:
        if index_ans:
            if ans_with_brack:
                idx = max(clean.rfind(f"({can})"), clean.rfind(f"({can.lower()})"))
            else:
                idx = max(clean.rfind(f" {can} "), clean.rfind(f" {can.lower()} "))
        else:
            idx = (response or "").lower().rfind(index2ans[can].lower())
        starts.append(idx)
    return candidates[starts.index(max(starts))]


def random_letter(choices: list[str], seed: int | None = None) -> str:
    """Random fallback for MCQ; exposed so callers can use a fixed seed in tests."""
    rng = _random.Random(seed) if seed is not None else _random
    return rng.choice(_all_choice_letters(len(choices)))


# ---------------------------------------------------------------------------
#  KPrim per-statement extraction + ETH half-point scoring
# ---------------------------------------------------------------------------

_TF_TRUE_WORDS = {"true", "t", "correct", "yes", "affirmative"}
_TF_FALSE_WORDS = {"false", "f", "incorrect", "no", "negative"}

_TF_AFFIRMATIVE_RE = re.compile(r"\b(is|are|appears|consistent|matches|describes|true|correct)\b")
_TF_NEGATIVE_RE = re.compile(
    r"\b(is\s+not|are\s+not|isn'?t|aren'?t|does\s+not|doesn'?t|cannot|never|"
    r"inconsistent|contradicts|false|incorrect)\b"
)


def parse_tf_response(response: str) -> bool | None:
    """Parse a True/False answer from a KPrim per-statement model response.

    Pure regex / heuristic. Returns ``True``, ``False``, or ``None`` if the
    model output is genuinely ambiguous (counts as incorrect under
    ``score_kprim``). The caller may invoke an LLM extractor on the ``None``
    case.
    """
    text = (response or "").strip()
    if not text:
        return None

    first_line = text.splitlines()[0].strip()
    first_line_lower = first_line.lower()
    first_token = re.split(r"[,.:;\-—\s]", first_line, maxsplit=1)[0].strip().lower()

    if first_token in _TF_TRUE_WORDS:
        return True
    if first_token in _TF_FALSE_WORDS:
        return False

    for w in ("true", "correct", "yes"):
        if re.search(rf"\b{w}\b", first_line_lower) and not re.search(
            rf"\b(not|isn'?t|doesn'?t|cannot|never)\s+{w}\b", first_line_lower
        ):
            return True
    for w in ("false", "incorrect", "no"):
        if re.search(rf"\b{w}\b", first_line_lower):
            return False

    full_lower = text.lower()
    neg = _TF_NEGATIVE_RE.search(full_lower)
    aff = _TF_AFFIRMATIVE_RE.search(full_lower)
    if neg and not aff:
        return False
    if aff and not neg:
        return True
    if neg and aff:
        # Whichever marker appears FIRST is the model's main claim.
        return neg.start() > aff.start()

    return None


def score_kprim(predictions: list[bool | None], ground_truth: list[bool]) -> float:
    """ETH Zürich half-point rule (Krebs 1997, Kanzow et al. 2018).

    4/4 correct → 1.0, 3/4 correct → 0.5, ≤2/4 correct → 0.0.
    Unparsed predictions count as incorrect.
    """
    n = len(ground_truth)
    correct = sum(
        1 for p, g in zip(predictions, ground_truth, strict=False) if p is not None and p == g
    )
    if correct == n:
        return 1.0
    if correct == n - 1:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
#  Free-text ROUGE-L
# ---------------------------------------------------------------------------


def rouge_l(predictions: list[str], references: list[str]) -> float:
    """Mean ROUGE-L F-score over (pred, ref) pairs."""
    if not predictions:
        return 0.0
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores: list[float] = []
        for pred, ref in zip(predictions, references, strict=False):
            if not pred or not ref:
                scores.append(0.0)
                continue
            scores.append(scorer.score(ref, pred[:2048])["rougeL"].fmeasure)
        return float(sum(scores) / len(scores))
    except ImportError:  # pragma: no cover; rouge-score is a hard dep
        logger.warning("rouge-score not installed; falling back to token overlap")
        return _token_overlap_f1(predictions, references)


def _token_overlap_f1(predictions: list[str], references: list[str]) -> float:
    scores: list[float] = []
    for pred, ref in zip(predictions, references, strict=False):
        pt = set(_normalise(pred).split())
        rt = set(_normalise(ref).split())
        if not pt or not rt:
            scores.append(0.0)
            continue
        common = pt & rt
        p = len(common) / len(pt)
        r = len(common) / len(rt)
        scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
#  MCQ permutation aggregator (paper §2: mean over 5 cyclic shifts)
# ---------------------------------------------------------------------------


def aggregate_mcq_with_rotations(per_rotation_scores: dict[str, list[float]]) -> float:
    """Per-question mean over its rotations, then mean over questions.

    Args:
        per_rotation_scores: mapping ``base_question_id -> [score_rotation_0,
            score_rotation_1, ...]``. Each score is in {0.0, 1.0}.
    """
    if not per_rotation_scores:
        return 0.0
    per_question_means = [
        sum(scores) / len(scores) for scores in per_rotation_scores.values() if scores
    ]
    return float(sum(per_question_means) / len(per_question_means)) if per_question_means else 0.0


# ---------------------------------------------------------------------------
#  Top-level ``score`` entry point
# ---------------------------------------------------------------------------


def score(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute paper-aligned VIPER metrics from a list of evaluated samples.

    Each sample dict must contain at least:
        question_type: "mcq" | "kprim" | "free_text"
        ref          : ground-truth answer (string for MCQ/free_text, JSON-list for kprim)
        pred         : model prediction (or list of per-statement TFs for kprim)
        score        : per-sample numeric score already computed by the runner

    For MCQ samples, ``base_question_id`` and ``rotation_id`` are required to
    aggregate over the 5 cyclic shifts. For KPrim samples, ``score`` is the ETH
    half-point value. For free-text samples, ``judge_score`` (and optionally
    ``rouge_score``) are read directly.

    Returns the canonical ``results.json`` payload (without the ``config`` block).
    """
    by_type: dict[str, list[dict[str, Any]]] = {"mcq": [], "kprim": [], "free_text": []}
    for s in samples:
        qt = s.get("question_type")
        if qt in by_type:
            by_type[qt].append(s)

    metrics: dict[str, Any] = {
        "num_samples": {qt: len(items) for qt, items in by_type.items()},
    }

    # MCQ: group by base_question_id, mean over rotations, then mean over base questions
    if by_type["mcq"]:
        groups: dict[str, list[float]] = {}
        for s in by_type["mcq"]:
            bqid = s.get("base_question_id") or s.get("image_id") or s.get("question")
            groups.setdefault(bqid, []).append(float(s.get("score", 0.0)))
        metrics["mcq_accuracy"] = aggregate_mcq_with_rotations(groups)
        metrics["mcq_num_base_questions"] = len(groups)
        metrics["mcq_num_rotations"] = sum(len(v) for v in groups.values())

    # KPrim: mean of ETH half-point scores
    if by_type["kprim"]:
        kprim_scores = [float(s.get("score", 0.0)) for s in by_type["kprim"]]
        metrics["kprim_score"] = sum(kprim_scores) / len(kprim_scores)

    # Free-text: judge mean (preferred) and ROUGE-L mean (always reported).
    if by_type["free_text"]:
        ft = by_type["free_text"]
        preds = [str(s.get("pred", "")) for s in ft]
        refs = [str(s.get("ref", "")) for s in ft]
        metrics["free_text_rouge"] = rouge_l(preds, refs)
        judge_scores = [s.get("judge_score") for s in ft if s.get("judge_score") is not None]
        if judge_scores:
            metrics["free_text_judge"] = float(sum(judge_scores) / len(judge_scores))

    metrics["overall_score"] = _overall_from_metrics(metrics, by_type)
    return metrics


def _overall_from_metrics(
    metrics: dict[str, Any],
    by_type: dict[str, list[dict[str, Any]]],
) -> float:
    """Sample-count-weighted mean across the three question types.

    Matches the paper convention: free-text contribution comes from the LLM
    judge if it ran, else falls back to ROUGE-L so the metric stays comparable.
    """
    parts: list[tuple[float, int]] = []
    if metrics.get("mcq_accuracy") is not None and by_type["mcq"]:
        parts.append((metrics["mcq_accuracy"], len(by_type["mcq"])))
    if metrics.get("kprim_score") is not None and by_type["kprim"]:
        parts.append((metrics["kprim_score"], len(by_type["kprim"])))
    if by_type["free_text"]:
        ft_score = metrics.get("free_text_judge", metrics.get("free_text_rouge", 0.0))
        parts.append((ft_score, len(by_type["free_text"])))
    if not parts:
        return 0.0
    total_n = sum(n for _, n in parts)
    return sum(s * n for s, n in parts) / total_n


# ---------------------------------------------------------------------------
#  Per-sample helpers used by the runner
# ---------------------------------------------------------------------------


def parse_kprim_ground_truth(answer: Any) -> list[bool]:
    """Decode the KPrim ground-truth from its dataset representation.

    The dataset stores KPrim answers as a JSON-encoded list of booleans, e.g.
    ``"[true, false, true, false]"``. Some legacy paths may pass a Python list
    directly; both are accepted.
    """
    if isinstance(answer, list):
        return [bool(x) for x in answer]
    if isinstance(answer, str):
        return [bool(x) for x in json.loads(answer)]
    raise ValueError(f"Unsupported KPrim ground-truth type: {type(answer)!r}")


def cyclic_shift_choices(choices: list[str], answer_letter: str, k: int) -> tuple[list[str], str]:
    """Return ``(rotated_choices, new_answer_letter)`` for cyclic shift ``k``.

    Rotation 0 is the canonical order. Rotation ``k`` shifts every option ``k``
    positions to the right (mod ``len(choices)``). The correct answer letter
    is updated accordingly so downstream scoring matches the paper.
    """
    n = len(choices)
    if n == 0:
        return choices, answer_letter
    k = k % n
    if k == 0:
        return list(choices), answer_letter
    rotated = choices[-k:] + choices[:-k]
    gold_idx = ord(answer_letter.upper()) - 65
    new_idx = (gold_idx + k) % n
    return rotated, chr(65 + new_idx)


def majority_vote_letter(letters: list[str]) -> str:
    """Return the most common letter (deterministic tie-break: lexicographically smallest)."""
    counts = Counter(letters)
    top = max(counts.values())
    winners = sorted([letter for letter, c in counts.items() if c == top])
    return winners[0]
