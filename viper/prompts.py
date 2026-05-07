"""Prompt templates for VIPER evaluation.

Single source of truth for all prompts that touch the model under test or the
LLM judge. Changing a prompt here changes its sha256 fingerprint, which is
serialized into ``results.json`` for provenance.

The free-text judge prompt was calibrated against pathologist-graded ratings
and is the same one used to produce the numbers in the paper. Do not redesign
it without recalibration.
"""

from __future__ import annotations

import hashlib

# ---------------------------------------------------------------------------
#  Question prompts (presented to the model under test)
# ---------------------------------------------------------------------------

MCQ_PROMPT_TEMPLATE = (
    "{question}\n\n{options}\n\nAnswer with the letter of the correct option only (e.g. A)."
)

KPRIM_SINGLE_PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Is the following statement true or false?\n\n"
    '"{statement}"\n\n'
    "Answer True or False."
)

FREE_TEXT_PROMPT_TEMPLATE = "{question}\n\nAnswer the question accurately and completely."


# ---------------------------------------------------------------------------
#  Answer-extraction prompts (LLM fallback when regex fails)
# ---------------------------------------------------------------------------

MCQ_EXTRACT_PROMPT = (
    "Given the following question with answer options and a model's response, "
    "determine which option letter (e.g., A, B, C, D, E) the model selected.\n\n"
    "Options:\n{options}\n\n"
    'Model\'s response: "{response}"\n\n'
    "Return ONLY the single letter of the selected option."
)

TF_EXTRACT_PROMPT = (
    "Does the following response affirm a statement (True) or deny it (False)?\n\n"
    'Response: "{response}"\n\n'
    'Return ONLY the word "True" or "False".'
)


# ---------------------------------------------------------------------------
#  Free-text LLM-as-judge prompt (calibrated against pathologist scores)
# ---------------------------------------------------------------------------

FREE_TEXT_JUDGE_PROMPT = (
    "You are an expert veterinary pathologist evaluating answers from a "
    "toxicologic histopathology visual question-answering system.\n\n"
    "Question: {question}\n"
    "Reference (gold-standard) answer: {ref}\n"
    "{synonyms_line}"
    "{rubric_line}"
    "Model prediction: {pred}\n\n"
    "Evaluate in two steps:\n\n"
    "**Step 1 — Diagnostic Accuracy (0-10):**\n"
    "Does the prediction identify the SAME finding, diagnosis, or "
    "pathologic process as the reference?\n"
    "  10 = exact match (synonyms acceptable: 'hyperemia'='congestion', "
    "'urothelium'='transitional epithelium')\n"
    "   8-9 = correct diagnosis, minor imprecision in terminology or detail\n"
    "   5-7 = partially correct (right morphologic category but wrong "
    "specific diagnosis, e.g. says 'vacuolation' when reference says "
    "'glycogen accumulation')\n"
    "   2-4 = wrong diagnosis but at least recognizes the correct tissue "
    "region, organ, or that an abnormality exists\n"
    "   0-1 = entirely wrong finding, different pathologic process, or "
    "fabricated\n\n"
    "Critical: naming the WRONG pathologic process is a major error even "
    "if the language sounds professional (e.g. 'inflammation' ≠ "
    "'extramedullary hematopoiesis'; 'necrosis' ≠ 'vacuolar degeneration'; "
    "'hypertrophy' ≠ 'degeneration').\n\n"
    "**Step 2 — Completeness & Quality (0-10):**\n"
    "Given whatever diagnostic accuracy was achieved, how complete and "
    "reliable is the answer?\n"
    "  10 = all key points from the reference are covered, no fabrication\n"
    "   7-9 = most key points present, minor omissions\n"
    "   4-6 = some key points present but significant gaps or some "
    "hallucinated/fabricated claims\n"
    "   0-3 = major omissions or substantial fabrication\n\n"
    "NOTE: if Diagnostic Accuracy is 0-1, Completeness should also be "
    "very low (0-2) since the wrong diagnosis means the details are moot.\n\n"
    "Respond with ONLY a JSON object:\n"
    '{{"diagnostic_accuracy": <int>, "completeness": <int>}}'
)

# Composite weighting: paper §2, 0.7 * diagnostic_accuracy + 0.3 * completeness.
JUDGE_ACCURACY_WEIGHT = 0.7
JUDGE_COMPLETENESS_WEIGHT = 0.3


def judge_prompt_fingerprint() -> str:
    """SHA-256 fingerprint of the free-text judge prompt; serialized into results.json."""
    return hashlib.sha256(FREE_TEXT_JUDGE_PROMPT.encode("utf-8")).hexdigest()[:12]


def format_mcq_prompt(question: str, choices: list[str]) -> str:
    """Render the MCQ prompt for a single question with its options."""
    options = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return MCQ_PROMPT_TEMPLATE.format(question=question, options=options)


def format_kprim_prompt(question: str, statement: str) -> str:
    """Render the per-statement KPrim prompt (decomposed inference)."""
    return KPRIM_SINGLE_PROMPT_TEMPLATE.format(question=question, statement=statement)


def format_free_text_prompt(question: str) -> str:
    """Render the free-text prompt."""
    return FREE_TEXT_PROMPT_TEMPLATE.format(question=question)


def format_judge_prompt(
    question: str,
    ref: str,
    pred: str,
    synonyms: list[str] | None = None,
    rubric: str | None = None,
) -> str:
    """Render the LLM-judge prompt for a single free-text answer."""
    synonyms_line = ""
    if synonyms:
        syn_str = ", ".join(str(s) for s in synonyms)
        synonyms_line = f"Accepted synonyms: {syn_str}\n"
    rubric_line = ""
    if rubric and str(rubric).strip():
        rubric_line = f"Scoring rubric: {rubric}\n"
    return FREE_TEXT_JUDGE_PROMPT.format(
        question=question,
        ref=ref,
        pred=pred,
        synonyms_line=synonyms_line,
        rubric_line=rubric_line,
    )
