"""Async inference loop for OpenAI-compatible endpoints.

Generates 5 cyclic-shift MCQ permutations on the fly so the published dataset
stays at 1,251 base rows, while evaluation matches the paper's reported
``MCQ accuracy = mean over 5 cyclic shifts``.

Usage (programmatic):

    from viper import load_viper, run_inference, score
    samples = run_inference(
        dataset=load_viper(),
        model="gpt-4o-mini",
        api_base=None,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    print(score(samples)["overall_score"])

The runner is intentionally synchronous on the outside and async on the
inside: callers don't need to know what an event loop is, but inference and
judging both run with high concurrency under the hood.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.auto import tqdm

from viper import ablation as _ablation
from viper.prompts import (
    JUDGE_ACCURACY_WEIGHT,
    JUDGE_COMPLETENESS_WEIGHT,
    MCQ_EXTRACT_PROMPT,
    TF_EXTRACT_PROMPT,
    format_free_text_prompt,
    format_judge_prompt,
    format_kprim_prompt,
    format_mcq_prompt,
)
from viper.scoring import (
    cyclic_shift_choices,
    parse_kprim_ground_truth,
    parse_mcq_response,
    parse_tf_response,
    random_letter,
    score_kprim,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Public dataclass returned by ``run_inference``
# ---------------------------------------------------------------------------


@dataclass
class InferenceItem:
    """Per-trial record produced by the inference loop, scored or not."""

    sample_index: int
    image_id: str
    question_type: str
    question: str
    ref: str
    pred: str
    score: float
    base_question_id: str | None = None
    rotation_id: int | None = None
    judge_score: float | None = None
    judge_details: dict[str, int] | None = None
    pred_bools: list[bool | None] | None = None
    raw_responses: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d


# ---------------------------------------------------------------------------
#  Image helpers
# ---------------------------------------------------------------------------


def _to_data_url(img: Image.Image) -> str:
    """Encode a PIL image as a ``data:`` URL for the OpenAI vision API."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _coerce_image(value: Any) -> Image.Image | None:
    """Accept a PIL image, a ``{"bytes": ...}`` dict (HF), or raw bytes."""
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and "bytes" in value:
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value))
    if isinstance(value, str):
        return Image.open(value)
    raise TypeError(f"Unsupported image type: {type(value)!r}")


# ---------------------------------------------------------------------------
#  OpenAI-compatible client wrapper (sync + async, with cache)
# ---------------------------------------------------------------------------


class _Client:
    """Thin wrapper around the OpenAI SDK with optional API base override.

    The same SDK is used for the model under test, the answer-extraction
    fallback, and the LLM judge. Different concurrency limits and judge
    caching are applied at the call sites.
    """

    def __init__(
        self,
        api_base: str | None,
        api_key: str | None,
        request_timeout: float = 120.0,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("openai>=1.40 is required for VIPER inference") from e
        self._async = AsyncOpenAI(
            api_key=api_key or "EMPTY",
            base_url=api_base,
            timeout=request_timeout,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: dict | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            resp = await self._async.chat.completions.create(**kwargs)
        except TypeError:
            # Some servers don't accept ``max_completion_tokens`` yet; retry with the legacy name.
            kwargs.pop("max_completion_tokens", None)
            kwargs["max_tokens"] = max_tokens
            resp = await self._async.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
#  Judge cache
# ---------------------------------------------------------------------------


def _judge_cache_path(prompt: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _load_judge_cache(prompt: str, cache_dir: Path) -> dict | None:
    path = _judge_cache_path(prompt, cache_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # pragma: no cover; corrupt cache = recompute
        return None


def _save_judge_cache(prompt: str, result: dict, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _judge_cache_path(prompt, cache_dir).write_text(json.dumps(result))


# ---------------------------------------------------------------------------
#  Per-question prompt builders
# ---------------------------------------------------------------------------


def _build_messages(prompt: str, image: Image.Image | None) -> list[dict[str, Any]]:
    if image is None:
        return [{"role": "user", "content": prompt}]
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _to_data_url(image)}},
            ],
        }
    ]


# ---------------------------------------------------------------------------
#  Top-level entry point
# ---------------------------------------------------------------------------


def run_inference(
    *,
    dataset: Iterable[dict[str, Any]],
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    judge_model: str = "gpt-5.4",
    judge_api_base: str | None = None,
    judge_api_key: str | None = None,
    extract_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    mcq_rotations: int = 5,
    ablation: str = "none",
    request_concurrency: int = 32,
    judge_concurrency: int = 50,
    judge_cache_dir: str | Path = ".judge_cache",
    progress: bool = True,
) -> list[InferenceItem]:
    """Run a model end-to-end against a VIPER-shaped iterable.

    See module docstring for the high-level workflow.
    """
    samples = list(dataset)
    cache_dir = Path(judge_cache_dir)

    client = _Client(api_base=api_base, api_key=api_key)
    extract_client = client  # extraction shares the model-under-test endpoint by default
    judge_client = (
        client
        if (judge_api_base is None and judge_api_key is None)
        else _Client(api_base=judge_api_base, api_key=judge_api_key)
    )

    image_pool: list[Image.Image] = []
    if ablation == "random-image":
        image_pool = [_coerce_image(s["image"]) for s in samples]

    return asyncio.run(
        _run_async(
            samples=samples,
            model=model,
            client=client,
            extract_client=extract_client,
            judge_client=judge_client,
            extract_model=extract_model,
            judge_model=judge_model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            mcq_rotations=mcq_rotations,
            ablation=ablation,
            image_pool=image_pool,
            request_concurrency=request_concurrency,
            judge_concurrency=judge_concurrency,
            cache_dir=cache_dir,
            progress=progress,
        )
    )


async def _run_async(
    *,
    samples: list[dict[str, Any]],
    model: str,
    client: _Client,
    extract_client: _Client,
    judge_client: _Client,
    extract_model: str,
    judge_model: str,
    temperature: float,
    max_new_tokens: int,
    mcq_rotations: int,
    ablation: str,
    image_pool: list[Image.Image],
    request_concurrency: int,
    judge_concurrency: int,
    cache_dir: Path,
    progress: bool,
) -> list[InferenceItem]:
    request_sem = asyncio.Semaphore(request_concurrency)
    judge_sem = asyncio.Semaphore(judge_concurrency)

    async def _llm_extract_mcq(response: str, choices: list[str]) -> str | None:
        options = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
        prompt = MCQ_EXTRACT_PROMPT.format(options=options, response=response)
        try:
            text = await extract_client.chat(
                model=extract_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4,
            )
        except Exception as exc:
            logger.warning("MCQ LLM extraction failed: %s", exc)
            return None
        valid = {chr(65 + i) for i in range(len(choices))}
        for ch in text.upper():
            if ch in valid:
                return ch
        return None

    async def _llm_extract_tf(response: str) -> bool | None:
        prompt = TF_EXTRACT_PROMPT.format(response=response[:500])
        try:
            text = await extract_client.chat(
                model=extract_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4,
            )
        except Exception as exc:
            logger.warning("TF LLM extraction failed: %s", exc)
            return None
        text = text.strip().lower()
        if "true" in text:
            return True
        if "false" in text:
            return False
        return None

    async def _ask(prompt: str, image: Image.Image | None) -> str:
        async with request_sem:
            messages = _build_messages(prompt, image)
            return await client.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )

    async def _judge(question: str, ref: str, pred: str, synonyms, rubric) -> dict | None:
        prompt = format_judge_prompt(
            question=question,
            ref=ref,
            pred=pred,
            synonyms=synonyms,
            rubric=rubric,
        )
        cached = _load_judge_cache(prompt, cache_dir)
        if cached is not None:
            return cached
        async with judge_sem:
            try:
                text = await judge_client.chat(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=128,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                logger.warning("Judge call failed: %s", exc)
                return None
        text = text.strip()
        if text.startswith("```"):
            import re

            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Judge returned non-JSON: %.100s", text)
            return None
        _save_judge_cache(prompt, result, cache_dir)
        return result

    async def _process_one(sample_index: int, sample: dict[str, Any]) -> list[InferenceItem]:
        qtype = sample["question_type"]
        question = sample["question"]
        ref = sample.get("answer", "")
        choices = list(sample.get("choices") or [])
        image_id = sample.get("image_id", f"sample_{sample_index}")
        image = _coerce_image(sample["image"]) if sample.get("image") is not None else None
        sent_image = (
            _ablation.apply_ablation(
                image,
                ablation,  # type: ignore[arg-type]
                pool=image_pool,
                sample_index=sample_index,
            )
            if image is not None
            else None
        )

        if qtype == "mcq":
            return await _process_mcq(
                sample_index=sample_index,
                image_id=image_id,
                question=question,
                ref=ref,
                choices=choices,
                image=sent_image,
            )
        if qtype == "kprim":
            return await _process_kprim(
                sample_index=sample_index,
                image_id=image_id,
                question=question,
                ref=ref,
                statements=choices,
                image=sent_image,
            )
        if qtype == "free_text":
            return await _process_free_text(
                sample=sample,
                sample_index=sample_index,
                image_id=image_id,
                question=question,
                ref=ref,
                image=sent_image,
            )
        raise ValueError(f"Unknown question_type: {qtype!r}")

    async def _process_mcq(
        *,
        sample_index: int,
        image_id: str,
        question: str,
        ref: str,
        choices: list[str],
        image: Image.Image | None,
    ) -> list[InferenceItem]:
        n_rotations = max(1, min(mcq_rotations, len(choices) or 1))
        gold_letter = (ref or "").strip().upper()[:1]
        items: list[InferenceItem] = []
        rotation_ids = list(range(n_rotations))

        async def _one_rotation(rot: int) -> InferenceItem:
            rotated_choices, new_gold = cyclic_shift_choices(choices, gold_letter, rot)
            prompt = format_mcq_prompt(question, rotated_choices)
            try:
                response = await _ask(prompt, image)
            except Exception as exc:
                logger.warning(
                    "MCQ inference failed (sample %s rot %s): %s", sample_index, rot, exc
                )
                response = ""
            extracted = parse_mcq_response(response, rotated_choices)
            if extracted is None:
                extracted = await _llm_extract_mcq(response, rotated_choices)
            if extracted is None:
                extracted = random_letter(rotated_choices, seed=sample_index * 7 + rot)
                logger.warning(
                    "MCQ extraction failed (sample %s rot %s); random fallback %s",
                    sample_index,
                    rot,
                    extracted,
                )
            score = float(extracted == new_gold)
            return InferenceItem(
                sample_index=sample_index,
                image_id=image_id,
                question_type="mcq",
                question=question,
                ref=new_gold,
                pred=extracted,
                score=score,
                base_question_id=image_id + "::" + str(sample_index),
                rotation_id=rot,
                raw_responses=[response],
            )

        items.extend(await asyncio.gather(*[_one_rotation(r) for r in rotation_ids]))
        return items

    async def _process_kprim(
        *,
        sample_index: int,
        image_id: str,
        question: str,
        ref: str,
        statements: list[str],
        image: Image.Image | None,
    ) -> list[InferenceItem]:
        gt_bools = parse_kprim_ground_truth(ref)
        prompts = [format_kprim_prompt(question, s) for s in statements]
        responses: list[str] = await asyncio.gather(
            *[_ask(p, image) for p in prompts],
            return_exceptions=False,
        )
        pred_bools: list[bool | None] = [parse_tf_response(r) for r in responses]
        for i, parsed in enumerate(pred_bools):
            if parsed is None and responses[i]:
                pred_bools[i] = await _llm_extract_tf(responses[i])
        kprim = score_kprim(pred_bools, gt_bools)
        item = InferenceItem(
            sample_index=sample_index,
            image_id=image_id,
            question_type="kprim",
            question=question,
            ref=json.dumps(gt_bools),
            pred=json.dumps(pred_bools),
            score=kprim,
            pred_bools=pred_bools,
            raw_responses=responses,
        )
        return [item]

    async def _process_free_text(
        *,
        sample: dict[str, Any],
        sample_index: int,
        image_id: str,
        question: str,
        ref: str,
        image: Image.Image | None,
    ) -> list[InferenceItem]:
        prompt = format_free_text_prompt(question)
        try:
            response = await _ask(prompt, image)
        except Exception as exc:
            logger.warning("Free-text inference failed (sample %s): %s", sample_index, exc)
            response = ""
        synonyms = sample.get("synonyms")
        rubric = sample.get("scoring_rubric")
        if isinstance(synonyms, str):
            try:
                synonyms = json.loads(synonyms)
            except (json.JSONDecodeError, TypeError):
                synonyms = [synonyms]
        judge_result = await _judge(question, ref, response, synonyms, rubric)
        judge_score: float | None = None
        judge_details: dict[str, int] | None = None
        if judge_result is not None:
            try:
                da = float(judge_result.get("diagnostic_accuracy", 0))
                comp = float(judge_result.get("completeness", 0))
                judge_score = (JUDGE_ACCURACY_WEIGHT * da + JUDGE_COMPLETENESS_WEIGHT * comp) / 10.0
                judge_details = {
                    "diagnostic_accuracy": int(da),
                    "completeness": int(comp),
                }
            except (TypeError, ValueError):
                judge_score = None
        item = InferenceItem(
            sample_index=sample_index,
            image_id=image_id,
            question_type="free_text",
            question=question,
            ref=ref,
            pred=response,
            score=judge_score if judge_score is not None else 0.0,
            judge_score=judge_score,
            judge_details=judge_details,
            raw_responses=[response],
        )
        return [item]

    tasks = [_process_one(i, s) for i, s in enumerate(samples)]
    iterator: Iterable
    if progress:
        bar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating")
        iterator = bar
    else:
        iterator = asyncio.as_completed(tasks)

    out: list[InferenceItem] = []
    for fut in iterator:
        items = await fut
        out.extend(items)
    if progress:
        bar.close()  # type: ignore[union-attr]
    out.sort(key=lambda it: (it.sample_index, it.rotation_id or 0))
    return out
