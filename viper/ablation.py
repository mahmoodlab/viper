"""Ablation transforms for paper §3 sanity checks.

Three ablations cover the three null-hypotheses tested in the paper:

- ``black-image``: replace the image with a solid black PNG (probes whether
  the model is using the image at all; a model with no visual grounding will
  still produce confident-looking answers).
- ``no-image``: omit the image attachment entirely (text-only baseline).
- ``random-image``: swap with another VIPER ROI sampled at fixed seed (probes
  whether the model is using the *correct* image versus reading the question
  + any image).

Each transform returns a (possibly None) PIL.Image plus a normalized label,
so the inference path can decide whether to attach the image to the request.
"""

from __future__ import annotations

import io
import random
from typing import Literal

from PIL import Image

AblationKind = Literal["none", "black-image", "no-image", "random-image"]
ABLATION_KINDS: tuple[AblationKind, ...] = (
    "none",
    "black-image",
    "no-image",
    "random-image",
)


def black_image(size: tuple[int, int] = (1024, 1024)) -> Image.Image:
    """Solid-black RGB PIL image."""
    return Image.new("RGB", size, (0, 0, 0))


def to_png_bytes(img: Image.Image) -> bytes:
    """Re-encode a PIL image as PNG (also strips EXIF/textual metadata)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def apply_ablation(
    image: Image.Image,
    kind: AblationKind,
    *,
    pool: list[Image.Image] | None = None,
    seed: int = 0,
    sample_index: int = 0,
) -> Image.Image | None:
    """Return the image to send to the model under the requested ablation.

    Args:
        image: Original ROI for this sample.
        kind: Ablation kind. ``"none"`` returns the input unchanged.
        pool: Pool of images to sample from for ``random-image``. Defaults to
            ``[image]`` (a no-op fallback).
        seed: Base seed. The per-sample seed is ``seed + sample_index`` so the
            random-image swap is deterministic and stratified.
        sample_index: Index of the current sample in the eval list.

    Returns:
        A PIL image, or ``None`` if the image should be omitted entirely
        (``no-image`` ablation).
    """
    if kind == "none":
        return image
    if kind == "no-image":
        return None
    if kind == "black-image":
        return black_image(size=image.size)
    if kind == "random-image":
        candidates = pool or [image]
        if len(candidates) <= 1:
            return image
        rng = random.Random(seed + sample_index)
        # Prevent self-selection: keep drawing until we get a different image.
        for _ in range(10):
            cand = rng.choice(candidates)
            if cand is not image:
                return cand
        return candidates[0]
    raise ValueError(f"Unknown ablation kind: {kind!r}")
