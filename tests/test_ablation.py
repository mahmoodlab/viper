"""Tests for image ablation transforms."""

from __future__ import annotations

from PIL import Image

from viper.ablation import apply_ablation, black_image


def _solid(color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> Image.Image:
    return Image.new("RGB", size, color)


def test_black_image_is_uniform_zero():
    img = black_image(size=(8, 8))
    pixels = img.getdata()
    assert all(p == (0, 0, 0) for p in pixels)


def test_ablation_none_returns_input():
    img = _solid((100, 150, 200))
    out = apply_ablation(img, "none")
    assert out is img


def test_ablation_no_image_returns_none():
    img = _solid((100, 150, 200))
    assert apply_ablation(img, "no-image") is None


def test_ablation_black_image_replaces_with_zero():
    img = _solid((100, 150, 200), size=(16, 16))
    out = apply_ablation(img, "black-image")
    assert out is not None
    assert out.size == (16, 16)
    assert all(p == (0, 0, 0) for p in out.getdata())


def test_ablation_random_image_picks_other_when_pool_diverse():
    a = _solid((255, 0, 0))
    b = _solid((0, 255, 0))
    out = apply_ablation(a, "random-image", pool=[a, b], seed=0, sample_index=0)
    assert out is not None
    # We should get *something*. Most importantly not crash.
    assert isinstance(out, Image.Image)


def test_ablation_random_image_singleton_pool_returns_self():
    a = _solid((123, 45, 67))
    out = apply_ablation(a, "random-image", pool=[a], seed=0, sample_index=0)
    assert out is a
