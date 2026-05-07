"""Shared pytest fixtures.

Builds a tiny in-memory VIPER-shaped dataset (6 questions over 2 images) so
unit tests run in milliseconds without a network round-trip to the Hub.
"""

from __future__ import annotations

import io
import json

import pytest
from datasets import Dataset, Features, Sequence, Value
from datasets import Image as HFImage
from PIL import Image


def _png_bytes(rgb: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (32, 32), rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def mini_dataset() -> Dataset:
    """6-row VIPER-shaped Dataset for fast unit tests.

    Layout:
        image_0: kidney, 1 MCQ + 1 KPrim + 1 free-text
        image_1: liver,  1 MCQ + 1 KPrim + 1 free-text
    """
    img_red = _png_bytes((220, 30, 30))
    img_blue = _png_bytes((30, 30, 220))

    rows = [
        {
            "image": {"bytes": img_red, "path": None},
            "image_id": "viper_kidney_aaaaaaaaaaaa",
            "question": "What is the predominant feature?",
            "question_type": "mcq",
            "answer": "C",
            "choices": ["fibrosis", "necrosis", "normal", "steatosis", "inflammation"],
            "synonyms": None,
            "scoring_rubric": None,
            "organ": "kidney",
            "category": "identify_pathology",
            "magnification": "20x",
            "source": "TG-GATEs",
        },
        {
            "image": {"bytes": img_red, "path": None},
            "image_id": "viper_kidney_aaaaaaaaaaaa",
            "question": "Are the following statements true or false?",
            "question_type": "kprim",
            "answer": json.dumps([True, False, True, False]),
            "choices": [
                "the tubules are dilated",
                "there is hemorrhage",
                "the cortex is normal",
                "necrosis is present",
            ],
            "synonyms": None,
            "scoring_rubric": None,
            "organ": "kidney",
            "category": "characterize_feature",
            "magnification": "20x",
            "source": "TG-GATEs",
        },
        {
            "image": {"bytes": img_red, "path": None},
            "image_id": "viper_kidney_aaaaaaaaaaaa",
            "question": "Describe the lesion in one sentence.",
            "question_type": "free_text",
            "answer": "Tubular dilatation with mild interstitial fibrosis.",
            "choices": [],
            "synonyms": json.dumps(["tubular ectasia", "tubule dilation"]),
            "scoring_rubric": "Award full credit for naming both tubular dilatation AND interstitial fibrosis.",
            "organ": "kidney",
            "category": "characterize_feature",
            "magnification": "20x",
            "source": "TG-GATEs",
        },
        {
            "image": {"bytes": img_blue, "path": None},
            "image_id": "viper_liver_bbbbbbbbbbbb",
            "question": "What is the predominant feature in the liver?",
            "question_type": "mcq",
            "answer": "A",
            "choices": ["necrosis", "fibrosis", "normal", "steatosis", "inflammation"],
            "synonyms": None,
            "scoring_rubric": None,
            "organ": "liver",
            "category": "identify_pathology",
            "magnification": "20x",
            "source": "MMO",
        },
        {
            "image": {"bytes": img_blue, "path": None},
            "image_id": "viper_liver_bbbbbbbbbbbb",
            "question": "Liver: true or false statements:",
            "question_type": "kprim",
            "answer": json.dumps([False, True, False, True]),
            "choices": [
                "centrilobular necrosis is present",
                "portal tracts are inflamed",
                "no steatosis is visible",
                "the architecture is preserved",
            ],
            "synonyms": None,
            "scoring_rubric": None,
            "organ": "liver",
            "category": "characterize_feature",
            "magnification": "20x",
            "source": "MMO",
        },
        {
            "image": {"bytes": img_blue, "path": None},
            "image_id": "viper_liver_bbbbbbbbbbbb",
            "question": "Describe the liver lesion.",
            "question_type": "free_text",
            "answer": "Centrilobular hepatocellular necrosis.",
            "choices": [],
            "synonyms": None,
            "scoring_rubric": "Award full credit for naming centrilobular necrosis.",
            "organ": "liver",
            "category": "identify_pathology",
            "magnification": "20x",
            "source": "MMO",
        },
    ]

    features = Features(
        {
            "image": HFImage(),
            "image_id": Value("string"),
            "question": Value("string"),
            "question_type": Value("string"),
            "answer": Value("string"),
            "choices": Sequence(Value("string")),
            "synonyms": Value("string"),
            "scoring_rubric": Value("string"),
            "organ": Value("string"),
            "category": Value("string"),
            "magnification": Value("string"),
            "source": Value("string"),
        }
    )
    return Dataset.from_list(rows, features=features)
