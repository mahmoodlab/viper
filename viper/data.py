"""Dataset loader for VIPER.

Wraps ``datasets.load_dataset("MahmoodLab/viper")`` with schema validation and
a local-parquet fallback for offline use or pre-release rebuilds.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset

# Canonical schema. Every published VIPER row carries exactly these columns.
EXPECTED_COLUMNS: frozenset[str] = frozenset(
    {
        "image",
        "question",
        "question_type",
        "answer",
        "choices",
        "synonyms",
        "scoring_rubric",
        "organ",
        "category",
        "magnification",
        "source",
        "image_id",
    }
)

EXPECTED_QUESTION_TYPES: frozenset[str] = frozenset({"mcq", "kprim", "free_text"})

EXPECTED_ROW_COUNT = 1251


def load_viper(
    source: str = "MahmoodLab/viper",
    *,
    split: str = "test",
    revision: str | None = None,
    streaming: bool = False,
) -> Dataset:
    """Load the VIPER benchmark.

    Args:
        source: Either a Hugging Face dataset ID (default ``"MahmoodLab/viper"``)
            or a path to a local ``viper.parquet`` (or directory containing one).
        split: Split name on the Hub. The published dataset only has ``"test"``.
        revision: Optional Hub revision (commit hash, branch, or tag).
        streaming: If ``True``, returns an ``IterableDataset`` for memory-light
            iteration. Schema validation is skipped in this mode.

    Returns:
        A ``datasets.Dataset`` with the canonical VIPER schema.
    """
    if _looks_like_local_path(source):
        ds = _load_local(source)
    else:
        ds = load_dataset(source, split=split, revision=revision, streaming=streaming)

    if not streaming:
        _validate_schema(ds)
    return ds


def _looks_like_local_path(source: str) -> bool:
    """Heuristic: contains a path separator, ``.parquet``, or actually exists."""
    if "/" in source and Path(source).expanduser().exists():
        return True
    return source.endswith(".parquet")


def _load_local(path_str: str) -> Dataset:
    path = Path(path_str).expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob("*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No .parquet file found in {path}")
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(f"Local VIPER parquet not found: {path}")
    return load_dataset("parquet", data_files=str(path), split="train")


def _validate_schema(ds: Dataset) -> None:
    """Assert the loaded dataset matches the published VIPER schema."""
    columns = set(ds.column_names)
    missing = EXPECTED_COLUMNS - columns
    extra = columns - EXPECTED_COLUMNS
    if missing:
        raise ValueError(
            f"VIPER dataset is missing required columns: {sorted(missing)}. Got: {sorted(columns)}"
        )
    if extra:
        # Extras are allowed but flag them so users know what they have.
        # (Hugging Face Hub may add a streaming-friendly index column, e.g.)
        pass
    if "question_type" in columns:
        seen_types = set(ds.unique("question_type"))
        unexpected = seen_types - EXPECTED_QUESTION_TYPES
        if unexpected:
            raise ValueError(
                f"VIPER dataset contains unexpected question_type values: {sorted(unexpected)}"
            )
