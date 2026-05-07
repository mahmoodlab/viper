"""``EvalConfig``, the provenance dataclass for every VIPER evaluation run.

Captures every parameter that affects the final score so two runs with the
same EvalConfig produce statistically identical numbers. The full config is
serialized into ``results.json`` under the ``config`` key.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import subprocess
from pathlib import Path

from viper.prompts import judge_prompt_fingerprint


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    """Immutable configuration for a single VIPER run."""

    # Model under test
    model: str
    api_base: str | None
    temperature: float
    max_new_tokens: int

    # Dataset
    dataset_source: str  # "MahmoodLab/viper" or local parquet path
    dataset_revision: str | None
    limit: int | None

    # MCQ rotations (paper §2: 5 cyclic shifts)
    mcq_rotations: int

    # Ablation
    ablation: str

    # Judge & extraction
    judge_model: str
    judge_prompt_hash: str
    extract_model: str

    # Concurrency
    request_concurrency: int
    judge_concurrency: int

    # Provenance
    viper_version: str
    code_git_hash: str
    code_dirty: bool
    run_timestamp: str

    @classmethod
    def build(
        cls,
        *,
        model: str,
        api_base: str | None,
        temperature: float,
        max_new_tokens: int,
        dataset_source: str,
        dataset_revision: str | None,
        limit: int | None,
        mcq_rotations: int,
        ablation: str,
        judge_model: str,
        extract_model: str,
        request_concurrency: int,
        judge_concurrency: int,
    ) -> EvalConfig:
        """Construct an EvalConfig, filling in provenance fields automatically."""
        from viper import __version__

        git_hash, dirty = _git_info()
        return cls(
            model=model,
            api_base=api_base,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            dataset_source=dataset_source,
            dataset_revision=dataset_revision,
            limit=limit,
            mcq_rotations=mcq_rotations,
            ablation=ablation,
            judge_model=judge_model,
            judge_prompt_hash=judge_prompt_fingerprint(),
            extract_model=extract_model,
            request_concurrency=request_concurrency,
            judge_concurrency=judge_concurrency,
            viper_version=__version__,
            code_git_hash=git_hash,
            code_dirty=dirty,
            run_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        return dataclasses.asdict(self)

    def output_dir_name(self) -> str:
        """Slugified human-readable directory name for ``eval_logs/<name>/``."""
        slug = self.model
        if "://" in slug:
            slug = slug.split("://", 1)[1]
        slug = slug.replace("/", "--")
        if self.ablation and self.ablation != "none":
            slug = f"{slug}_{self.ablation}"
        return slug


def _git_info() -> tuple[str, bool]:
    """Return ``(short_hash, is_dirty)`` for the package source tree.

    Best-effort: returns ``("", False)`` when git is unavailable or the
    package was installed from a wheel.
    """
    repo_dir = Path(__file__).resolve().parent.parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=5,
        )
        git_hash = head.stdout.strip() if head.returncode == 0 else ""
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            capture_output=True,
            cwd=repo_dir,
            timeout=5,
        )
        is_dirty = dirty.returncode != 0
        return git_hash, is_dirty
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "", False


# ---------------------------------------------------------------------------
#  Environment-variable resolution helpers used by the CLI
# ---------------------------------------------------------------------------


def resolve_judge_model(cli_value: str | None) -> str:
    """CLI flag > ``JUDGE_MODEL`` env var > paper default ``gpt-5.4``."""
    return cli_value or os.environ.get("JUDGE_MODEL", "gpt-5.4")


def resolve_extract_model(cli_value: str | None) -> str:
    """CLI flag > ``EXTRACT_MODEL`` env var > default ``gpt-4o-mini``."""
    return cli_value or os.environ.get("EXTRACT_MODEL", "gpt-4o-mini")


def resolve_int_env(name: str, default: int) -> int:
    """Parse an int env var with a fallback default."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
