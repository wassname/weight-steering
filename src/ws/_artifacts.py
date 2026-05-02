"""Artifact naming helpers for collision-free run outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


def timestamp_prefix() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def model_slug(model_id: str) -> str:
    slug = model_id.split("/")[-1].strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug or "model"


def latest_matching(out_dir: Path, pattern: str, legacy_name: str | None = None) -> Path:
    matches = sorted(out_dir.glob(pattern))
    if matches:
        return matches[-1]
    if legacy_name is not None:
        legacy = out_dir / legacy_name
        if legacy.exists():
            return legacy
    raise FileNotFoundError(f"no artifact in {out_dir} matching {pattern!r}")


def preferred_matching(out_dir: Path, patterns: list[str], legacy_name: str | None = None) -> Path:
    for pattern in patterns:
        matches = sorted(out_dir.glob(pattern))
        if matches:
            return matches[-1]
    if legacy_name is not None:
        legacy = out_dir / legacy_name
        if legacy.exists():
            return legacy
    joined = ", ".join(repr(p) for p in patterns)
    raise FileNotFoundError(f"no artifact in {out_dir} matching any of [{joined}]")

