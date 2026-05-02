"""Token-efficient loguru setup + BLUF helper.

Call ``setup_logging("replicate")`` once at the top of an entrypoint's main().
Stdout sink: plain, no-color, ``{message}`` only.
File sink: ``logs/<name>.verbose.log`` at DEBUG with timestamp/location.

Use ``final_summary(...)`` at the very end of main() to emit the standard
last-30-lines block (out: / argv: / main metric: / cue table) that a dumb
summary LLM reads first.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

from loguru import logger
from tabulate import tabulate

_CONFIGURED: set[str] = set()


def quiet_external_logs() -> None:
    """Suppress third-party progress bars and advisory warnings on stdout."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated! Use `dtype` instead!")
    try:
        import datasets

        datasets.disable_progress_bars()
    except Exception:
        pass
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        if hasattr(hf_logging, "disable_progress_bar"):
            hf_logging.disable_progress_bar()
    except Exception:
        pass


def setup_logging(name: str, log_dir: str | Path = "logs") -> Path:
    """Configure loguru once per entrypoint name. Returns the verbose log path."""
    log_path = Path(log_dir) / f"{name}.verbose.log"
    if name in _CONFIGURED:
        return log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    quiet_external_logs()

    logger.remove()
    level = os.environ.get("LOG_LEVEL", "INFO")
    # Stdout: plain, no colors
    logger.add(sys.stdout, level=level, colorize=False, format="{message}")
    # File: full traces for on-demand debugging
    logger.add(
        str(log_path),
        format="{time} | {level} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        enqueue=False,
    )
    _CONFIGURED.add(name)
    logger.info(f"verbose log: {log_path}")
    return log_path


def final_summary(
    *,
    out: str | Path,
    argv: Sequence[str] | str,
    main_metric: str,
    cue: str,
    table_rows: Sequence[Sequence[Any]],
    headers: Sequence[str],
    floatfmt: str = "+.3f",
) -> None:
    """Print the last-30-lines BLUF block.

    cue: '🟢' pass / '🟡' partial / '🔴' fail. Use exactly once per run.
    """
    argv_str = argv if isinstance(argv, str) else " ".join(map(str, argv))
    print()
    print(f"out: {out}")
    print(f"argv: {argv_str}")
    print(f"main metric: {main_metric}")
    rows = [[cue, *r] for r in table_rows]
    print(
        tabulate(
            rows,
            headers=["cue", *headers],
            tablefmt="tsv",
            floatfmt=floatfmt,
        )
    )


def get_argv() -> str:
    """Best-effort argv reconstruction for the BLUF block."""
    return " ".join(sys.argv)
