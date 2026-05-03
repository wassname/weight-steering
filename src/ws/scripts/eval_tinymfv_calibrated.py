"""Run tiny-mfv airisk eval per-adapter at iso-KL calibrated alphas.

Reads `out/<behavior>/kl_calibration/summary.csv` (produced by `ws.kl_calibrate`)
and invokes `ws.eval.tinymfv_airisk` once per adapter with --coeffs
-alpha_neg 0.0 +alpha_pos. Each run writes its own per-frame / per-vignette /
foundations / Δlogit CSVs under `out/<behavior>/<adapter>/`, which are then
consumed by `ws.scripts.readme_tinymfv_table`.

Why a wrapper: kl_calibrate produces asymmetric alpha_pos / alpha_neg per
adapter (steering directions don't have symmetric KL footprint). The base
eval module takes a single `coeffs` tuple, so we read the calibrated values
and forward them as a CLI list -- one process per adapter so signs are clean.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import tyro
from loguru import logger


@dataclass
class EvalTinymfvCalibratedCfg:
    behavior: str = "authority"
    out: Path = Path("out")
    adapters: tuple[str, ...] = ("lora", "dora", "pissa", "delora", "oft", "ia3")
    model: str = "Qwen/Qwen3-4B"
    bootstrap_samples: int = 256
    limit: int = 0
    batch_size: int = 16


def _run(cmd: list[str]) -> int:
    logger.info(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd)


def main(cfg: EvalTinymfvCalibratedCfg) -> None:
    summary_path = cfg.out / cfg.behavior / "kl_calibration" / "summary.csv"
    if not summary_path.exists():
        sys.exit(f"missing kl_calibration summary at {summary_path} -- run ws.kl_calibrate first")
    summary = pl.read_csv(summary_path)

    by_method = {row["method"]: row for row in summary.to_dicts()}

    for adapter in cfg.adapters:
        key = f"dW:{adapter}"
        if key not in by_method:
            logger.warning(f"no calibration for {key}; skipping")
            continue
        row = by_method[key]
        alpha_pos = float(row["alpha_pos"])
        alpha_neg = float(row["alpha_neg"])
        coeffs = [-alpha_neg, 0.0, alpha_pos]
        logger.info(f"=== {adapter}: alpha_pos={alpha_pos:+.3f} alpha_neg={alpha_neg:+.3f} ===")
        rc = _run([
            "uv", "run", "python", "-m", "ws.eval.tinymfv_airisk",
            "--model", cfg.model,
            "--behavior", cfg.behavior,
            "--adapter", adapter,
            "--coeffs", *[f"{c:+.6f}" for c in coeffs],
            "--batch-size", str(cfg.batch_size),
            "--bootstrap-samples", str(cfg.bootstrap_samples),
            *(["--limit", str(cfg.limit)] if cfg.limit > 0 else []),
        ])
        if rc != 0:
            logger.error(f"adapter {adapter} eval exited with rc={rc}")


if __name__ == "__main__":
    main(tyro.cli(EvalTinymfvCalibratedCfg))
