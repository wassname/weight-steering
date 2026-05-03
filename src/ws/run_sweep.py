"""Phase 3 entrypoint: run replicate.py for each adapter in {lora, dora, pissa, delora}.

Data is shared across adapters via data_root (no re-generation).
Real evaluation is done by ws.kl_calibrate + ws.scripts.eval_tinymfv_calibrated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import tyro
from loguru import logger
from tabulate import tabulate

from ws._log import final_summary, get_argv, setup_logging
from ws.replicate import Cfg as ReplicateCfg
from ws.replicate import main as replicate_main


@dataclass
class SweepCfg:
    model: str = "Qwen/Qwen3-0.6B"
    behavior: str = "authority"
    adapters: tuple[str, ...] = ("lora", "dora", "pissa", "delora", "oft", "boft", "ia3")
    rank: int = 32
    lr: float = 2e-4
    epochs: float = 1.0
    max_steps: int = -1
    out: Path = Path("out")
    data_root: Path = Path("out/data")
    coeffs: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    n_topics: int = 20
    n_personas: int = 5  # clamped to len(persona_list); narrow honesty uses 1
    n_samples: int = 10  # bump (e.g. 50) when n_personas clamps to keep total pairs


def _run_one(cfg: SweepCfg, adapter: str) -> dict:
    rcfg = ReplicateCfg(
        model=cfg.model, behavior=cfg.behavior, adapter=adapter,
        rank=cfg.rank, lr=cfg.lr, epochs=cfg.epochs, max_steps=cfg.max_steps,
        out=cfg.out, data_root=cfg.data_root, coeffs=cfg.coeffs,
        n_topics=cfg.n_topics, n_personas=cfg.n_personas, n_samples=cfg.n_samples,
    )
    t0 = time.time()
    replicate_main(rcfg)
    wall = time.time() - t0
    return {"adapter": adapter, "wall_s": wall}


def main(cfg: SweepCfg) -> None:
    setup_logging("sweep")
    rows = []
    for adapter in cfg.adapters:
        logger.info(f"=== adapter={adapter} ===")
        row = _run_one(cfg, adapter)
        rows.append(row)
        logger.info(f"adapter={adapter} wall={row['wall_s']:.0f}s")

    df = pl.DataFrame(rows)
    out_path = cfg.out / cfg.behavior / "sweep_summary.csv"
    df.write_csv(out_path)

    print("\nsweep_summary")
    print("SHOULD: all adapters complete without error. Real eval is via ws.kl_calibrate + ws.scripts.eval_tinymfv_calibrated.")
    print(tabulate(df.to_pandas(), tablefmt="tsv", headers="keys", showindex=False))

    cue = "🟢" if len(rows) == len(cfg.adapters) else "🟡"
    final_summary(
        out=out_path, argv=get_argv(),
        main_metric=f"adapters={len(rows)}/{len(cfg.adapters)} wall_s=[{min(r['wall_s'] for r in rows):.0f}, {max(r['wall_s'] for r in rows):.0f}]",
        cue=cue,
        table_rows=[[r["adapter"], f"{r['wall_s']:.0f}"] for r in rows],
        headers=["adapter", "wall_s"],
        floatfmt="",
    )


if __name__ == "__main__":
    main(tyro.cli(SweepCfg))
