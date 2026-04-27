"""Phase 3 entrypoint: run replicate.py for each adapter in {lora, dora, pissa, delora}.

Final output: polars table with columns
    (adapter, logratio_spread, pmass_min, ratio_weak_write, wall_s)

Data is shared across adapters via data_root (no re-generation).
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
    behavior: str = "sycophancy"
    adapters: tuple[str, ...] = ("lora", "dora", "pissa", "delora", "oft", "boft", "ia3")
    rank: int = 32
    lr: float = 2e-4
    epochs: float = 1.0
    max_steps: int = -1
    out: Path = Path("out")
    data_root: Path = Path("out/data")
    coeffs: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _run_one(cfg: SweepCfg, adapter: str) -> dict:
    rcfg = ReplicateCfg(
        model=cfg.model, behavior=cfg.behavior, adapter=adapter,
        rank=cfg.rank, lr=cfg.lr, epochs=cfg.epochs, max_steps=cfg.max_steps,
        out=cfg.out, data_root=cfg.data_root, coeffs=cfg.coeffs,
    )
    t0 = time.time()
    replicate_main(rcfg)
    wall = time.time() - t0

    out_dir = cfg.out / cfg.behavior / adapter
    summary = pl.read_csv(out_dir / "eval_summary.csv").sort("coeff")
    spread = float(summary["mean_logratio"][-1]) - float(summary["mean_logratio"][0])
    pmin = float(summary["mean_pmass"].min())

    align = pl.read_csv(out_dir / "subspace_summary.csv")
    write_rows = align.filter(pl.col("kind").is_in(["o_proj", "down_proj"]))
    ratio_weak = float(write_rows["mean_ratio_weak"].mean()) if len(write_rows) else float("nan")

    return {
        "adapter": adapter,
        "logratio_spread": spread,
        "pmass_min": pmin,
        "ratio_weak_write": ratio_weak,
        "wall_s": wall,
    }


def main(cfg: SweepCfg) -> None:
    setup_logging("sweep")
    rows = []
    for adapter in cfg.adapters:
        logger.info(f"=== adapter={adapter} ===")
        row = _run_one(cfg, adapter)
        rows.append(row)
        logger.info(f"adapter={adapter} spread={row['logratio_spread']:+.3f} wall={row['wall_s']:.0f}s")

    df = pl.DataFrame(rows)
    out_path = cfg.out / cfg.behavior / "sweep_summary.csv"
    df.write_csv(out_path)

    print("\nsweep_summary")
    print("SHOULD: lora baseline spread ~12.8 (task 53). dora/pissa within 20% = adapter family "
          "doesn't change the steering subspace much. Large outlier = that init/optimizer alters "
          "which subspace w lands in. ratio_weak_write > 1 = w avoids the lm_head readout.")
    print(tabulate(df.to_pandas(), tablefmt="tsv", headers="keys", floatfmt="+.3f", showindex=False))

    spread_vals = [r["logratio_spread"] for r in rows]
    cue = "🟢" if all(s > 1.0 for s in spread_vals) else ("🟡" if any(s > 0.3 for s in spread_vals) else "🔴")
    final_summary(
        out=out_path, argv=get_argv(),
        main_metric=f"spread [{min(spread_vals):+.2f}, {max(spread_vals):+.2f}]",
        cue=cue,
        table_rows=[[r["adapter"], f"{r['logratio_spread']:+.3f}", f"{r['pmass_min']:.3f}",
                     f"{r['ratio_weak_write']:+.3f}", f"{r['wall_s']:.0f}"]
                    for r in rows],
        headers=["adapter", "logratio_spread", "pmass_min", "ratio_weak_write", "wall_s"],
        floatfmt="",
    )


if __name__ == "__main__":
    main(tyro.cli(SweepCfg))
