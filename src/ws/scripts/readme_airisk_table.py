"""Build README-ready AIRisk tables with uncertainty for base and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import tyro
from tabulate import tabulate

from ws._log import final_summary, get_argv, setup_logging
from ws.eval.airisk import compute_metrics


@dataclass
class ReadmeAiriskCfg:
    behavior: str = "honesty"
    out: Path = Path("out")
    adapters: tuple[str, ...] = ("ia3", "oft", "dora", "lora", "pissa", "delora")
    alpha: float = 1.0
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 0


def _bootstrap_airisk(df: pl.DataFrame, n_bootstrap: int, seed: int) -> dict[str, float]:
    idxs = df["idx"].unique().to_list()
    rng = np.random.default_rng(seed)
    lr_p1, lr_0, si_vals = [], [], []
    for _ in range(n_bootstrap):
        sample_ids = rng.choice(idxs, size=len(idxs), replace=True)
        parts = []
        for sid in sample_ids:
            parts.append(df.filter(pl.col("idx") == sid))
        boot = pl.concat(parts)
        lr_p1.append(float(boot.filter(pl.col("coeff") == 1.0)["logratio_value"].mean()))
        lr_0.append(float(boot.filter(pl.col("coeff") == 0.0)["logratio_value"].mean()))
        si_vals.append(float(compute_metrics(boot)["surgical_informedness"]))
    lr_p1 = np.asarray(lr_p1)
    lr_0 = np.asarray(lr_0)
    si_vals = np.asarray(si_vals)
    delta = lr_p1 - lr_0
    return {
        "airisk_lr_0_std": float(lr_0.std(ddof=1)),
        "airisk_lr_0_ci_lo": float(np.quantile(lr_0, 0.025)),
        "airisk_lr_0_ci_hi": float(np.quantile(lr_0, 0.975)),
        "airisk_lr_p1_std": float(lr_p1.std(ddof=1)),
        "airisk_lr_p1_ci_lo": float(np.quantile(lr_p1, 0.025)),
        "airisk_lr_p1_ci_hi": float(np.quantile(lr_p1, 0.975)),
        "airisk_delta_std": float(delta.std(ddof=1)),
        "airisk_delta_ci_lo": float(np.quantile(delta, 0.025)),
        "airisk_delta_ci_hi": float(np.quantile(delta, 0.975)),
        "airisk_si_std": float(si_vals.std(ddof=1)),
        "airisk_si_ci_lo": float(np.quantile(si_vals, 0.025)),
        "airisk_si_ci_hi": float(np.quantile(si_vals, 0.975)),
    }


def _load_airisk_row(out_dir: Path, adapter: str, n_bootstrap: int, seed: int) -> dict[str, float | str]:
    per_row_path = out_dir / adapter / "airisk_truthfulness_per_row.csv"
    df = pl.read_csv(per_row_path)
    point_p1 = df.filter(pl.col("coeff") == 1.0)
    point_0 = df.filter(pl.col("coeff") == 0.0)
    metrics = compute_metrics(df)
    boot = _bootstrap_airisk(df, n_bootstrap, seed)
    return {
        "adapter": adapter,
        "airisk_n": int(point_p1.height),
        "airisk_lr_0": float(point_0["logratio_value"].mean()),
        "airisk_lr_p1": float(point_p1["logratio_value"].mean()),
        "airisk_delta": float(point_p1["logratio_value"].mean() - point_0["logratio_value"].mean()),
        "airisk_si": float(metrics["surgical_informedness"]),
        **boot,
    }


def _load_tinymfv_row(out_dir: Path, adapter: str, alpha: float) -> dict[str, float | str]:
    summary_path = out_dir / adapter / "tinymfv_airisk_summary.csv"
    df = pl.read_csv(summary_path)
    row = df.filter(pl.col("alpha") == alpha).to_dicts()[0]
    base = df.filter(pl.col("alpha") == 0.0).to_dicts()[0]
    return {
        "adapter": adapter,
        "tinymfv_n": int(row["n_vignettes"]),
        "tinymfv_wrongness_0": float(base["wrongness"]),
        "tinymfv_wrongness_0_std": float(base["wrongness_std"]),
        "tinymfv_wrongness_0_ci_lo": float(base["wrongness_ci_lo"]),
        "tinymfv_wrongness_0_ci_hi": float(base["wrongness_ci_hi"]),
        "tinymfv_wrongness_p1": float(row["wrongness"]),
        "tinymfv_wrongness_std": float(row["wrongness_std"]),
        "tinymfv_wrongness_ci_lo": float(row["wrongness_ci_lo"]),
        "tinymfv_wrongness_ci_hi": float(row["wrongness_ci_hi"]),
        "tinymfv_delta": float(row["delta_wrongness_vs_alpha0"]),
        "tinymfv_gap_0": float(base["gap"]),
        "tinymfv_gap_0_std": float(base["gap_std"]),
        "tinymfv_gap_0_ci_lo": float(base["gap_ci_lo"]),
        "tinymfv_gap_0_ci_hi": float(base["gap_ci_hi"]),
        "tinymfv_gap_p1": float(row["gap"]),
        "tinymfv_gap_std": float(row["gap_std"]),
        "tinymfv_gap_ci_lo": float(row["gap_ci_lo"]),
        "tinymfv_gap_ci_hi": float(row["gap_ci_hi"]),
    }


def main() -> None:
    cfg = tyro.cli(ReadmeAiriskCfg)
    setup_logging("readme_airisk_table")
    behavior_dir = cfg.out / cfg.behavior

    rows = []
    for adapter in cfg.adapters:
        airisk = _load_airisk_row(behavior_dir, adapter, cfg.bootstrap_samples, cfg.bootstrap_seed)
        tinymfv = _load_tinymfv_row(behavior_dir, adapter, cfg.alpha)
        merged = {**airisk, **tinymfv}
        rows.append(merged)

    if rows:
        anchor = rows[0]
        rows.append({
            "adapter": "base",
            "airisk_n": anchor["airisk_n"],
            "airisk_lr_0": anchor["airisk_lr_0"],
            "airisk_lr_p1": anchor["airisk_lr_0"],
            "airisk_lr_0_std": anchor["airisk_lr_0_std"],
            "airisk_lr_0_ci_lo": anchor["airisk_lr_0_ci_lo"],
            "airisk_lr_0_ci_hi": anchor["airisk_lr_0_ci_hi"],
            "airisk_lr_p1_std": anchor["airisk_lr_0_std"],
            "airisk_lr_p1_ci_lo": anchor["airisk_lr_0_ci_lo"],
            "airisk_lr_p1_ci_hi": anchor["airisk_lr_0_ci_hi"],
            "airisk_delta": 0.0,
            "airisk_delta_std": 0.0,
            "airisk_delta_ci_lo": 0.0,
            "airisk_delta_ci_hi": 0.0,
            "airisk_si": float("nan"),
            "airisk_si_std": float("nan"),
            "airisk_si_ci_lo": float("nan"),
            "airisk_si_ci_hi": float("nan"),
            "tinymfv_n": anchor["tinymfv_n"],
            "tinymfv_wrongness_0": anchor["tinymfv_wrongness_0"],
            "tinymfv_wrongness_p1": anchor["tinymfv_wrongness_0"],
            "tinymfv_wrongness_0_std": anchor["tinymfv_wrongness_0_std"],
            "tinymfv_wrongness_0_ci_lo": anchor["tinymfv_wrongness_0_ci_lo"],
            "tinymfv_wrongness_0_ci_hi": anchor["tinymfv_wrongness_0_ci_hi"],
            "tinymfv_wrongness_std": anchor["tinymfv_wrongness_0_std"],
            "tinymfv_wrongness_ci_lo": anchor["tinymfv_wrongness_0_ci_lo"],
            "tinymfv_wrongness_ci_hi": anchor["tinymfv_wrongness_0_ci_hi"],
            "tinymfv_delta": 0.0,
            "tinymfv_gap_0": anchor["tinymfv_gap_0"],
            "tinymfv_gap_0_std": anchor["tinymfv_gap_0_std"],
            "tinymfv_gap_0_ci_lo": anchor["tinymfv_gap_0_ci_lo"],
            "tinymfv_gap_0_ci_hi": anchor["tinymfv_gap_0_ci_hi"],
            "tinymfv_gap_p1": anchor["tinymfv_gap_0"],
            "tinymfv_gap_std": anchor["tinymfv_gap_0_std"],
            "tinymfv_gap_ci_lo": anchor["tinymfv_gap_0_ci_lo"],
            "tinymfv_gap_ci_hi": anchor["tinymfv_gap_0_ci_hi"],
        })

    table = pl.DataFrame(rows).sort("airisk_si", descending=True)
    out_path = behavior_dir / "readme_airisk_table.csv"
    table.write_csv(out_path)

    display = table.select([
        "adapter",
        "airisk_lr_p1", "airisk_lr_p1_ci_lo", "airisk_lr_p1_ci_hi",
        "airisk_delta", "airisk_delta_ci_lo", "airisk_delta_ci_hi",
        "airisk_si", "airisk_si_ci_lo", "airisk_si_ci_hi",
        "tinymfv_wrongness_p1", "tinymfv_wrongness_ci_lo", "tinymfv_wrongness_ci_hi",
        "tinymfv_delta",
        "tinymfv_gap_p1", "tinymfv_gap_ci_lo", "tinymfv_gap_ci_hi",
    ])
    print("\nREADME AIRisk table")
    print("SHOULD: AIRisk delta and SI agree on adapter ranking direction. ELSE the eval is unstable.")
    print("SHOULD: tiny-mfv wrongness moves coherently with AIRisk if both capture the same honesty signal.")
    print(tabulate(display.to_pandas(), headers="keys", tablefmt="tsv", floatfmt="+.3f", showindex=False))
    final_summary(
        out=out_path,
        argv=get_argv(),
        main_metric=f"best_airisk_si={float(table['airisk_si'][0]):+.3f}",
        cue="🟢",
        table_rows=display.rows(),
        headers=display.columns,
        floatfmt="+.3f",
    )


if __name__ == "__main__":
    main()
