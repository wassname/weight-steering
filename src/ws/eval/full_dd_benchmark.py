"""Full daily-dilemmas benchmark for current Qwen adapter `dW`s.

Writes the central artifact required by `fork_plan.md`:
`out/sycophancy/cross_adapter_full_dd/dilemmas_summary.csv` with 394 base rows
per coeff for the full 197-dilemma AntiPaSTO exact-`Value/Honesty` split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
import tyro
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from ws._log import final_summary, get_argv, setup_logging
from ws.diff import DIFF_FILENAME, load_diff
from ws.eval.dilemmas import DilemmasCfg, compute_full_metrics, evaluate


@dataclass
class FullDDBenchmarkCfg:
    model: str = "Qwen/Qwen3-0.6B"
    behavior: str = "sycophancy"
    adapters: tuple[str, ...] = ("lora", "pissa", "delora", "dora", "oft", "ia3")
    coeffs: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    n_dilemmas: int = 223
    batch_size: int = 8
    out: Path = Path("out")

    @property
    def expected_base_rows_per_coeff(self) -> int:
        return 2 * self.n_dilemmas


def _summarize(df: pl.DataFrame) -> pl.DataFrame:
    summary = df.group_by(["adapter", "coeff"]).agg(
        pl.col("logratio_honesty").mean().alias("mean_logratio_honesty"),
        pl.col("logratio_honesty").std().alias("std_logratio_honesty"),
        pl.col("pmass").mean().alias("mean_pmass"),
        pl.col("low_pmass").mean().alias("frac_low_pmass"),
        pl.len().alias("n_base_rows_per_coeff"),
    )
    zero = summary.filter(pl.col("coeff") == 0.0).select(
        "adapter", pl.col("mean_logratio_honesty").alias("mean_logratio_honesty_0")
    )
    summary = summary.join(zero, on="adapter", how="left").with_columns(
        (pl.col("mean_logratio_honesty") - pl.col("mean_logratio_honesty_0")).alias("delta_vs_0"),
    ).sort(["adapter", "coeff"])

    # SI per adapter (bidirectional; uses coeff=-1/0/+1)
    si_rows = []
    for adapter in df["adapter"].unique().to_list():
        adf = df.filter(pl.col("adapter") == adapter)
        m = compute_full_metrics(adf)
        si_rows.append({"adapter": adapter, "SI": m["surgical_informedness"], "si_fwd": m["si_fwd"], "si_rev": m.get("si_rev", float("nan"))})
    si_df = pl.DataFrame(si_rows)
    return summary.join(si_df, on="adapter", how="left")


def main(cfg: FullDDBenchmarkCfg) -> None:
    setup_logging("full_dd_benchmark")
    out_dir = cfg.out / cfg.behavior / "cross_adapter_full_dd"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    parts = []
    dcfg = DilemmasCfg(
        model_id=cfg.model,
        coeffs=cfg.coeffs,
        n_dilemmas=cfg.n_dilemmas,
        batch_size=cfg.batch_size,
        n_think=128,
    )
    for adapter in cfg.adapters:
        w_path = cfg.out / cfg.behavior / adapter / DIFF_FILENAME
        w = load_diff(w_path)
        logger.info(f"\n=== adapter={adapter} ===")
        df = evaluate(dcfg, w, model=model, tok=tok).with_columns(pl.lit(adapter).alias("adapter"))
        parts.append(df)

    per_row = pl.concat(parts)
    per_row_path = out_dir / "dilemmas_per_row.csv"
    per_row.write_csv(per_row_path)
    summary = _summarize(per_row)
    summary_path = out_dir / "dilemmas_summary.csv"
    summary.write_csv(summary_path)

    row_counts = summary.group_by("adapter").agg(
        pl.col("n_base_rows_per_coeff").min().alias("min_rows"),
        pl.col("n_base_rows_per_coeff").max().alias("max_rows"),
    )
    expected_rows = cfg.expected_base_rows_per_coeff
    bad_counts = row_counts.filter((pl.col("min_rows") != expected_rows) | (pl.col("max_rows") != expected_rows)).height
    best = summary.filter(pl.col("coeff") == 1.0).sort("SI", descending=True, nulls_last=True)
    print("\nfull daily-dilemmas benchmark")
    print(
        f"SHOULD: every adapter has n_base_rows_per_coeff={expected_rows} for every coeff. "
        "ELSE requested split size was not used."
    )
    print("SI = surgical_informedness (ref-anchored, bidirectional, k_fpr=2). Higher=better.")
    print(tabulate(best.to_pandas(), headers="keys", tablefmt="tsv", floatfmt="+.3f", showindex=False))
    cue = "🟢" if bad_counts == 0 else "🔴"
    final_summary(
        out=summary_path,
        argv=get_argv(),
        main_metric=f"bad_row_count_adapters={bad_counts}; best_SI={best['adapter'][0]} SI={float(best['SI'][0]):+.3f}",
        cue=cue,
        table_rows=best.select("adapter", "SI", "si_fwd", "si_rev", "delta_vs_0", "mean_pmass", "n_base_rows_per_coeff").rows(),
        headers=["adapter", "SI", "si_fwd", "si_rev", "delta_vs_0", "pmass", "n_rows"],
        floatfmt="",
    )


if __name__ == "__main__":
    main(tyro.cli(FullDDBenchmarkCfg))
