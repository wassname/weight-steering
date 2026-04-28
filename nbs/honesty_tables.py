"""Three tables from existing per-row CSVs: SI, raw logratios, raw flips.

Loads `out/honesty/cross_adapter_full_dd/dilemmas_per_row.csv` (6 adapters ×
5 coeffs × 438 rows) and `out/honesty/prompt_baseline/dilemmas_per_row.csv`
(5 prompts + dW × {-1,0,1}). Computes:

  table 1: SI per method (bidirectional ref-anchored)
  table 2: raw mean_logratio_honesty mean ± std at coeff in {-1, 0, +1}
  table 3: raw flip counts (fix_fwd / broke_fwd / flip_rev / counter_rev)

n_seeds is fixed to 1 here. When multiseed runs land, group by (method, seed)
and aggregate mean/std across seeds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from tabulate import tabulate

from ws.eval.dilemmas import compute_full_metrics


N_SEEDS = 1  # update when multiseed runs exist


def _flip_counts(df: pl.DataFrame) -> dict:
    """Per-method/adapter flip counts. Requires coeff ∈ {-1,0,+1}."""
    ref = df.filter(pl.col("coeff") == 0.0).sort("idx")
    pos = df.filter(pl.col("coeff") == 1.0).sort("idx")
    neg = df.filter(pl.col("coeff") == -1.0).sort("idx")
    if len(ref) == 0 or len(pos) == 0:
        return {"n_cho": 0, "n_rej": 0, "fix_fwd": 0, "broke_fwd": 0,
                "flip_rev": 0, "counter_rev": 0}
    y_ref = ref["logratio_honesty"].to_numpy()
    y_pos = pos["logratio_honesty"].to_numpy()
    cho = y_ref > 0
    rej = y_ref < 0
    n_cho = int(cho.sum())
    n_rej = int(rej.sum())
    fix_fwd = int(((rej) & (y_pos > 0)).sum())
    broke_fwd = int(((cho) & (y_pos < 0)).sum())
    flip_rev = counter_rev = 0
    if len(neg) > 0:
        y_neg = neg["logratio_honesty"].to_numpy()
        flip_rev = int(((cho) & (y_neg < 0)).sum())
        counter_rev = int(((rej) & (y_neg > 0)).sum())
    return {"n_cho": n_cho, "n_rej": n_rej,
            "fix_fwd": fix_fwd, "broke_fwd": broke_fwd,
            "flip_rev": flip_rev, "counter_rev": counter_rev}


def tables_for(per_row_path: Path, group_col: str, base_filter: str | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = pl.read_csv(per_row_path)
    groups = df[group_col].unique().to_list()
    if base_filter is None:
        base_ref = df
    else:
        base_ref = df.filter(pl.col(group_col) == base_filter)

    # Table 1: SI
    si_rows = []
    for g in groups:
        gdf = df.filter(pl.col(group_col) == g)
        if base_filter is not None and g == base_filter:
            # base vs itself is identity; SI undefined, skip in table
            si_rows.append({group_col: g, "SI": float("nan"), "si_fwd": 0.0, "si_rev": float("nan"),
                            "n_seeds": N_SEEDS, "n_samples": len(gdf)})
            continue
        if base_filter is not None:
            ref0 = base_ref.filter(pl.col("coeff") == 0.0).sort("idx")
            pos = gdf.filter(pl.col("coeff") == 1.0)
            if len(pos) == 0:
                pos = gdf.filter(pl.col("coeff") == 0.0)  # prompt method, coeff=0 only
            neg = gdf.filter(pl.col("coeff") == -1.0)
            stitched = pl.concat([
                ref0.select(["idx", "logratio_honesty", "pmass"]).with_columns(pl.lit(0.0).alias("coeff")),
                pos.sort("idx").select(["idx", "logratio_honesty", "pmass"]).with_columns(pl.lit(1.0).alias("coeff")),
            ] + ([neg.sort("idx").select(["idx", "logratio_honesty", "pmass"]).with_columns(pl.lit(-1.0).alias("coeff"))] if len(neg) > 0 else []))
            m = compute_full_metrics(stitched)
        else:
            m = compute_full_metrics(gdf)
        si_rows.append({
            group_col: g,
            "SI": float(m.get("surgical_informedness", float("nan")) or float("nan")),
            "si_fwd": float(m.get("si_fwd", float("nan")) or float("nan")),
            "si_rev": float(m.get("si_rev", float("nan")) or float("nan")),
            "n_seeds": N_SEEDS,
            "n_samples": int(m.get("n_samples", len(gdf.filter(pl.col("coeff") == 0.0)))),
        })
    si_df = pl.DataFrame(si_rows).sort("SI", descending=True, nulls_last=True)

    # Table 2: raw logratios at coeff ∈ {-1,0,+1}
    lr_rows = []
    for g in groups:
        gdf = df.filter(pl.col(group_col) == g)
        for c in [-1.0, 0.0, 1.0]:
            cdf = gdf.filter(pl.col("coeff") == c)
            if len(cdf) == 0:
                continue
            y = cdf["logratio_honesty"].to_numpy()
            lr_rows.append({
                group_col: g,
                "coeff": c,
                "mean_lr_honesty": float(np.mean(y)),
                "std_lr_honesty": float(np.std(y, ddof=1)) if len(y) > 1 else float("nan"),
                "mean_pmass": float(cdf["pmass"].mean()),
                "n_rows": len(cdf),
                "n_seeds": N_SEEDS,
            })
    lr_df = pl.DataFrame(lr_rows).sort([group_col, "coeff"])

    # Table 3: raw flip counts
    flip_rows = []
    for g in groups:
        gdf = df.filter(pl.col(group_col) == g)
        # For prompt methods, ref must be base. Otherwise use own coeff=0.
        if base_filter is not None and g != base_filter:
            ref0 = base_ref.filter(pl.col("coeff") == 0.0).sort("idx")
            pos = gdf.filter(pl.col("coeff") == 1.0)
            if len(pos) == 0:
                pos = gdf.filter(pl.col("coeff") == 0.0)
            neg = gdf.filter(pl.col("coeff") == -1.0)
            stitched = pl.concat([
                ref0.select(["idx", "logratio_honesty"]).with_columns(pl.lit(0.0).alias("coeff")),
                pos.sort("idx").select(["idx", "logratio_honesty"]).with_columns(pl.lit(1.0).alias("coeff")),
            ] + ([neg.sort("idx").select(["idx", "logratio_honesty"]).with_columns(pl.lit(-1.0).alias("coeff"))] if len(neg) > 0 else []))
            counts = _flip_counts(stitched)
        else:
            counts = _flip_counts(gdf)
        flip_rows.append({group_col: g, **counts, "n_seeds": N_SEEDS})
    flip_df = pl.DataFrame(flip_rows).sort(group_col)

    return si_df, lr_df, flip_df


def fmt(df: pl.DataFrame, name: str) -> str:
    return f"\n=== {name} ===\n" + tabulate(df.to_pandas(), headers="keys", tablefmt="pipe", floatfmt="+.3f", showindex=False)


def main():
    out_root = Path("out/honesty")

    # Adapter tables (T2 source)
    print("\n" + "=" * 70)
    print("ADAPTERS (out/honesty/cross_adapter_full_dd/dilemmas_per_row.csv)")
    print("=" * 70)
    si, lr, fl = tables_for(out_root / "cross_adapter_full_dd/dilemmas_per_row.csv", "adapter")
    print(fmt(si, "1. SI per adapter (ref-anchored bidirectional)"))
    print(fmt(lr, "2. Raw mean_logratio_honesty per (adapter, coeff)"))
    print(fmt(fl, "3. Raw flip counts per adapter (n_cho/n_rej at ref; fix/broke fwd; flip/counter rev)"))

    # Prompt baseline tables (T3 source); ref against base
    print("\n" + "=" * 70)
    print("PROMPTS + dW (out/honesty/prompt_baseline/dilemmas_per_row.csv)")
    print("=" * 70)
    si, lr, fl = tables_for(out_root / "prompt_baseline/dilemmas_per_row.csv", "method", base_filter="base")
    print(fmt(si, "1. SI per method (prompts ref-anchored vs base@0; dW bidirectional vs own 0)"))
    print(fmt(lr, "2. Raw mean_logratio_honesty per (method, coeff)"))
    print(fmt(fl, "3. Raw flip counts per method"))

    # RepE / activation baseline (T1)
    repe_path = out_root / "activation_baseline/dilemmas_per_row.csv"
    if repe_path.exists():
        print("\n" + "=" * 70)
        print(f"REPE / ACTIVATION BASELINE ({repe_path})")
        print("=" * 70)
        # filter to {-1, 0, +1} coeffs for SI bidirectional
        df = pl.read_csv(repe_path).filter(pl.col("coeff").is_in([-1.0, 0.0, 1.0]))
        df.write_csv(out_root / "activation_baseline/_dilemmas_per_row_pm1.csv")
        si, lr, fl = tables_for(out_root / "activation_baseline/_dilemmas_per_row_pm1.csv", "method")
        print(fmt(si, "1. SI per method (RepE vs own 0; bidirectional)"))
        print(fmt(lr, "2. Raw mean_logratio_honesty per (method, coeff)"))
        print(fmt(fl, "3. Raw flip counts per method"))


if __name__ == "__main__":
    main()
