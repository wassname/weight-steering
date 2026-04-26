# %% [markdown]
# # Cross-adapter v9 comparison
#
# Aggregate v9 scope diagnostics + dilemmas summaries across adapters
# (lora, dora, pissa, delora, oft, ia3) and produce a single comparison
# table + figure.
#
# Goals:
# 1. Which adapter family has the biggest scope vs substance gap (block
#    oracle - cumulative oracle agreement with w_oracle)?
# 2. Does behavioral steering (dilemmas mean_logratio at coeff=+1) rank
#    the adapters consistently with subspace metrics?
# 3. Headline table: per adapter, w_oracle pct on its own axis,
#    block-act overlap with w_oracle, dilemmas behavioral score.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from loguru import logger
from tabulate import tabulate

logger.remove()
logger.add(sys.stdout, level="INFO", format="{message}")

ROOT = Path("out/sycophancy")
ADAPTERS = ["lora", "dora", "pissa", "delora", "oft", "ia3"]
OUT_DIR = Path("out/sycophancy/cross_adapter_v9")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        logger.warning(f"missing {path}")
        return None
    return pl.read_csv(path)


# %% [markdown]
# ## Aggregate per-adapter scope diagnostics

# %%
scope_rows = []
for adapter in ADAPTERS:
    scope_path = ROOT / adapter / "v9" / "v9_scope_diagnostic.csv"
    df = safe_read_csv(scope_path)
    if df is None:
        continue
    lora_layers = df.filter(pl.col("is_lora_layer"))
    if lora_layers.height == 0:
        continue
    scope_rows.append({
        "adapter": adapter,
        "n_lora_layers": lora_layers.height,
        "mean_overlap_w_vs_act_cum": float(lora_layers["overlap_w_vs_act_cumulative"].mean()),
        "mean_overlap_w_vs_act_block": float(lora_layers["overlap_w_vs_act_block"].mean()),
        "mean_overlap_act_cum_vs_block": float(lora_layers["overlap_act_cum_vs_block"].mean()),
        "mean_block_over_cum_norm": float(lora_layers["block_over_cumulative"].mean()),
        # Sanity at first LoRA layer.
        "L_first": int(lora_layers["layer"].min()),
        "first_layer_cum_vs_block": float(
            df.filter(pl.col("layer") == lora_layers["layer"].min())["overlap_act_cum_vs_block"][0]
        ),
    })

scope_summary = pl.DataFrame(scope_rows)
print("\n=== cross-adapter scope diagnostic (v9, mean over LoRA-touched layers) ===")
print(
    "SHOULD: mean_overlap_w_vs_act_block > mean_overlap_w_vs_act_cum -- block-local act oracle agrees with weight oracle better than cumulative does. "
    "ELSE: scope is not the only mismatch -- adapter writes into directions that don't show up in the residual stream's principal axes."
)
print(tabulate(scope_summary.to_pandas(), headers="keys", tablefmt="pipe", floatfmt="+.3f", showindex=False))
scope_summary.write_csv(OUT_DIR / "scope_summary.csv")


# %% [markdown]
# ## Aggregate dilemmas behavioral evals

# %%
dil_rows = []
for adapter in ADAPTERS:
    df = safe_read_csv(ROOT / adapter / "dilemmas_summary.csv")
    if df is None:
        continue
    # mean over coeff=+1 minus coeff=0 = behavioral steering effect (more honest)
    if 0.0 not in df["coeff"].to_list() or 1.0 not in df["coeff"].to_list():
        logger.warning(f"{adapter} dilemmas missing coeffs 0,1")
        continue
    base = float(df.filter(pl.col("coeff") == 0.0)["mean_logratio_honesty"][0])
    pos = float(df.filter(pl.col("coeff") == 1.0)["mean_logratio_honesty"][0])
    neg = (
        float(df.filter(pl.col("coeff") == -1.0)["mean_logratio_honesty"][0])
        if -1.0 in df["coeff"].to_list() else float("nan")
    )
    dil_rows.append({
        "adapter": adapter,
        "logratio_at_neg1": neg,
        "logratio_at_0": base,
        "logratio_at_pos1": pos,
        "delta_pos_minus_zero": pos - base,
        "delta_pos_minus_neg": pos - neg,
    })

dil_summary = pl.DataFrame(dil_rows)
print("\n=== cross-adapter dilemmas behavioral steering (v9) ===")
print(
    "SHOULD: delta_pos_minus_zero > 0 (steering at +alpha makes model more honest). "
    "Larger delta = stronger behavioral signal. "
    "ELSE: w doesn't transfer from sycophancy training to honesty dilemmas (OOD failure)."
)
print(tabulate(dil_summary.to_pandas(), headers="keys", tablefmt="pipe", floatfmt="+.3f", showindex=False))
dil_summary.write_csv(OUT_DIR / "dilemmas_summary.csv")


# %% [markdown]
# ## Joint headline table

# %%
if scope_summary.height > 0 and dil_summary.height > 0:
    headline = scope_summary.select([
        "adapter", "mean_overlap_w_vs_act_cum", "mean_overlap_w_vs_act_block",
        "first_layer_cum_vs_block",
    ]).join(
        dil_summary.select(["adapter", "logratio_at_0", "logratio_at_pos1", "delta_pos_minus_zero"]),
        on="adapter", how="full",
    )
    print("\n=== HEADLINE: subspace alignment vs behavioral steering, per adapter ===")
    print(tabulate(headline.to_pandas(), headers="keys", tablefmt="pipe", floatfmt="+.3f", showindex=False))
    headline.write_csv(OUT_DIR / "headline.csv")


# %% [markdown]
# ## Figure: per-adapter scope diagnostic bars

# %%
if scope_summary.height > 0:
    pdf = scope_summary.to_pandas().set_index("adapter")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(pdf))
    width = 0.35
    ax.bar([i - width / 2 for i in x], pdf["mean_overlap_w_vs_act_cum"], width,
           label="cumulative act_oracle", color="#888")
    ax.bar([i + width / 2 for i in x], pdf["mean_overlap_w_vs_act_block"], width,
           label="block-local act_oracle (v9)", color="#2a7")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pdf.index, rotation=20)
    ax.set_ylabel("mean subspace overlap with w_oracle")
    ax.set_title("v9: scope vs substance -- block-local act oracle alignment with weight oracle")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scope_bars.png", dpi=140)
    fig.savefig(OUT_DIR / "scope_bars.pdf")
    plt.show()

logger.info(f"cross-adapter v9 outputs in {OUT_DIR}")
