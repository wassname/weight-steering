# %% [markdown]
# # Where does the trained dW live?
#
# We have a weight-steering method that works via two LoRA adapters: train one
# on a positive persona, one on a negative, and merge `dW = theta_pos -
# theta_neg` into the base weights. dW steers honesty on daily-dilemmas.
#
# Question: which subspace, modules, or layers of dW carry the steering effect?
#
# Method: causal ablation. Zero parts of dW, re-evaluate on identical
# daily-dilemmas rows (438 rows = 219 dilemmas x 2 actions, base persona),
# report `retained = dd_delta(ablated) / dd_delta(full)`. Close to 0 = the
# zeroed part was necessary; close to 1 = it was redundant. Ratios are
# commensurable across variants because every variant uses the same row keys
# (`max_idx_symmetric_diff = 0` enforced).
#
# Two ablation axes:
# - S-space (`parameterization_ablation.py`): zero singular components of
#   each tensor's own SVD `dW = U S Vh`. Crops: top/mid/bottom by index,
#   top by cumulative energy, and their complements.
# - Layer/module (`layer_module_ablation.py`): zero by layer index and module
#   family. Variants: residual-write only (o_proj, down_proj), attention
#   only, mlp only, single-layer keep, leave-one-layer-out.
#
# Adapters: lora, dora, pissa, delora, oft, ia3. ia3 has no o_proj weight, so
# o_proj-dependent module variants are logged unavailable and skipped.
#
# ## Two-goal frame
#
# Goal A (descriptive, what's run here): given a trained dW, find a coordinate
# system that makes it sparse / low-rank / interpretable. Lenses below: dW's
# own SVD, layer index, module family. Other lenses we have not run yet:
# base-W SVD (`dS = U0.T @ dW @ V0h`, does dW ride pretrained directions?),
# shared cross-adapter SVD (do different adapters converge?), activation-PCA
# (does dW lie in the behavioral contrast subspace?), adapter-architecture
# decompositions (DoRA magnitude vs direction, DeLoRA lambda vs direction,
# OFT rotation, IA3 gates).
#
# Goal B (constructive, deferred): predict a `dW'` from pretrained weights and
# base activations alone, no training. Candidates: TaskDiff/RepE persona
# contrast, function vectors, write-not-read, OV-write, gate-kernel, signed
# SAE features, ReFT-r1, attention min/max/diff. Benchmark would be
# trained-vs-constructed dW on identical DD rows.
#
# ## Coverage gaps in the current ablation set
#
# - Read-side modules (q/k/v/up/gate-only) are not in the layer/module
#   variant list. Any read-side mechanism story is currently untestable.
# - The S-space lens uses each tensor's own SVD. Catalog spec also wanted
#   base-W SVD (`U0, S0, V0h = svd(W_base); dS = U0.T @ dW @ V0h`) which
#   answers a different question.
# - Adapter-parameterization-specific decompositions (DoRA mag/dir, DeLoRA
#   lambda/dir, OFT rotation, IA3 gates) are not in the S-space variant set.
# - Sufficiency claims from keep tests need a norm-matched random control
#   (T7/lm has `random_norm_matched_full`, T8/S-space does not).

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
OUT_DIR = ROOT / "ablation_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load
#
# `s_space` = T8 = parameterization_ablation. `lm` = T7 = layer_module_ablation.

# %%
s_space = pl.read_csv(ROOT / "parameterization_ablation" / "summary.csv")
lm = pl.read_csv(ROOT / "layer_module_ablation" / "summary.csv")

assert int(s_space["max_idx_symmetric_diff"].max()) == 0, "S-space same-row check failed"
assert int(lm["max_idx_symmetric_diff"].max()) == 0, "layer/module same-row check failed"
assert int(lm["max_claim_idx_symmetric_diff"].max()) == 0, "layer/module sycophancy same-row check failed"
logger.info(f"loaded s_space={s_space.height} layer/module={lm.height}")


# %% [markdown]
# ## Helpers
#
# `retained` is the single quantitative measure: ablated dd_delta divided by
# full dd_delta on the same rows. ia3 is excluded from joint plots because
# its full dd_delta = +0.033 is at the noise floor; reported separately.

# %%
def with_retained(df: pl.DataFrame, full_variant: str = "full_dW") -> pl.DataFrame:
    full = (
        df.filter((pl.col("variant" if "variant" in df.columns else "component") == full_variant) & (pl.col("coeff") == 1.0))
        .select("adapter", pl.col("dd_delta").alias("full_dd"))
    )
    return df.filter(pl.col("coeff") == 1.0).join(full, on="adapter").with_columns(
        (pl.col("dd_delta") / pl.col("full_dd")).alias("retained")
    )


REAL_ADAPTERS = [a for a in ADAPTERS if a != "ia3"]


# %% [markdown]
# ## Lens 1: S-space (each tensor's own SVD)
#
# Asks: is dW low-rank in the basis it picked? Does keeping the top-K singular
# components of each tensor reproduce the full effect, and does dropping them
# remove it?

# %%
s = s_space.rename({"component": "variant"}) if "component" in s_space.columns else s_space
sR = with_retained(s)

s_table = (
    sR.filter(pl.col("variant") != "full_dW")
    .select("adapter", "variant", "keep_or_drop", "energy_frac", "dd_delta", "full_dd", "retained")
    .sort(["adapter", "retained"], descending=[False, True])
)
print("\nS-space retained ratio per (adapter, crop)")
print(tabulate(s_table.to_pandas(), headers="keys", tablefmt="pipe", floatfmt="+.3f", showindex=False))


# %% [markdown]
# ### S-space rank concentration
#
# Two ratios per adapter at the top-25%-by-index crop:
# - `retained_keep_top25` = dd_delta(top 25% S kept) / dd_delta(full)
# - `retained_drop_top25` = dd_delta(top 25% S dropped) / dd_delta(full)
#
# If keep is near 1 and drop is near 0, the top-25% slice is sufficient and
# the rest is redundant in this basis.

# %%
top25_keep = sR.filter(pl.col("variant") == "top_25pct_S").select("adapter", pl.col("retained").alias("retained_keep_top25"), pl.col("energy_frac").alias("energy_keep_top25"))
top25_drop = sR.filter(pl.col("variant") == "residual_not_top_25pct_S").select("adapter", pl.col("retained").alias("retained_drop_top25"))
top25 = top25_keep.join(top25_drop, on="adapter").sort("adapter")
print("\nTop-25%-S concentration")
print(tabulate(top25.to_pandas(), headers="keys", tablefmt="pipe", floatfmt="+.3f", showindex=False))


# %% [markdown]
# ### S-space figure: retained vs energy_frac
#
# x = fraction of dW Frobenius energy retained by the crop; y = retained
# steering ratio. Diagonal = behavior tracks energy. Above diagonal = the
# crop punches above its energy share.

# %%
fig, ax = plt.subplots(figsize=(7, 5))
for adapter in ADAPTERS:
    sub = sR.filter((pl.col("adapter") == adapter) & (pl.col("variant") != "full_dW"))
    ax.scatter(sub["energy_frac"], sub["retained"], label=adapter, alpha=0.7, s=40)
xs = [0, 1]
ax.plot(xs, xs, color="k", lw=0.5, alpha=0.3, linestyle="--", label="energy = retained")
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.axvline(0, color="k", lw=0.5, alpha=0.3)
ax.set_xlabel("energy_frac (fraction of dW Frobenius energy in crop)")
ax.set_ylabel("retained dd_delta / full")
ax.set_title("Lens 1: S-space crop, energy vs retained behavior")
ax.legend(fontsize=8, loc="best")
fig.tight_layout()
fig_path = OUT_DIR / "lens1_s_space.png"
fig.savefig(fig_path, dpi=120)
logger.info(f"saved {fig_path}")


# %% [markdown]
# ### S-space caveat: norm shrinkage
#
# Cropping reduces dW Frobenius norm. Model is nonlinear in alpha. So a small
# kept dW that scores well could be the right direction, OR could just be a
# smaller effective coefficient working better. To rule that out we would
# need a `random_norm_matched_full` control in S-space crops; T7 has it,
# this experiment does not. Sufficiency claims (keep alone steers) are
# weaker until that control is added. Necessity claims (drop kills it) are
# unaffected.


# %% [markdown]
# ## Lens 2: layer index
#
# Asks: is dW localized in depth? Two probes from the layer/module run:
# - single_layer_keep: zero everything except this one layer's dW.
# - leave_one_layer_out: zero this layer's dW, keep the rest.

# %%
lmR = with_retained(lm)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
single = lmR.filter((pl.col("variant") == "single_layer_keep"))
loo = lmR.filter((pl.col("variant") == "leave_one_layer_out"))

for ax, df, title, ylabel in [
    (axes[0], single, "single_layer_keep (sufficiency)", "retained dd_delta / full"),
    (axes[1], loo, "leave_one_layer_out (necessity)", None),
]:
    for adapter in ADAPTERS:
        sub = df.filter(pl.col("adapter") == adapter).with_columns(
            pl.col("layer_or_block").cast(pl.Int64).alias("layer")
        ).sort("layer")
        if sub.height == 0:
            continue
        ax.plot(sub["layer"], sub["retained"], marker="o", label=adapter, alpha=0.8)
    ax.axhline(1.0, color="k", lw=0.5, alpha=0.3, linestyle="--")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("layer index")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
fig.suptitle("Lens 2: layer-index ablations")
fig.tight_layout()
fig_path = OUT_DIR / "lens2_layer_index.png"
fig.savefig(fig_path, dpi=120)
logger.info(f"saved {fig_path}")

# %% [markdown]
# Reading: in `single_layer_keep` (left), retained close to 1 means that one
# layer alone reproduces the full effect (concentrated). In
# `leave_one_layer_out` (right), retained < 1 means dropping that layer
# matters. A flat curve near 1 in LOO means no single layer is necessary
# (distributed).


# %% [markdown]
# ## Lens 3: module family
#
# Asks: which module families carry the dW behavior? Residual writes
# (`o_proj`, `down_proj`), attention as a block, mlp as a block, attention
# o_proj alone, mlp down_proj alone. ia3 has no o_proj, so o_proj-dependent
# variants are unavailable for ia3.

# %%
module_variants = ["full_dW", "residual_write_only", "attention_only", "mlp_only",
                   "attn_o_proj_only", "mlp_down_proj_only", "layers_8_21_only",
                   "random_norm_matched_full", "zero"]
module = lmR.filter(pl.col("variant").is_in(module_variants))
pivot = module.pivot(values="retained", index="adapter", on="variant", aggregate_function="first")
cols = ["adapter"] + [v for v in module_variants if v in pivot.columns]
print("\nModule-family retained per adapter")
print(tabulate(pivot.select(cols).to_pandas(), headers="keys", tablefmt="pipe", floatfmt="+.3f", showindex=False))


# %% [markdown]
# ### Module-family figure
#
# One bar per (adapter, module_variant). Ordered to put `full_dW` and
# `random_norm_matched_full` as anchors at the ends.

# %%
order = ["zero", "random_norm_matched_full", "attention_only", "attn_o_proj_only",
         "mlp_only", "mlp_down_proj_only", "residual_write_only", "layers_8_21_only", "full_dW"]
fig, ax = plt.subplots(figsize=(10, 5))
import numpy as np
n_v = len(order)
n_a = len(ADAPTERS)
width = 0.8 / n_a
xs = np.arange(n_v)
for i, adapter in enumerate(ADAPTERS):
    sub = lmR.filter(pl.col("adapter") == adapter)
    rs = []
    for v in order:
        row = sub.filter(pl.col("variant") == v)["retained"]
        rs.append(float(row[0]) if row.len() else np.nan)
    ax.bar(xs + i * width - 0.4 + width / 2, rs, width=width, label=adapter, alpha=0.8)
ax.axhline(1.0, color="k", lw=0.5, alpha=0.3, linestyle="--")
ax.axhline(0.0, color="k", lw=0.5, alpha=0.3)
ax.set_xticks(xs)
ax.set_xticklabels(order, rotation=30, ha="right")
ax.set_ylabel("retained dd_delta / full")
ax.set_title("Lens 3: module-family ablations")
ax.legend(fontsize=8, loc="best")
fig.tight_layout()
fig_path = OUT_DIR / "lens3_module_family.png"
fig.savefig(fig_path, dpi=120)
logger.info(f"saved {fig_path}")


# %% [markdown]
# ## Joint summary
#
# One number per lens per adapter, computed from the data above:
# - `s_space_top25_keep`: retained when only top 25% of each tensor's S kept.
# - `s_space_top25_drop`: retained when top 25% of each tensor's S dropped.
# - `lm_residual_write`: retained when only o_proj and down_proj of dW kept.
# - `lm_random_norm_matched`: retained for a Frobenius-matched random dW
#   (necessity-side anchor; should be near 0 if the trained direction matters).

# %%
joint = top25.join(
    lmR.filter(pl.col("variant") == "residual_write_only").select("adapter", pl.col("retained").alias("lm_residual_write")),
    on="adapter",
).join(
    lmR.filter(pl.col("variant") == "random_norm_matched_full").select("adapter", pl.col("retained").alias("lm_random_norm_matched")),
    on="adapter",
).rename({
    "retained_keep_top25": "s_space_top25_keep",
    "retained_drop_top25": "s_space_top25_drop",
}).select("adapter", "s_space_top25_keep", "s_space_top25_drop", "lm_residual_write", "lm_random_norm_matched")
print("\nJoint summary")
print(tabulate(joint.to_pandas(), headers="keys", tablefmt="pipe", floatfmt="+.3f", showindex=False))


# %% [markdown]
# ## Caveats
#
# - Single seed, single base model (Qwen3-0.6B), single behavior pair
#   (sycophancy / honesty), single eval (daily-dilemmas base persona).
# - ia3 full dd_delta = +0.033 is at the noise floor, so its retained ratios
#   are unstable. Reported separately in the tables.
# - S-space lens uses each tensor's own SVD. Other lenses on the same dW
#   (base-W SVD, shared cross-adapter SVD, activation-PCA) are different
#   experiments and answer different questions.
# - S-space lens has no `random_norm_matched_full` control. Necessity claims
#   (drop kills behavior) are unaffected. Sufficiency claims (small kept
#   slice still steers) need a Frobenius-matched random S-crop control.
# - Layer/module lens covers write-side modules (o_proj, down_proj) and
#   block attention/mlp. Read-side per-module variants (q/k/v/up/gate
#   alone) are not yet implemented, so any read-side mechanism story is
#   currently untestable here.
# - Adapter-architecture decompositions (DoRA magnitude vs direction,
#   DeLoRA lambda vs direction, OFT rotation, IA3 gates) are not in either
#   variant list; they would constrain the dW the optimizer can produce and
#   sit between Goal A (where does dW live) and Goal B (predict dW from
#   pretrained W and hs_diff without training). See docs/human_journal.md
#   for the full hypothesis list.
