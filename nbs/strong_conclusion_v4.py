# %% [markdown]
# # Strong conclusion notebook: held-out label recovery
#
# **Question.** Which A-side recipe, built without seeing the trained LoRA, best predicts where the LoRA steering signal lives?
#
# **Single method.** Treat the trained LoRA activation difference as a held-out label:
#
# $$
# R_{m,\ell}=\frac{\mathbb{E}\|P_{V_{m,\ell}}\Delta h^B_\ell\|^2/\|\Delta h^B_\ell\|^2}{k/d}
# $$
#
# where $m$ is an A-side recipe, $V_{m,\ell}$ is its rank-$k$ basis at layer $\ell$, and $\Delta h^B_\ell$ is the LoRA-induced activation difference on held-out plain probes.
#
# Success is not "a curve looks high". Success means one A-side recipe has:
#
# - concentration $R \gg 1$ against the random-subspace null,
# - a positive paired log-margin over the next-best A-side recipe across LoRA layers,
# - a nontrivial fraction of the LoRA-fitted ceiling.
#
# This follows the plotting discipline in Wendler et al. (few phase curves), Gromov et al. (one geometry statistic), and Feucht et al. (causal/evidence score first, diagnostics second).

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tabulate import tabulate


# %%
ROOT = Path.cwd()
IN_CSV = ROOT / "out/sycophancy/lora/v3_per_layer.csv"
IN_OVERLAP = ROOT / "out/sycophancy/lora/v3_recipe_overlap.csv"
OUT_DIR = ROOT / "out/sycophancy/lora"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LORA_LAYERS = range(8, 22)
PCS = 8
PHASE_NAMES = ["TaskDiff_lora_ceiling", "write_not_read", "TaskDiff_contrast"]
A_NAMES = ["write_not_read", "TaskDiff_contrast", "lm_head_read", "suppressed", "logits_null"]
RANDOM_NULL = 1.0
BOOT = 20_000
RNG = np.random.default_rng(0)


# %% [markdown]
# ## Load v3 held-out recovery scores
#
# `v3_per_layer.csv` already enforces the A/B split:
#
# - A-side recipes use only pretrained weights and base-model activations.
# - B-side labels come from the trained LoRA, scored on held-out plain prompts.
# - The ceiling row uses LoRA FIT activations to predict LoRA EVAL activations and is not a deployable recipe.

# %%
df = pl.read_csv(IN_CSV)
active = df.filter(pl.col("layer").is_in(list(LORA_LAYERS)))

required = set(PHASE_NAMES + A_NAMES)
observed = set(active["subspace"].to_list())
missing = required - observed
if missing:
    raise ValueError(f"missing subspaces in {IN_CSV}: {sorted(missing)}")

wide = active.select("layer", "subspace", "conc_in_B").pivot(
    index="layer", on="subspace", values="conc_in_B"
).sort("layer")
layers = wide["layer"].to_numpy()


# %%
def bootstrap_ci(values: np.ndarray, *, boot: int = BOOT) -> tuple[float, float, float]:
    idx = RNG.integers(0, len(values), size=(boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_log_margin(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    margins = np.log2(a) - np.log2(b)
    mean, lo, hi = bootstrap_ci(margins)
    p_positive = float((margins > 0).mean())
    return mean, lo, hi, p_positive


@dataclass(frozen=True)
class RecipeSummary:
    subspace: str
    kind: str
    mean_conc: float
    ci_low: float
    ci_high: float
    pct_ceiling: float
    mean_z: float
    layer_wins: int


# %% [markdown]
# ## Strong-conclusion statistics
#
# These are designed to fail visibly if the apparent winner is just noise:
#
# - Paired log-margin: layerwise $\log_2 R_\text{winner}-\log_2 R_\text{runner-up}$.
# - Bootstrap CI over LoRA layers.
# - Layer-win count among A-side recipes.
# - Fraction of the LoRA-derived ceiling.

# %%
ceiling_values = wide["TaskDiff_lora_ceiling"].to_numpy()
ceiling_mean = float(ceiling_values.mean())

summaries: list[RecipeSummary] = []
for name in ["TaskDiff_lora_ceiling", *A_NAMES]:
    values = wide[name].to_numpy()
    mean, lo, hi = bootstrap_ci(values)
    kind = "ceiling" if name == "TaskDiff_lora_ceiling" else "A-hypothesis"
    mean_z = float(active.filter(pl.col("subspace") == name)["z"].mean())
    if name in A_NAMES:
        layer_wins = int(
            sum(
                wide[name].to_numpy()[i]
                == max(wide[a].to_numpy()[i] for a in A_NAMES)
                for i in range(wide.height)
            )
        )
    else:
        layer_wins = 0
    summaries.append(
        RecipeSummary(
            subspace=name,
            kind=kind,
            mean_conc=mean,
            ci_low=lo,
            ci_high=hi,
            pct_ceiling=100 * mean / ceiling_mean,
            mean_z=mean_z,
            layer_wins=layer_wins,
        )
    )

summary_df = pl.DataFrame([s.__dict__ for s in summaries]).sort("mean_conc", descending=True)

best_a = summary_df.filter(pl.col("kind") == "A-hypothesis")["subspace"][0]
runner_up = summary_df.filter(pl.col("kind") == "A-hypothesis")["subspace"][1]
margin_mean, margin_lo, margin_hi, p_layer_positive = paired_log_margin(
    wide[best_a].to_numpy(), wide[runner_up].to_numpy()
)
layer_margins = np.log2(wide[best_a].to_numpy()) - np.log2(wide[runner_up].to_numpy())
reversal_layers = layers[layer_margins < 0]
reversal_text = ", ".join(str(int(ℓ)) for ℓ in reversal_layers)

best_pct_ceiling = float(summary_df.filter(pl.col("subspace") == best_a)["pct_ceiling"][0])
best_mean_z = float(summary_df.filter(pl.col("subspace") == best_a)["mean_z"][0])
best_layer_wins = int(summary_df.filter(pl.col("subspace") == best_a)["layer_wins"][0])

claim = (
    f"{best_a} is the strongest tested A-side recipe: {best_pct_ceiling:.0f}% of ceiling, "
    f"mean z={best_mean_z:.1f}, wins {best_layer_wins}/{len(list(LORA_LAYERS))} LoRA layers, "
    f"paired log2 margin over {runner_up} = {margin_mean:+.2f} "
    f"[{margin_lo:+.2f}, {margin_hi:+.2f}], with reversals on {len(reversal_layers)}/14 layers."
)
print("BLUF:", claim)
print(tabulate(summary_df.to_pandas(), headers="keys", tablefmt="github", floatfmt="+.2f"))


# %% [markdown]
# ## Complementarity diagnostic
#
# A modest win can mean two different things:
#
# 1. `write_not_read` and `TaskDiff_contrast` recover the same subspace, so the winner is fragile.
# 2. They recover mostly different parts of the LoRA label, so the winner is a real but partial route.
#
# The distinguishing check is the principal angle between the two bases and the energy recovered by their rank-16 union.

# %%
overlap_summary_path = OUT_DIR / "v4_overlap_summary.tsv"
if IN_OVERLAP.exists():
    overlap = pl.read_csv(IN_OVERLAP).filter(pl.col("layer").is_in(list(LORA_LAYERS)))
    overlap = overlap.with_columns(
        union_vs_best_energy_log2=(pl.col("union_vs_best_log2") + np.log2(pl.col("union_rank") / PCS))
    )
    overlap_summary = overlap.select(
        pl.col("mean_principal_angle_deg").mean().alias("mean_angle_deg"),
        pl.col("mean_principal_angle_deg").min().alias("min_angle_deg"),
        pl.col("mean_principal_angle_deg").max().alias("max_angle_deg"),
        pl.col("union_rank").mean().alias("mean_union_rank"),
        pl.col("union_vs_best_energy_log2").mean().alias("mean_union_vs_best_energy_log2"),
        (pl.col("union_vs_best_energy_log2") > 0).sum().alias("union_energy_beats_best_layers"),
    )
    overlap_summary.write_csv(overlap_summary_path, separator="\t")
    mean_angle = float(overlap_summary["mean_angle_deg"][0])
    mean_union_gain = float(overlap_summary["mean_union_vs_best_energy_log2"][0])
    union_win_layers = int(overlap_summary["union_energy_beats_best_layers"][0])
    print("overlap diagnostic:")
    print(tabulate(overlap_summary.to_pandas(), headers="keys", tablefmt="github", floatfmt="+.2f"))
else:
    overlap_summary = None
    mean_angle = float("nan")
    mean_union_gain = float("nan")
    union_win_layers = 0
    print(f"overlap diagnostic skipped: missing {IN_OVERLAP}")


# %% [markdown]
# ## Main figure
#
# Panel A is the phase plot: random null, best A-side recipe, runner-up, and LoRA ceiling.
# Panel B is the sorted scorecard with uncertainty and fraction-of-ceiling labels.

# %%
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 240,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})

colors = {
    "TaskDiff_lora_ceiling": "#1f77b4",
    "write_not_read": "#ff7f0e",
    "TaskDiff_contrast": "#d62728",
    "lm_head_read": "#9467bd",
    "suppressed": "#2ca02c",
    "logits_null": "#8c564b",
}
labels = {
    "TaskDiff_lora_ceiling": "LoRA-fitted ceiling",
    "write_not_read": "W-only write-not-read",
    "TaskDiff_contrast": "base prompt contrast",
    "lm_head_read": "lm_head read",
    "suppressed": "suppressed turnover",
    "logits_null": "lm_head null",
}

fig, (ax_phase, ax_margin, ax_bar) = plt.subplots(
    1, 3, figsize=(15.5, 4.9), gridspec_kw={"width_ratios": [1.25, 0.75, 1.0]}
)

ax_phase.axhline(RANDOM_NULL, color="black", linestyle="--", linewidth=1.0, label="random null")
for name in ["TaskDiff_lora_ceiling", best_a, runner_up]:
    linestyle = "--" if name == "TaskDiff_lora_ceiling" else "-"
    linewidth = 2.4 if name in {best_a, "TaskDiff_lora_ceiling"} else 1.9
    ax_phase.plot(
        layers,
        wide[name].to_numpy(),
        marker="o",
        linewidth=linewidth,
        linestyle=linestyle,
        color=colors[name],
        label=labels[name],
    )
ax_phase.set_yscale("log")
ax_phase.set_xlabel("layer ℓ (LoRA layers only)")
ax_phase.set_ylabel("held-out label recovery R↑ (random = 1)")
ax_phase.set_title("A. Strongest tested recipe tracks the LoRA label")
ax_phase.grid(alpha=0.28, which="both")
ax_phase.legend(loc="upper center", frameon=True)
ax_phase.text(
    0.02,
    0.03,
    f"{best_a} vs {runner_up}: log2 margin {margin_mean:+.2f}\n95% CI [{margin_lo:+.2f}, {margin_hi:+.2f}]",
    transform=ax_phase.transAxes,
    ha="left",
    va="bottom",
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
)

margin_colors = np.where(layer_margins >= 0, colors[best_a], colors[runner_up])
ax_margin.axhline(0, color="black", linewidth=1.0)
ax_margin.bar(layers, layer_margins, color=margin_colors, alpha=0.9)
ax_margin.set_xlabel("layer ℓ")
ax_margin.set_ylabel(f"log2({best_a} / {runner_up})")
ax_margin.set_title("B. Paired layer margin")
ax_margin.grid(axis="y", alpha=0.28)
ax_margin.text(
    0.02,
    0.03,
    f"positive on {best_layer_wins}/14\nreversals: {reversal_text}",
    transform=ax_margin.transAxes,
    ha="left",
    va="bottom",
    fontsize=8,
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
)

bar_df = summary_df.to_pandas()
y = np.arange(len(bar_df))
bar_colors = [colors[s] for s in bar_df["subspace"]]
bar_width = bar_df["mean_conc"].to_numpy()
xerr = np.vstack([
    bar_df["mean_conc"].to_numpy() - bar_df["ci_low"].to_numpy(),
    bar_df["ci_high"].to_numpy() - bar_df["mean_conc"].to_numpy(),
])
ax_bar.barh(y, bar_width, xerr=xerr, color=bar_colors, alpha=0.88, capsize=3)
ax_bar.axvline(RANDOM_NULL, color="black", linestyle="--", linewidth=1.0)
ax_bar.set_yticks(y, [labels[s] for s in bar_df["subspace"]])
ax_bar.invert_yaxis()
ax_bar.set_xlabel("mean recovery R over layers 8..21")
ax_bar.set_title("C. Scorecard with layer bootstrap CI")
ax_bar.grid(axis="x", alpha=0.28)
for yi, row in enumerate(bar_df.itertuples(index=False)):
    if row.kind == "ceiling":
        suffix = "ceiling"
    else:
        suffix = f"{row.pct_ceiling:.0f}% ceil, {row.layer_wins}/14 wins"
    ax_bar.text(row.ci_high + 0.3, yi, suffix, va="center", fontsize=8)

fig.suptitle("Qwen3-0.6B sycophancy LoRA: held-out label recovery, not spaghetti", y=1.02, fontsize=14)
fig.tight_layout()
main_png = OUT_DIR / "v4_strong_conclusion_main.png"
main_pdf = OUT_DIR / "v4_strong_conclusion_main.pdf"
fig.savefig(main_png, bbox_inches="tight")
fig.savefig(main_pdf, bbox_inches="tight")
plt.close(fig)


# %% [markdown]
# ## Appendix figure: all candidates
#
# This keeps the full search visible without making the main claim unreadable.

# %%
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.axhline(RANDOM_NULL, color="black", linestyle="--", linewidth=1.0, label="random null")
for name in ["TaskDiff_lora_ceiling", *A_NAMES]:
    ax.plot(
        layers,
        wide[name].to_numpy(),
        marker="o",
        linewidth=2.2 if name in {best_a, "TaskDiff_lora_ceiling"} else 1.2,
        linestyle="--" if name == "TaskDiff_lora_ceiling" else "-",
        alpha=1.0 if name in {best_a, "TaskDiff_lora_ceiling", runner_up} else 0.55,
        color=colors[name],
        label=labels[name],
    )
ax.set_yscale("log")
ax.set_xlabel("layer ℓ (LoRA layers only)")
ax.set_ylabel("held-out label recovery R↑")
ax.set_title("Appendix: all A-side candidates")
ax.grid(alpha=0.28, which="both")
ax.legend(ncol=2, frameon=True)
fig.tight_layout()
appendix_png = OUT_DIR / "v4_all_candidates_appendix.png"
appendix_pdf = OUT_DIR / "v4_all_candidates_appendix.pdf"
fig.savefig(appendix_png, bbox_inches="tight")
fig.savefig(appendix_pdf, bbox_inches="tight")
plt.close(fig)


# %% [markdown]
# ## Save tables and conclusion

# %%
summary_path = OUT_DIR / "v4_strong_conclusion_summary.tsv"
margin_path = OUT_DIR / "v4_layer_margins.tsv"
conclusion_path = OUT_DIR / "v4_conclusion.md"

summary_df.write_csv(summary_path, separator="\t")

margin_df = pl.DataFrame({
    "layer": layers,
    "best_a": [best_a] * len(layers),
    "runner_up": [runner_up] * len(layers),
    "best_conc": wide[best_a].to_numpy(),
    "runner_up_conc": wide[runner_up].to_numpy(),
    "log2_margin": np.log2(wide[best_a].to_numpy()) - np.log2(wide[runner_up].to_numpy()),
    "ceiling_conc": wide["TaskDiff_lora_ceiling"].to_numpy(),
})
margin_df.write_csv(margin_path, separator="\t")

conclusion = f"""# v4 strong conclusion

## BLUF

{claim}

## What would have falsified this

- If the best A-side recipe were noise, mean recovery would be near 1 and z near 0.
- If `write_not_read` and `TaskDiff_contrast` were tied, the paired log2-margin CI would include 0 and layer wins would be split.
- If no from-scratch recipe recovered the LoRA label, every A-side row would sit near the random null and far below the ceiling.

## Actual evidence

- Best A-side recipe: `{best_a}`.
- Runner-up: `{runner_up}`.
- Paired log2 margin: {margin_mean:+.2f} [{margin_lo:+.2f}, {margin_hi:+.2f}] over layers 8..21.
- Layer wins: {best_layer_wins}/14.
- Reversal layers where `{runner_up}` beats `{best_a}`: {reversal_text or "none"}.
- Fraction of ceiling: {best_pct_ceiling:.1f}%.
- Mean z above random-subspace bootstrap null: {best_mean_z:.1f}.

## Interpretation discipline

This is an exploratory post-hoc winner among five tested A-side recipes. The result supports
"`{best_a}` is the strongest current recipe" more than "we fully found the mechanism": it still captures only {best_pct_ceiling:.1f}% of the LoRA-fitted ceiling, and `{runner_up}` wins on {len(reversal_layers)}/14 layers.

## Complementarity diagnostic

The two strongest recipes are not redundant: their mean principal angle is {mean_angle:.1f}° across LoRA layers. The rank-16 union recovers {2 ** mean_union_gain:.2f}× the energy of the better rank-8 recipe on average ({mean_union_gain:+.2f} log2 units), and beats the better individual recipe on {union_win_layers}/14 layers. This points to complementary routes rather than a single shared subspace with noisy ranking.

## Artifacts

- Main figure: `{main_png}` and `{main_pdf}`
- Appendix figure: `{appendix_png}` and `{appendix_pdf}`
- Summary table: `{summary_path}`
- Layer margins: `{margin_path}`
- Overlap summary: `{overlap_summary_path}`
"""
conclusion_path.write_text(conclusion)

print("wrote:")
for path in [main_png, main_pdf, appendix_png, appendix_pdf, summary_path, margin_path, conclusion_path]:
    print(f"  {path} ({path.stat().st_size} bytes)")