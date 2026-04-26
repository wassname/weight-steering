"""One-question notebook: where does the steering signal become simple?

Question:
    Does the steering-induced activation difference on task,
    Δa = a_{alpha=+1} - a_{alpha=-1}, concentrate in task-derived subspaces
    more than in pretrained structural bases?

Why this notebook exists:
    The older notebook mixes several geometry questions: dW magnitude,
    rank, linearity, and activation hooks. This one asks one falsifiable
    question and tries to reach one of three concrete conclusions:

      1. Δa concentrates in task-derived hidden subspaces
         (TaskDiff / Suppressed / Stenographic).
      2. Δa concentrates in pretrained readable structural bases
         (lm_head read / logits null).
      3. Δa is diffuse and none of the simple geometry stories survive.

Method:
    - Use the 32 sycophancy topics already in ws.data as probe prompts.
    - Split prompts into fit/eval halves so TaskDiff concentration is not
      tautological.
    - Capture block outputs at the last token under alpha=+1 and alpha=-1.
    - Build 4 candidate subspaces:
        * TaskDiff(layer): PCA of Δa on fit prompts, per layer
        * Suppressed(global): PCA of turnover magnitudes across layers
        * Stenographic(layer): TaskDiff(layer) ∩ Suppressed(global)
        * Structural controls: lm_head_read and logits_null
    - Score concentration on eval prompts using:

          energy_ratio = E[||P_V Δa||^2] / E[||Δa||^2] / (k / d)

      where k is subspace rank and d is residual dimension. Ratio > 1 means
      more concentration than a matched random k-dim subspace.

Interpretation:
    - mean_ratio(TaskDiff / Suppressed / Stenographic) >> 1 and
      mean_ratio(structural controls) ~ 1:
        steering signal is task-specific, not just pretrained geometry.
    - structural controls >> task-derived:
        steering mostly rides readable pretrained directions.
    - everything ~ 1:
        no simple low-dim subspace story.
"""

# %%
import os
import sys
from pathlib import Path

import polars as pl
import torch
from baukit import TraceDict
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from ws.data import SYCOPHANCY_TOPICS
from ws.diff import load_diff
from ws.steer import weight_steer

# token-efficient logging: plain message format, tqdm-safe; verbose to file
logger.remove()
logger.add(sys.stdout, level=os.environ.get("LOG_LEVEL", "INFO"), colorize=False, format="{message}")
Path("logs").mkdir(exist_ok=True)
logger.add("logs/analyze_diff_v2.verbose.log", level="DEBUG",
           format="{time} | {level} | {name}:{function}:{line} - {message}")

torch.set_grad_enabled(False)


# %% [markdown]
# ## Config
#
# One question, one metric, one prompt split.

# %%
MODEL_ID = "Qwen/Qwen3-0.6B"
W_PATH = Path("out/sycophancy/lora/w.pt")
OUT_DIR = Path("out/sycophancy/lora")

PCS = 8
MIN_OVERLAP = 0.5

PROBE_PROMPTS = [
    f"I think {claim}. Do you agree? Answer Yes or No."
    for claim, _q in SYCOPHANCY_TOPICS
]
FIT_PROMPTS = PROBE_PROMPTS[: len(PROBE_PROMPTS) // 2]
EVAL_PROMPTS = PROBE_PROMPTS[len(PROBE_PROMPTS) // 2 :]


# %% [markdown]
# ## Load model and diff

# %%
w = load_diff(W_PATH)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

n_layers = model.config.num_hidden_layers
HOOKS = [f"model.layers.{i}" for i in range(n_layers)]
lm_head_W = model.state_dict().get("lm_head.weight")
if lm_head_W is None:
    lm_head_W = model.state_dict()["model.embed_tokens.weight"]
lm_head_W = lm_head_W.float().cpu()

logger.info(f"loaded model={MODEL_ID} layers={n_layers} hooks={len(HOOKS)}")
logger.info(f"loaded w with {len(w)} tensors from {W_PATH}")


# %% [markdown]
# ## Helpers

# %%
def orthonormalize(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0 or matrix.shape[1] == 0:
        return matrix.new_zeros(matrix.shape[0], 0)
    q, _r = torch.linalg.qr(matrix, mode="reduced")
    return q


def pca_basis(samples: torch.Tensor, k: int) -> torch.Tensor:
    """samples: [n, d] -> orthonormal basis [d, k_eff]."""
    centered = samples - samples.mean(dim=0, keepdim=True)
    if centered.shape[0] <= 1:
        return centered.new_zeros(centered.shape[1], 0)
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    k_eff = min(k, vh.shape[0])
    return vh[:k_eff].T.contiguous()


def structural_bases(lm_head: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
    _u, _s, vh = torch.linalg.svd(lm_head, full_matrices=False)
    return {
        "lm_head_read": vh[:k].T.contiguous(),
        "logits_null": vh[-k:].T.contiguous(),
    }


def intersect_bases(a: torch.Tensor, b: torch.Tensor, min_overlap: float) -> torch.Tensor:
    if a.shape[1] == 0 or b.shape[1] == 0:
        return a.new_zeros(a.shape[0], 0)
    u, s, vh = torch.linalg.svd(a.T @ b, full_matrices=False)
    keep = s >= min_overlap
    if not keep.any():
        return a.new_zeros(a.shape[0], 0)
    va = a @ u[:, keep]
    vb = b @ vh.T[:, keep]
    return orthonormalize((va + vb) / 2)


def concentration_stats(samples: torch.Tensor, basis: torch.Tensor) -> dict[str, float]:
    d = samples.shape[1]
    k = basis.shape[1]
    total = samples.pow(2).sum(dim=1)
    if k == 0:
        return {
            "rank": 0,
            "energy": 0.0,
            "null": 0.0,
            "ratio": 0.0,
        }
    proj = samples @ basis
    energy = (proj.pow(2).sum(dim=1) / (total + 1e-12)).mean().item()
    null = k / d
    return {
        "rank": int(k),
        "energy": float(energy),
        "null": float(null),
        "ratio": float(energy / null),
    }


def capture_block_outputs(prompts: list[str], alpha: float) -> torch.Tensor:
    """Return [layers, batch, d_model] last-token block outputs."""
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(model.device)
    seq_idx = enc.attention_mask.sum(dim=-1) - 1

    ctx = weight_steer(model, w, alpha) if alpha != 0 else torch.no_grad()
    with ctx:
        with TraceDict(model, HOOKS, retain_output=True) as ret:
            _ = model(**enc)

    rows = []
    for hook in HOOKS:
        x = ret[hook].output
        if isinstance(x, tuple):
            x = x[0]
        batch, _seq, d_model = x.shape
        gather_idx = seq_idx.view(batch, 1, 1).expand(batch, 1, d_model)
        last_tok = x.gather(1, gather_idx).squeeze(1).float().cpu()
        rows.append(last_tok)
    return torch.stack(rows, dim=0)


def suppressed_features(acts: torch.Tensor) -> torch.Tensor:
    """acts: [layers, batch, d] -> turnover features [batch, d]."""
    mag = acts.abs().permute(1, 0, 2)
    delta = mag[:, 1:] - mag[:, :-1]
    increases = torch.relu(delta).sum(dim=1)
    decreases = torch.relu(-delta).sum(dim=1)
    return torch.minimum(increases, decreases)


# %% [markdown]
# ## Capture fit/eval activations under alpha=+1 and alpha=-1

# %%
pos_fit = capture_block_outputs(FIT_PROMPTS, alpha=+1.0)
neg_fit = capture_block_outputs(FIT_PROMPTS, alpha=-1.0)
pos_eval = capture_block_outputs(EVAL_PROMPTS, alpha=+1.0)
neg_eval = capture_block_outputs(EVAL_PROMPTS, alpha=-1.0)

delta_fit = pos_fit - neg_fit
delta_eval = pos_eval - neg_eval

logger.info(
    "captured fit/eval activations: fit={} eval={} shape={}",
    len(FIT_PROMPTS),
    len(EVAL_PROMPTS),
    tuple(delta_fit.shape),
)


# %% [markdown]
# ## Fit candidate subspaces on the fit split
#
# We compare task-derived candidates against structural controls.

# %%
taskdiff_bases = [pca_basis(delta_fit[layer], PCS) for layer in range(n_layers)]

suppressed_fit = 0.5 * (suppressed_features(pos_fit) + suppressed_features(neg_fit))
suppressed_basis = pca_basis(suppressed_fit, PCS)

structural = structural_bases(lm_head_W, PCS)
stenographic_bases = [
    intersect_bases(taskdiff_bases[layer], suppressed_basis, min_overlap=MIN_OVERLAP)
    for layer in range(n_layers)
]

logger.info(
    "basis ranks: suppressed={} lm_head_read={} logits_null={}",
    suppressed_basis.shape[1],
    structural["lm_head_read"].shape[1],
    structural["logits_null"].shape[1],
)


# %% [markdown]
# ## Score concentration on held-out prompts
#
# Main metric:
#
#   energy_ratio = E[||P_V Δa||²] / E[||Δa||²] / (k / d)
#
# Ratio > 1 means more concentration than a matched random k-dim subspace.

# %%
rows = []
for layer in range(n_layers):
    x = delta_eval[layer]
    candidates = {
        "taskdiff": taskdiff_bases[layer],
        "suppressed": suppressed_basis,
        "stenographic": stenographic_bases[layer],
        "lm_head_read": structural["lm_head_read"],
        "logits_null": structural["logits_null"],
    }
    for name, basis in candidates.items():
        stats = concentration_stats(x, basis)
        rows.append({
            "layer": layer,
            "subspace": name,
            **stats,
        })

df = pl.DataFrame(rows)

summary = (
    df.group_by("subspace")
    .agg(
        pl.col("ratio").mean().alias("mean_ratio"),
        pl.col("ratio").max().alias("max_ratio"),
        pl.col("layer").sort_by("ratio").last().alias("peak_layer"),
        pl.col("energy").mean().alias("mean_energy"),
        pl.col("rank").mean().alias("mean_rank"),
    )
    .sort("mean_ratio", descending=True)
)

print("\nconcentration summary on held-out prompts")
print(
    "SHOULD: if task-derived subspaces are real, taskdiff / suppressed / stenographic "
    "have mean_ratio >> 1 and beat structural controls. ELSE: if lm_head_read wins, "
    "the signal is already readable; if everything ~= 1, the geometry story is weak."
)
print(
    tabulate(
        summary.to_pandas(),
        tablefmt="tsv",
        headers="keys",
        floatfmt="+.3f",
        showindex=False,
    )
)

print("\nper-layer table")
print(
    tabulate(
        df.sort(["subspace", "layer"]).to_pandas(),
        tablefmt="tsv",
        headers="keys",
        floatfmt="+.3f",
        showindex=False,
    )
)


# %% [markdown]
# ## Decision rule
#
# Read the summary table as a model selection result:
#
# - task-derived >> structural:
#     the steering signal is task-specific and hidden / dynamic.
# - structural >> task-derived:
#     the steering mostly rides pretrained readable axes.
# - all near 1:
#     the signal is diffuse and this basis story is probably wrong.
#
# If task-derived wins, *then* it becomes worth doing stage-2 mechanism tests
# like rotation-vs-gain fits or stage-3 intervention tests like LEACE.

# %%
df.write_csv(OUT_DIR / "analyze_diff_v2_concentration_per_layer.csv")
summary.write_csv(OUT_DIR / "analyze_diff_v2_concentration_summary.csv")
logger.info("saved v2 concentration tables to {}", OUT_DIR)


# %% [markdown]
# ## Stage-1.5: principal angles between TaskDiff(layer) and lm_head_read
#
# Concentration says TaskDiff captures most Delta-a energy and lm_head_read does
# not. This is sufficient evidence that the signal is not the readout direction,
# but principal angles make the geometric relationship explicit.
#
# For two rank-k orthonormal bases A, B in R^d, the principal cosines are the
# singular values of A.T @ B. All near 1 means the subspaces nearly coincide;
# all near 0 means they are orthogonal.

# %%
angle_rows = []
lm_basis = structural["lm_head_read"]
for layer in range(n_layers):
    A = taskdiff_bases[layer]
    if A.shape[1] == 0:
        continue
    cos_angles = torch.linalg.svdvals(A.T @ lm_basis).clamp(0, 1)
    angle_rows.append({
        "layer": layer,
        "max_cos": float(cos_angles.max()),
        "mean_cos": float(cos_angles.mean()),
        "min_cos": float(cos_angles.min()),
    })

angle_df = pl.DataFrame(angle_rows)
print("\nprincipal cosines between TaskDiff(layer) and lm_head_read")
print(
    "SHOULD: if TaskDiff is largely orthogonal to readout, mean_cos << 1 and "
    "max_cos < 0.7 in active layers (>=8). ELSE TaskDiff is a relabel of readout."
)
print(
    tabulate(
        angle_df.to_pandas(),
        tablefmt="tsv",
        headers="keys",
        floatfmt="+.3f",
        showindex=False,
    )
)
angle_df.write_csv(OUT_DIR / "analyze_diff_v2_taskdiff_vs_lmhead_angles.csv")


# %% [markdown]
# ## Final summary (BLUF for log readers)
#
# Last ~30 lines of stdout: cue emoji + main metric, then argv/out paths, then
# a tight TSV result table for a downstream LLM/agent to read.

# %%
active = df.filter(pl.col("layer") >= 8)
active_summary = (
    active.group_by("subspace")
    .agg(
        pl.col("ratio").mean().alias("mean_ratio_active"),
        pl.col("ratio").max().alias("max_ratio"),
        pl.col("layer").sort_by("ratio").last().alias("peak_layer"),
    )
    .sort("mean_ratio_active", descending=True)
)
td_mean = active_summary.filter(pl.col("subspace") == "taskdiff")["mean_ratio_active"][0]
lm_mean = active_summary.filter(pl.col("subspace") == "lm_head_read")["mean_ratio_active"][0]
ratio_td_lm = td_mean / lm_mean if lm_mean > 0 else float("inf")
angles_active = angle_df.filter(pl.col("layer") >= 8)
max_cos_active = angles_active["max_cos"].max() if angles_active.height else float("nan")

cue = "🟢" if (td_mean >= 5.0 and ratio_td_lm >= 3.0) else ("🟡" if td_mean >= 2.0 else "🔴")

print()
print(f"out: {OUT_DIR}/analyze_diff_v2_concentration_summary.csv")
print(f"argv: nbs/analyze_diff_v2.py model={MODEL_ID} w={W_PATH} pcs={PCS} min_overlap={MIN_OVERLAP}")
print(
    f"main metric: {cue} taskdiff_active_mean={td_mean:.2f} | "
    f"lm_head_read_active_mean={lm_mean:.2f} | "
    f"taskdiff/lm_head_read={ratio_td_lm:.2f} | "
    f"max_cos(TaskDiff,lm_head_read)_active={max_cos_active:.2f}"
)
print()
print(
    "SHOULD: cue=🟢 means taskdiff dominates lm_head_read by >=3x AND active-mean>=5; "
    "🟡 means taskdiff active-mean>=2 (weak); 🔴 means signal is diffuse or rides readout. "
    "max_cos<0.7 confirms TaskDiff is geometrically distinct from the unembedding readout."
)
print(
    tabulate(
        active_summary.to_pandas(),
        headers=["subspace", "mean_ratio↑", "max_ratio", "peak_layer"],
        tablefmt="tsv",
        floatfmt="+.2f",
        showindex=False,
    )
)