# %% [markdown]
# # v10: Wendler-style functional metrics for LoRA-induced Δh
#
# v9 used PCA-span overlap. All A-side hypotheses scored <15% of oracle, but
# adapters behaviorally steer at 2-5% overlap -- span-overlap is missing the
# load-bearing signal. This file uses the Wendler et al. 2024 ("Do Llamas Work
# in English?") methodology: logit-lens + token energy. Both are one matmul,
# no SVD, no PCA, no oracle.
#
# Token energy (their Eq. 2):
#
#     E²(h) = (v/d) * ‖Û h‖² / ‖Û Ûᵀ‖²_F
#
# where Û is the row-normalized unembedding. E² ≈ 1 means h has the typical
# projection of a token onto the readout subspace; E² ≈ 0 means h is mostly
# orthogonal to readout (functionally invisible to lm_head).
#
# Adapted to sycophancy:
# - Δh = h(α=+1) − h(α=−1) at each layer, averaged over EVAL prompts.
# - Token energy of Δh per layer: does the LoRA write into readable space?
# - Logit-lens of Δh: lm_head @ Δh, read off logit(Yes) − logit(No).
# - Re-score each candidate B by preserved_E, cap_yn, and abs_ldiff_proj after
#   projecting Δh onto B's span.

# %%
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from baukit import TraceDict
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from ws.data import SYCOPHANCY_NEG_PERSONAS, SYCOPHANCY_POS_PERSONAS, SYCOPHANCY_TOPICS
from ws.diff import load_diff
from ws.steer import weight_steer

# %%
logger.remove()
logger.add(sys.stdout, level=os.environ.get("LOG_LEVEL", "INFO"), colorize=False, format="{message}")
Path("logs").mkdir(exist_ok=True)
logger.add(
    "logs/v10_llama.verbose.log",
    level="DEBUG",
    format="{time} | {level} | {name}:{function}:{line} - {message}",
)
torch.set_grad_enabled(False)

MODEL_ID = "Qwen/Qwen3-0.6B"
ADAPTER = os.environ.get("ADAPTER", "lora")
W_PATH = Path(os.environ.get("W_PATH", f"out/sycophancy/{ADAPTER}/w.pt"))
OUT_DIR = Path(f"out/sycophancy/{ADAPTER}/v10")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PCS = 8
LORA_LAYERS = range(8, 22)

PROBE_PROMPTS = [
    f"I think {claim}. Do you agree? Answer Yes or No." for claim, _q in SYCOPHANCY_TOPICS
]
FIT = PROBE_PROMPTS[: len(PROBE_PROMPTS) // 2]
EVAL = PROBE_PROMPTS[len(PROBE_PROMPTS) // 2 :]

logger.info(f"v10 wendler: adapter={ADAPTER} W_PATH={W_PATH} OUT_DIR={OUT_DIR}")

# %%
w = load_diff(W_PATH)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
)
model.eval()
state = model.state_dict()
n_layers = model.config.num_hidden_layers
HOOKS = [f"model.layers.{i}" for i in range(n_layers)]

lm_head_W = state.get("lm_head.weight")
if lm_head_W is None:
    lm_head_W = state["model.embed_tokens.weight"]  # tied
lm_head_W = lm_head_W.float().cpu()
v_vocab, d_model = lm_head_W.shape
logger.info(f"loaded {MODEL_ID}: layers={n_layers}, d={d_model}, v={v_vocab}")

YES_ID = tok(" Yes", add_special_tokens=False).input_ids[0]
NO_ID = tok(" No", add_special_tokens=False).input_ids[0]
logger.info(f"YES_ID={YES_ID} ' Yes' | NO_ID={NO_ID} ' No'")

# Yes/No readout direction in residual space.
e_yes_minus_no = (lm_head_W[YES_ID] - lm_head_W[NO_ID]).contiguous()  # [d]


# %% [markdown]
# ## Capture Δh per layer on EVAL prompts
#
# Two flavors:
# - cumulative: residual stream output at block L (with all upstream LoRA writes)
# - block-local: post-block − pre-block at L (only what block L itself wrote)

# %%
def capture_blocks_pre_post(prompts, *, alpha=0.0):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
    seq_idx = enc.attention_mask.sum(-1) - 1
    ctx = weight_steer(model, w, alpha) if alpha != 0 else torch.no_grad()
    with ctx:
        out = model(**enc, output_hidden_states=True)
    b = enc.input_ids.shape[0]
    pre, post = [], []
    for layer in range(n_layers):
        hs_pre = out.hidden_states[layer].float().cpu()
        hs_post = out.hidden_states[layer + 1].float().cpu()
        idx = seq_idx.cpu().view(b, 1, 1).expand(b, 1, d_model)
        pre.append(hs_pre.gather(1, idx).squeeze(1))
        post.append(hs_post.gather(1, idx).squeeze(1))
    return torch.stack(pre), torch.stack(post)  # [n_layers, b, d]


def capture_blocks(prompts, *, alpha=0.0, system=None):
    if system is None:
        texts = prompts
    else:
        msgs = [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in prompts]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
    seq_idx = enc.attention_mask.sum(-1) - 1
    ctx = weight_steer(model, w, alpha) if alpha != 0 else torch.no_grad()
    with ctx, TraceDict(model, HOOKS, retain_output=True) as ret:
        _ = model(**enc)
    rows = []
    for hook in HOOKS:
        x = ret[hook].output
        if isinstance(x, tuple):
            x = x[0]
        b, _s, d = x.shape
        rows.append(x.gather(1, seq_idx.view(b, 1, 1).expand(b, 1, d)).squeeze(1).float().cpu())
    return torch.stack(rows)  # [n_layers, b, d]


logger.info("capturing pre/post block residuals at α=±1 on EVAL")
hs_pre_pos, hs_post_pos = capture_blocks_pre_post(EVAL, alpha=+1.0)
hs_pre_neg, hs_post_neg = capture_blocks_pre_post(EVAL, alpha=-1.0)

hs_diff_cumul = hs_post_pos - hs_post_neg                              # [n_layers, b, d]
hs_diff_block = (hs_post_pos - hs_pre_pos) - (hs_post_neg - hs_pre_neg)  # [n_layers, b, d]

# Mean Δh over EVAL prompts -- this is the per-layer "what does the LoRA do to the
# residual stream on average". We score this single direction per layer; the b dim
# is collapsed to 1 here.
delta_h_cumul = hs_diff_cumul.mean(dim=1)  # [n_layers, d]
delta_h_block = hs_diff_block.mean(dim=1)  # [n_layers, d]
logger.info(f"Δh cumul shape={tuple(delta_h_cumul.shape)} | block shape={tuple(delta_h_block.shape)}")

# Need FIT-half captures too for TaskDiff_contrast and TaskDiff_lora_fit candidates.
hs_pos_fit_b = capture_blocks(FIT, alpha=+1.0)
hs_neg_fit_b = capture_blocks(FIT, alpha=-1.0)
hs_diff_B_fit = hs_pos_fit_b - hs_neg_fit_b  # [n_layers, b_fit, d]

hs_persona_pos_fit = capture_blocks(FIT, system=SYCOPHANCY_POS_PERSONAS[0])
hs_persona_neg_fit = capture_blocks(FIT, system=SYCOPHANCY_NEG_PERSONAS[0])
hs_diff_A_fit = hs_persona_pos_fit - hs_persona_neg_fit  # [n_layers, b_fit, d]


# %% [markdown]
# ## Wendler quantities
#
# Token energy per Eq. 2 of the paper. The denominator ‖Û Ûᵀ‖²_F / v² is the
# mean squared cosine among token embeddings -- normalises so a generic token
# has E² ≈ 1.

# %%
U_hat = lm_head_W / lm_head_W.norm(dim=1, keepdim=True).clamp(min=1e-12)  # [v, d]
# ‖Û Ûᵀ‖²_F = sum of squared pairwise cosines. Computed efficiently as ‖ÛᵀÛ‖²_F.
UtU_fro_sq = float((U_hat.T @ U_hat).pow(2).sum())  # scalar
logger.info(f"‖Ûᵀ Û‖²_F = {UtU_fro_sq:.4g} (= ‖Û Ûᵀ‖²_F)")


def token_energy_sq(h: torch.Tensor) -> float:
    """E²(h) = (v/d) * ‖Û h‖² / ‖Û Ûᵀ‖²_F. Scalar in/out."""
    h = h.float().cpu()
    return float((v_vocab / d_model) * (U_hat @ h).pow(2).sum() / UtU_fro_sq)


def logit_diff(h: torch.Tensor) -> float:
    """(e_yes - e_no) @ h, i.e. the Yes-vs-No logit-lens score on h."""
    return float((e_yes_minus_no @ h.float().cpu()))


# Per-layer Wendler curves on Δh.
energy_cumul = np.array([token_energy_sq(delta_h_cumul[L]) for L in range(n_layers)])
energy_block = np.array([token_energy_sq(delta_h_block[L]) for L in range(n_layers)])
ldiff_cumul = np.array([logit_diff(delta_h_cumul[L]) for L in range(n_layers)])
ldiff_block = np.array([logit_diff(delta_h_block[L]) for L in range(n_layers)])

# Reference scale: token energy of clean residuals (no steering). Should be
# roughly Wendler's 0.2 in early layers, rising near final layers.
hs_clean_eval = capture_blocks(EVAL)
energy_clean = np.array([token_energy_sq(hs_clean_eval[L].mean(0)) for L in range(n_layers)])

logger.info(f"energy_cumul[8..21] = {energy_cumul[8:22].round(3)}")
logger.info(f"energy_block[8..21] = {energy_block[8:22].round(3)}")
logger.info(f"ldiff_cumul[8..21]  = {ldiff_cumul[8:22].round(2)}")
logger.info(f"ldiff_block[8..21]  = {ldiff_block[8:22].round(2)}")


# %% [markdown]
# ## Top-K decoded tokens from Δh at the peak |logit_diff| layer
#
# Sanity check: what does the LoRA's residual write decode to?
# SHOULD: " Yes"-flavored tokens (yes / agree / right / true) dominating positive
# end, and " No"-flavored tokens (no / disagree / false) on the other end.

# %%
peak_layer = int(np.argmax(np.abs(ldiff_cumul)))
peak_h = delta_h_cumul[peak_layer]
peak_logits = (lm_head_W @ peak_h.float().cpu())  # [v]
top_pos = torch.topk(peak_logits, 12)
top_neg = torch.topk(-peak_logits, 12)
logger.info(f"peak layer (|ldiff_cumul|): {peak_layer}, ldiff={ldiff_cumul[peak_layer]:.2f}")
top_pos_tokens = [tok.decode([i]) for i in top_pos.indices.tolist()]
top_neg_tokens = [tok.decode([i]) for i in top_neg.indices.tolist()]
logger.info(f"  +Δh boosts: {list(zip(top_pos_tokens, top_pos.values.round(decimals=2).tolist()))}")
logger.info(f"  -Δh boosts: {list(zip(top_neg_tokens, top_neg.values.round(decimals=2).tolist()))}")


# %% [markdown]
# ## Build 6 candidate bases + random null
#
# We re-use definitions from v9 (lm_head_read, write, TaskDiff_contrast,
# TaskDiff_lora_fit, act_oracle, w_oracle) and add a per-layer random
# orthonormal null.

# %%
def pca(samples: torch.Tensor, k: int) -> torch.Tensor:
    if samples.shape[0] <= 1:
        return samples.new_zeros(samples.shape[1], 0)
    centered = samples - samples.mean(0, keepdim=True)
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[: min(k, vh.shape[0])].T.contiguous()


def left_svd_basis(M: torch.Tensor, k: int = PCS) -> torch.Tensor:
    if M.shape[1] == 0:
        return torch.zeros(M.shape[0], 0)
    U, _s, _Vh = torch.linalg.svd(M.float().cpu(), full_matrices=False)
    return U[:, : min(k, U.shape[1])].contiguous()


def write_cols(layer: int) -> torch.Tensor:
    cols = []
    for proj in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
        W = state.get(f"model.layers.{layer}.{proj}")
        if W is not None:
            cols.append(W.float().cpu())
    return torch.cat(cols, dim=1) if cols else torch.zeros(d_model, 0)


def lora_dW_left_basis(layer: int) -> torch.Tensor:
    cols = []
    for proj in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
        key = f"model.layers.{layer}.{proj}"
        if key in w:
            cols.append(w[key].float().cpu())
    if not cols:
        return torch.zeros(d_model, 0)
    return left_svd_basis(torch.cat(cols, dim=1))


def act_oracle_basis(layer: int) -> torch.Tensor:
    """Top-PCS right SVs of L2-normalized cumulative Δh on EVAL (in-sample)."""
    X = hs_diff_cumul[layer].float().cpu()  # [b, d]
    Xn = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)
    _U, _s, Vh = torch.linalg.svd(Xn, full_matrices=False)
    return Vh[:PCS].T.contiguous()


_u_lm, _s_lm, vh_lm = torch.linalg.svd(lm_head_W, full_matrices=False)
lm_head_read = vh_lm[:PCS].T.contiguous()  # [d, PCS], constant across layers


def random_basis(layer: int, k: int = PCS) -> torch.Tensor:
    g = torch.Generator().manual_seed(7919 + layer)
    M = torch.randn(d_model, k, generator=g)
    Q, _ = torch.linalg.qr(M)
    return Q


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    basis_by_layer: list[torch.Tensor]
    note: str


candidates: list[Candidate] = [
    Candidate("lm_head_read", "W:unembed", [lm_head_read] * n_layers,
              "top-PCS right SVs of lm_head; the canonical 'readable' subspace"),
    Candidate("write", "W:write",
              [left_svd_basis(write_cols(L)) for L in range(n_layers)],
              "per-layer top-PCS left SVs of [W_o | W_down]; base-model write subspace"),
    Candidate("TaskDiff_contrast", "act:persona",
              [pca(hs_diff_A_fit[L], PCS) for L in range(n_layers)],
              "PCA of persona+ minus persona- residual diff (FIT half)"),
    Candidate("TaskDiff_lora_fit", "act:cluster",
              [pca(hs_diff_B_fit[L], PCS) for L in range(n_layers)],
              "PCA of LoRA α=+1 vs α=-1 residual diff (FIT half, held out from EVAL)"),
    Candidate("act_oracle", "ceiling",
              [act_oracle_basis(L) for L in range(n_layers)],
              "top-PCS right SVs of L2-normalized Δh on EVAL (IN-SAMPLE; functional ceiling)"),
    Candidate("w_oracle", "ceiling",
              [lora_dW_left_basis(L) for L in range(n_layers)],
              "top-PCS left SVs of LoRA dW (residual-output tensors only)"),
    Candidate("random_null", "null",
              [random_basis(L) for L in range(n_layers)],
              "rank-PCS random orthonormal; expected ratio ~ PCS/d"),
]


# %% [markdown]
# ## Score each candidate
#
# Three functional metrics, all computed on the same B at each layer L:
#
# 1. preserved_E(B, L) = E²(B Bᵀ Δh_L) / E²(Δh_L)
#    Energy preservation: fraction of Δh's readable mass that survives projection
#    onto B. In [0, 1]; random null = PCS/d_model.
#
# 2. cap_yn(B, L) = ‖P_B (e_yes − e_no)‖² / ‖e_yes − e_no‖²
#    Yes-No direction capture: fraction of the (e_yes − e_no) readout direction
#    that lies in B's span. Δh-independent — purely a property of B vs the
#    canonical Yes/No axis. In [0, 1]; random null = PCS/d_model.
#
# 3. abs_ldiff_proj(B, L) = |(e_yes − e_no)ᵀ B Bᵀ Δh_L|  (in nats)
#    Absolute Yes-No signal that the projected Δh carries. Reported in nats
#    rather than as a ratio because Δh at LoRA layers has small Yes/No content
#    (the LoRA writes in concept space; Yes/No emerges only after downstream
#    layers, see panels (a)/(b)) -- normalising by ldiff_full gives unstable
#    >>1 ratios. Final-layer ldiff_full ≈ peak |ldiff_cumul| is the right scale.

# %%
def project(B: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    if B.shape[1] == 0:
        return torch.zeros_like(h)
    return B @ (B.T @ h)


e_yn = e_yes_minus_no
e_yn_sq = float(e_yn.pow(2).sum())
peak_ldiff_full = float(np.abs(ldiff_cumul).max())  # final-layer scale, nats

rows = []
for c in candidates:
    for L in range(n_layers):
        B = c.basis_by_layer[L]
        if B.shape[1] == 0:
            continue
        h = delta_h_cumul[L]
        h_proj = project(B, h)
        e_full = token_energy_sq(h)
        e_proj = token_energy_sq(h_proj)
        # cap_yn: how much of (e_yes - e_no) lies in B's span?
        e_yn_proj = project(B, e_yn)
        cap_yn = float(e_yn_proj.pow(2).sum()) / max(e_yn_sq, 1e-12)
        rows.append({
            "subspace": c.name,
            "family": c.family,
            "layer": L,
            "rank": int(B.shape[1]),
            "energy_full": e_full,
            "energy_proj": e_proj,
            "preserved_E": e_proj / max(e_full, 1e-12),
            "cap_yn": cap_yn,
            "ldiff_full": logit_diff(h),
            "ldiff_proj": logit_diff(h_proj),
            "abs_ldiff_proj": abs(logit_diff(h_proj)),
        })

per_layer = pl.DataFrame(rows)
per_layer.write_csv(OUT_DIR / "v10_per_layer.csv")

active = per_layer.filter(pl.col("layer").is_in(list(LORA_LAYERS)))
summary = (
    active.group_by(["subspace", "family"])
    .agg(
        pl.col("preserved_E").mean().alias("mean_preserved_E"),
        pl.col("cap_yn").mean().alias("mean_cap_yn"),
        pl.col("abs_ldiff_proj").mean().alias("mean_abs_ldiff_proj"),
        pl.col("ldiff_proj").mean().alias("mean_ldiff_proj"),  # signed
        pl.col("rank").mean().alias("mean_rank"),
    )
    .sort("mean_cap_yn", descending=True)
)
summary_path = OUT_DIR / "v10_table.tsv"
summary.write_csv(summary_path, separator="\t")

print("\n=== v10 summary (LoRA layers 8..22) ===")
print(tabulate(summary.to_pandas(), headers="keys", tablefmt="github", floatfmt="+.4f"))
print(f"\npeak |ldiff_cumul| across all layers = {peak_ldiff_full:.3f} nats (at layer {int(np.argmax(np.abs(ldiff_cumul)))})")
print(f"random-null reference for preserved_E and cap_yn: PCS/d = {PCS}/{d_model} = {PCS/d_model:.5f}")


# %% [markdown]
# ## Figure: 3 panels
# (a) Token energy of Δh per layer (cumulative, block-local, clean reference)
# (b) Logit-lens Yes-No diff of Δh per layer
# (c) Per-candidate cap_yn (mean over LoRA layers)

# %%
plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 240, "font.size": 9})
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
layers = np.arange(n_layers)

ax = axes[0]
ax.plot(layers, energy_clean, label="clean h̄ (reference)", color="gray", linewidth=1.2, linestyle="--")
ax.plot(layers, energy_cumul, label="Δh cumulative", color="#5B8FF9", linewidth=1.6, marker="o", markersize=3)
ax.plot(layers, energy_block, label="Δh block-local", color="#F6BD16", linewidth=1.6, marker="s", markersize=3)
ax.axvspan(LORA_LAYERS[0], LORA_LAYERS[-1], alpha=0.10, color="green", label="LoRA layers")
ax.set_xlabel("layer")
ax.set_ylabel(r"token energy  $E^2(h)$")
ax.set_title("(a) token energy: how readable is Δh?")
ax.grid(alpha=0.25)
ax.legend(fontsize=7, loc="upper left")

ax = axes[1]
ax.axhline(0, color="black", linewidth=0.5)
ax.plot(layers, ldiff_cumul, label="Δh cumulative", color="#5B8FF9", linewidth=1.6, marker="o", markersize=3)
ax.plot(layers, ldiff_block, label="Δh block-local", color="#F6BD16", linewidth=1.6, marker="s", markersize=3)
ax.axvspan(LORA_LAYERS[0], LORA_LAYERS[-1], alpha=0.10, color="green", label="LoRA layers")
ax.set_xlabel("layer")
ax.set_ylabel(r"$\mathrm{logit}(\,Yes\,) - \mathrm{logit}(\,No\,)$  (lens of Δh)")
ax.set_title("(b) logit lens: does Δh decode Yes/No?")
ax.grid(alpha=0.25)
ax.legend(fontsize=7, loc="upper left")

ax = axes[2]
sumdf = summary.to_pandas()
sumdf = sumdf.sort_values("mean_cap_yn", ascending=True)
ypos = np.arange(len(sumdf))
colors = []
for fam in sumdf["family"]:
    if fam == "ceiling":
        colors.append("#E8684A")
    elif fam == "null":
        colors.append("#999999")
    else:
        colors.append("#5B8FF9")
ax.barh(ypos, sumdf["mean_cap_yn"], color=colors, edgecolor="black", linewidth=0.4)
ax.set_yticks(ypos)
ax.set_yticklabels(sumdf["subspace"], fontsize=8)
ax.axvline(0, color="black", linewidth=0.5)
ax.axvline(1, color="black", linewidth=0.5, linestyle=":", alpha=0.5)
null_ref = PCS / d_model
ax.axvline(null_ref, color="#999999", linewidth=0.6, linestyle="--", alpha=0.7)
ax.set_xlabel(r"mean $\mathrm{cap}_{yn}(B) = \|P_B(e_{yes}-e_{no})\|^2 / \|e_{yes}-e_{no}\|^2$")
ax.set_title("(c) Yes-No direction capture per candidate (rank-8)")
ax.grid(axis="x", alpha=0.25)

fig.suptitle(
    "Wendler-style functional probe of LoRA-induced Δh on Qwen3-0.6B (sycophancy LoRA, EVAL=12 prompts)",
    fontsize=10,
)
fig.tight_layout()
fig_png = OUT_DIR / "v10_wendler_metrics.png"
fig_pdf = OUT_DIR / "v10_wendler_metrics.pdf"
fig.savefig(fig_png, bbox_inches="tight")
fig.savefig(fig_pdf, bbox_inches="tight")
plt.close(fig)
logger.info(f"wrote figure: {fig_png}")


# %% [markdown]
# ## Caption + interp sequence (paper-style SHOULD/ELSE diagnostics)

# %%
peak_layer_pos = int(np.argmax(ldiff_cumul))
peak_layer_neg = int(np.argmin(ldiff_cumul))
peak_E_layer = int(np.argmax(energy_cumul))
peak_E = float(energy_cumul[peak_E_layer])
peak_E_clean = float(energy_clean[peak_E_layer])
peak_ldiff = float(ldiff_cumul[peak_layer_pos])

# Score sanity for headline interpretation.
def _get(name: str, col: str) -> float:
    return float(summary.filter(pl.col("subspace") == name)[col][0])

oracle_cap = _get("act_oracle", "mean_cap_yn")
woracle_cap = _get("w_oracle", "mean_cap_yn")
null_cap = _get("random_null", "mean_cap_yn")
lmread_cap = _get("lm_head_read", "mean_cap_yn")
write_cap = _get("write", "mean_cap_yn")
taskdiff_cap = _get("TaskDiff_lora_fit", "mean_cap_yn")
taskcontrast_cap = _get("TaskDiff_contrast", "mean_cap_yn")

oracle_E = _get("act_oracle", "mean_preserved_E")
woracle_E = _get("w_oracle", "mean_preserved_E")
taskdiff_E = _get("TaskDiff_lora_fit", "mean_preserved_E")
lmread_E = _get("lm_head_read", "mean_preserved_E")
write_E = _get("write", "mean_preserved_E")
null_E = _get("random_null", "mean_preserved_E")

caption = f"""# v10 figure caption + interp sequence

## Caption (paper-quality)

**Figure.** Wendler-style functional probe of the LoRA-induced residual-stream
shift Δh = h(α=+1) − h(α=−1) on Qwen3-0.6B (sycophancy LoRA, 12 held-out
EVAL prompts). **(a)** Token energy E²(h) per layer (Wendler et al. 2024,
Eq. 2): the fraction of h's mass that projects onto the unembedding rowspace,
normalised so a typical token has E² ≈ 1. Cumulative Δh (residual stream after
block L) and block-local Δh (post − pre at L) compared against clean mean
residuals. LoRA-active layers shaded green. **(b)** Logit-lens Yes-vs-No score
on Δh per layer: lm_head @ Δh evaluated at the " Yes" and " No" token rows.
A non-zero value means Δh directly contributes to the Yes-No logit difference
at that layer (no further forward computation required). **(c)** Yes-No
direction capture per candidate rank-8 subspace B:
cap_yn(B) = ‖P_B(e_yes − e_no)‖² / ‖e_yes − e_no‖² averaged over LoRA layers
8..21. This is Δh-independent (it asks "does B contain the readout axis?"),
so it stays in [0, 1] and avoids the small-denominator instability that
ldiff(B Bᵀ Δh)/ldiff(Δh) suffers from at LoRA layers (where ldiff(Δh) is
near zero by construction; see panel (b)). Orange = oracles, blue =
base-model and persona hypotheses, grey = random-orthonormal null
(reference line at PCS/d).

## Headline numbers

- Peak token energy of Δh: **E² = {peak_E:.3f}** at layer {peak_E_layer} (clean
  reference at same layer: {peak_E_clean:.3f}). Peak |logit-lens Yes-No|:
  **{abs(peak_ldiff):.2f} nats** at layer {peak_layer_pos}.
- act_oracle:        cap_yn = **{oracle_cap:.3f}**, preserved_E = **{oracle_E:.3f}**  (in-sample ceiling, IN-EVAL).
- w_oracle:          cap_yn = **{woracle_cap:.3f}**, preserved_E = **{woracle_E:.3f}**  (LoRA dW left SVD).
- lm_head_read:      cap_yn = **{lmread_cap:.3f}**, preserved_E = **{lmread_E:.3f}**.
- TaskDiff_lora_fit: cap_yn = **{taskdiff_cap:.3f}**, preserved_E = **{taskdiff_E:.3f}** (FIT-half PCA).
- TaskDiff_contrast: cap_yn = **{taskcontrast_cap:.3f}**.
- write:             cap_yn = **{write_cap:.3f}**, preserved_E = **{write_E:.3f}**.
- random_null:       cap_yn = **{null_cap:.3f}**, preserved_E = **{null_E:.3f}**  (expected ≈ {PCS}/{d_model} = {PCS/d_model:.4f}).

## Interpretation sequence (read top to bottom)

(a) Token energy of Δh.

> SHOULD: E² ≪ 1 in early layers (Δh orthogonal to readout, doing concept-space
> work) and rise toward final layers if the LoRA writes into readable space.
> ELSE: if E² ≈ 0 throughout, the LoRA is *entirely* in concept space and any
> token-readout-based hypothesis (lm_head_read) is structurally wrong.

(b) Logit-lens Yes-No diff on Δh.

> SHOULD: monotone-in-magnitude rise across LoRA-active layers, sign matching
> the steering direction (positive α = LoRA was trained to be more sycophantic
> = should boost " Yes" on these "I think X. Do you agree?" prompts, hence
> ldiff > 0). ELSE: if ldiff stays at noise across all LoRA layers, the LoRA's
> effect on Yes/No is mediated entirely by downstream non-linear computation
> -- story (B) nonlinearity is forced and Wendler-style readout cannot reach it.

(c) Per-candidate Yes-No direction capture.

> SHOULD: lm_head_read should score highest among A-side hypotheses by
> construction (its top-PCS right SVs of lm_head are exactly the directions
> that decode strongly into vocabulary -- so the (e_yes − e_no) axis should
> live mostly in this subspace). Random null ≈ {PCS/d_model:.4f} (rank/d).
> ELSE: if even lm_head_read scores low (<<0.5), then the rank-8 PCA of
> lm_head doesn't capture the Yes-No axis -- it's spread across many
> singular directions, and rank-8 unembedding-readable is not a useful
> hypothesis class at this size.

## What this tells us vs v9

v9 said all A-side candidates score <15% of the PCA-span oracle on the LoRA
delta. v10 asks an orthogonal functional question: regardless of the LoRA,
how much of the (e_yes − e_no) readout axis lives in each candidate
subspace? Panel (b) shows that Δh's effect on Yes/No is *not* readable at
LoRA layers (it emerges only post-LoRA, at layer ~{peak_layer_pos}), so the
LoRA writes in concept space (Wendler Phase 2), not directly in token
space. cap_yn separates "does B contain the readout direction" (panel c)
from "does the LoRA write toward that direction" (panel b) -- two failures
that v9's PCA-span metric conflated.
"""
caption_path = OUT_DIR / "v10_caption.md"
caption_path.write_text(caption)
logger.info(f"wrote caption: {caption_path}")

print("\n=== v10 outputs ===")
for p in [OUT_DIR / "v10_per_layer.csv", summary_path, fig_png, fig_pdf, caption_path]:
    print(f"  {p} ({p.stat().st_size} bytes)")
