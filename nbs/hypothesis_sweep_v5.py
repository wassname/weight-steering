# %% [markdown]
# # v5 hypothesis sweep: broaden A-side recipes, keep one score
#
# **Question.** Which LoRA-free recipe best predicts where the trained sycophancy LoRA
# writes its activation-space steering signal?
#
# **One score.** Every candidate is an A-side basis $V_{m,\ell}$, built from pretrained
# weights and/or base-model activations only. We score it against the held-out B-side
# label $\Delta h^B_\ell = h_\ell(\alpha=+1)-h_\ell(\alpha=-1)$:
#
# $$
# R_{m,\ell}=\frac{\mathbb{E}\|P_{V_{m,\ell}}\Delta h^B_\ell\|^2/\|\Delta h^B_\ell\|^2}{\dim(V_{m,\ell})/d}.
# $$
#
# This notebook deliberately tests many hypotheses, but refuses to change the scoring
# rule per hypothesis. That makes the winner meaningful and failure legible.

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
    "logs/hypothesis_sweep_v5.verbose.log",
    level="DEBUG",
    format="{time} | {level} | {name}:{function}:{line} - {message}",
)
torch.set_grad_enabled(False)

MODEL_ID = "Qwen/Qwen3-0.6B"
W_PATH = Path("out/sycophancy/lora/w.pt")
OUT_DIR = Path("out/sycophancy/lora")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PCS = 8
K_READ_BROAD = 64
N_NULL = 120
LORA_LAYERS = range(8, 22)
BOOT = 20_000
RNG = np.random.default_rng(0)

PROBE_PROMPTS = [
    f"I think {claim}. Do you agree? Answer Yes or No." for claim, _q in SYCOPHANCY_TOPICS
]
FIT = PROBE_PROMPTS[: len(PROBE_PROMPTS) // 2]
EVAL = PROBE_PROMPTS[len(PROBE_PROMPTS) // 2 :]


# %% [markdown]
# ## Load model and capture the B-side label

# %%
w = load_diff(W_PATH)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
state = model.state_dict()
n_layers = model.config.num_hidden_layers
HOOKS = [f"model.layers.{i}" for i in range(n_layers)]
UP_HOOKS = [f"model.layers.{i}.mlp.up_proj" for i in range(n_layers)]

lm_head_W = state.get("lm_head.weight")
if lm_head_W is None:
    lm_head_W = state["model.embed_tokens.weight"]
lm_head_W = lm_head_W.float().cpu()
d_model = lm_head_W.shape[1]
logger.info(f"loaded {MODEL_ID} | layers={n_layers} | d_model={d_model} | LoRA tensors={len(w)}")


# %%
def pca(samples: torch.Tensor, k: int) -> torch.Tensor:
    if samples.shape[0] <= 1:
        return samples.new_zeros(samples.shape[1], 0)
    centered = samples - samples.mean(0, keepdim=True)
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[: min(k, vh.shape[0])].T.contiguous()


def basis_from_gram(gram: torch.Tensor, k: int) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(gram.float().cpu())
    keep = torch.argsort(evals, descending=True)[:k]
    return evecs[:, keep].contiguous()


def orthonormalize(M: torch.Tensor, *, eps: float = 1e-5) -> torch.Tensor:
    if M.numel() == 0:
        return M.new_zeros(M.shape[0], 0)
    Q, R = torch.linalg.qr(M)
    keep = R.diag().abs() > eps
    return Q[:, keep]


def orthonormal_union(*basis_list: torch.Tensor) -> torch.Tensor:
    nonempty = [B for B in basis_list if B.shape[1] > 0]
    if not nonempty:
        return torch.zeros(d_model, 0)
    return orthonormalize(torch.cat(nonempty, dim=1))


def principal_cos(A: torch.Tensor, B: torch.Tensor) -> float:
    if A.shape[1] == 0 or B.shape[1] == 0:
        return float("nan")
    return float(torch.linalg.svdvals(A.T @ B).clamp(0, 1).mean())


def mean_principal_angle_deg(A: torch.Tensor, B: torch.Tensor) -> float:
    if A.shape[1] == 0 or B.shape[1] == 0:
        return float("nan")
    cos = torch.linalg.svdvals(A.T @ B).clamp(0, 1)
    return float(torch.rad2deg(torch.arccos(cos)).mean())


def capture_blocks(prompts: list[str], *, alpha: float = 0.0, system: str | None = None) -> torch.Tensor:
    if system is not None:
        msgs = [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in prompts]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    else:
        texts = prompts
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
    return torch.stack(rows, 0)


def capture_up_inputs(prompts: list[str], *, system: str | None = None) -> torch.Tensor:
    if system is not None:
        msgs = [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in prompts]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    else:
        texts = prompts
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
    seq_idx = enc.attention_mask.sum(-1) - 1
    with TraceDict(model, UP_HOOKS, retain_input=True) as ret:
        _ = model(**enc)
    rows = []
    for hook in UP_HOOKS:
        x = ret[hook].input
        if isinstance(x, tuple):
            x = x[0]
        b, _s, d = x.shape
        rows.append(x.gather(1, seq_idx.view(b, 1, 1).expand(b, 1, d)).squeeze(1).float().cpu())
    return torch.stack(rows, 0)


def capture_up_outputs_written(prompts: list[str], *, system: str | None = None) -> torch.Tensor:
    if system is not None:
        msgs = [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in prompts]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    else:
        texts = prompts
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
    seq_idx = enc.attention_mask.sum(-1) - 1
    with TraceDict(model, UP_HOOKS, retain_output=True) as ret:
        _ = model(**enc)
    rows = []
    for layer, hook in enumerate(UP_HOOKS):
        x = ret[hook].output
        if isinstance(x, tuple):
            x = x[0]
        b, _s, d_mlp = x.shape
        x_last = x.gather(1, seq_idx.view(b, 1, 1).expand(b, 1, d_mlp)).squeeze(1).float().cpu()
        W_down = state[f"model.layers.{layer}.mlp.down_proj.weight"].float().cpu()
        rows.append(x_last @ W_down.T)
    return torch.stack(rows, 0)


def suppressed_features(acts: torch.Tensor) -> torch.Tensor:
    mag = acts.abs().permute(1, 0, 2)
    delta = mag[:, 1:] - mag[:, :-1]
    return torch.minimum(torch.relu(delta).sum(1), torch.relu(-delta).sum(1))


def amplified_features(acts: torch.Tensor) -> torch.Tensor:
    mag = acts.abs().permute(1, 0, 2)
    return torch.relu(mag[:, -1] - mag[:, 0])


def procrustes_rotation_basis(X: torch.Tensor, Y: torch.Tensor, *, k: int = PCS, rank: int = 32) -> torch.Tensor:
    joint = pca(torch.cat([X, Y], dim=0), min(rank, X.shape[0] + Y.shape[0] - 2, X.shape[1]))
    if joint.shape[1] < 2:
        return torch.zeros(X.shape[1], 0)
    Xr = (X - X.mean(0, keepdim=True)) @ joint
    Yr = (Y - Y.mean(0, keepdim=True)) @ joint
    U, _s, Vh = torch.linalg.svd(Xr.T @ Yr, full_matrices=False)
    R = U @ Vh
    skew = R - R.T
    U_skew, _s_skew, _Vh_skew = torch.linalg.svd(skew, full_matrices=False)
    return orthonormalize(joint @ U_skew[:, : min(k, U_skew.shape[1])])


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    basis_by_layer: list[torch.Tensor]
    definition: str


# %%
logger.info("capturing B-side label and A-side activations")
hs_pos_eval = capture_blocks(EVAL, alpha=+1.0)
hs_neg_eval = capture_blocks(EVAL, alpha=-1.0)
hs_diff_B = hs_pos_eval - hs_neg_eval
hs_pos_fit = capture_blocks(FIT, alpha=+1.0)
hs_neg_fit = capture_blocks(FIT, alpha=-1.0)
hs_diff_B_fit = hs_pos_fit - hs_neg_fit

hs_persona_pos_fit = capture_blocks(FIT, system=SYCOPHANCY_POS_PERSONAS[0])
hs_persona_neg_fit = capture_blocks(FIT, system=SYCOPHANCY_NEG_PERSONAS[0])
hs_diff_A_fit = hs_persona_pos_fit - hs_persona_neg_fit
hs_clean_fit = capture_blocks(FIT)

up_persona_pos_fit = capture_up_inputs(FIT, system=SYCOPHANCY_POS_PERSONAS[0])
up_persona_neg_fit = capture_up_inputs(FIT, system=SYCOPHANCY_NEG_PERSONAS[0])
up_diff_A_fit = up_persona_pos_fit - up_persona_neg_fit
up_written_pos_fit = capture_up_outputs_written(FIT, system=SYCOPHANCY_POS_PERSONAS[0])
up_written_neg_fit = capture_up_outputs_written(FIT, system=SYCOPHANCY_NEG_PERSONAS[0])
up_written_diff_A_fit = up_written_pos_fit - up_written_neg_fit
logger.info(
    f"captured activations | label shape={tuple(hs_diff_B.shape)} | "
    f"up input shape={tuple(up_diff_A_fit.shape)} | up written shape={tuple(up_written_diff_A_fit.shape)}"
)


# %% [markdown]
# ## Build expanded A-side hypothesis set

# %%
def write_cols(layer: int, kinds: tuple[str, ...] = ("self_attn.o_proj.weight", "mlp.down_proj.weight")) -> torch.Tensor:
    cols = []
    for proj in kinds:
        key = f"model.layers.{layer}.{proj}"
        W = state.get(key)
        if W is not None:
            cols.append(W.float().cpu())
    if not cols:
        return torch.zeros(d_model, 0)
    return torch.cat(cols, dim=1)


def read_gram(layer: int) -> torch.Tensor:
    gram = torch.zeros(d_model, d_model)
    for proj in (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "mlp.up_proj.weight",
        "mlp.gate_proj.weight",
    ):
        W = state.get(f"model.layers.{layer}.{proj}")
        if W is not None:
            Wf = W.float().cpu()
            gram += Wf.T @ Wf
    return gram


def left_svd_basis(M: torch.Tensor, k: int = PCS) -> torch.Tensor:
    if M.shape[1] == 0:
        return torch.zeros(M.shape[0], 0)
    U, _s, _Vh = torch.linalg.svd(M.float().cpu(), full_matrices=False)
    return U[:, : min(k, U.shape[1])].contiguous()


def right_svd_basis(M: torch.Tensor, k: int = PCS) -> torch.Tensor:
    if M.shape[0] == 0:
        return torch.zeros(M.shape[1], 0)
    _U, _s, Vh = torch.linalg.svd(M.float().cpu(), full_matrices=False)
    return Vh[: min(k, Vh.shape[0])].T.contiguous()


def expand_gqa_v_rows(W_v: torch.Tensor, W_o: torch.Tensor) -> torch.Tensor:
    if W_v.shape[0] == W_o.shape[1]:
        return W_v
    repeats = W_o.shape[1] // W_v.shape[0]
    if repeats * W_v.shape[0] != W_o.shape[1]:
        raise ValueError(f"cannot align W_v rows {tuple(W_v.shape)} to W_o {tuple(W_o.shape)}")
    return W_v.repeat_interleave(repeats, dim=0)


_u_lm, _s_lm, vh_lm = torch.linalg.svd(lm_head_W, full_matrices=False)
lm_head_read = vh_lm[:PCS].T.contiguous()
logits_null = vh_lm[-PCS:].T.contiguous()
lm_read_broad = vh_lm[:K_READ_BROAD].T.contiguous()

read_grams = [read_gram(layer) for layer in range(n_layers)]
global_read_gram = sum(read_grams, torch.zeros(d_model, d_model)) + lm_head_W.T @ lm_head_W
global_read = basis_from_gram(global_read_gram, PCS)
global_read_broad = basis_from_gram(global_read_gram, K_READ_BROAD)
global_write_cols = torch.cat([write_cols(layer) for layer in range(n_layers)], dim=1)
global_write = left_svd_basis(global_write_cols)

downstream_read_broad = []
running = lm_head_W.T @ lm_head_W
for layer in reversed(range(n_layers)):
    if layer < n_layers - 1:
        running = running + read_grams[layer + 1]
    downstream_read_broad.append(basis_from_gram(running, K_READ_BROAD))
downstream_read_broad = list(reversed(downstream_read_broad))

eye = torch.eye(d_model)
P_lm = lm_read_broad @ lm_read_broad.T
P_global_read = global_read_broad @ global_read_broad.T

candidate_list: list[Candidate] = []


def add(name: str, family: str, basis_by_layer: list[torch.Tensor], definition: str) -> None:
    if len(basis_by_layer) != n_layers:
        raise ValueError(f"{name} has {len(basis_by_layer)} layers, expected {n_layers}")
    candidate_list.append(Candidate(name, family, basis_by_layer, definition))


add("lm_head_read", "W:unembed", [lm_head_read] * n_layers, "top right singular vectors of lm_head")
add("logits_null", "W:unembed", [logits_null] * n_layers, "bottom right singular vectors of lm_head")
add("global_read", "W:read", [global_read] * n_layers, "top eigenspace of all q/k/v/up/gate reads + lm_head")
add("global_write", "W:write", [global_write] * n_layers, "top left singular vectors of all o/down residual writers")
add(
    "global_write_not_global_read",
    "W:write-not-read",
    [left_svd_basis((eye - P_global_read) @ global_write_cols)] * n_layers,
    "global residual write projected away from global read directions",
)

write = [left_svd_basis(write_cols(layer)) for layer in range(n_layers)]
attn_write = [left_svd_basis(write_cols(layer, ("self_attn.o_proj.weight",))) for layer in range(n_layers)]
mlp_write = [left_svd_basis(write_cols(layer, ("mlp.down_proj.weight",))) for layer in range(n_layers)]
write_not_lm = [left_svd_basis((eye - P_lm) @ write_cols(layer)) for layer in range(n_layers)]
write_not_global_read = [left_svd_basis((eye - P_global_read) @ write_cols(layer)) for layer in range(n_layers)]
write_not_downstream_read = [
    left_svd_basis((eye - downstream_read_broad[layer] @ downstream_read_broad[layer].T) @ write_cols(layer))
    for layer in range(n_layers)
]
add("write", "W:write", write, "per-layer top left singular vectors of [W_o | W_down]")
add("attn_write", "W:write", attn_write, "per-layer top left singular vectors of W_o")
add("mlp_write", "W:write", mlp_write, "per-layer top left singular vectors of W_down")
add("write_not_lm_head_read", "W:write-not-read", write_not_lm, "per-layer write projected away from lm_head top read")
add("write_not_global_read", "W:write-not-read", write_not_global_read, "per-layer write projected away from global read")
add("write_not_downstream_read", "W:write-not-read", write_not_downstream_read, "per-layer write projected away from downstream read + lm_head")

mlp_up_read = []
mlp_gate_read = []
attn_qkv_read = []
attn_ov_write = []
mlp_roundtrip = []
for layer in range(n_layers):
    up = state[f"model.layers.{layer}.mlp.up_proj.weight"].float().cpu()
    gate = state[f"model.layers.{layer}.mlp.gate_proj.weight"].float().cpu()
    qkv = torch.cat([
        state[f"model.layers.{layer}.self_attn.q_proj.weight"].float().cpu(),
        state[f"model.layers.{layer}.self_attn.k_proj.weight"].float().cpu(),
        state[f"model.layers.{layer}.self_attn.v_proj.weight"].float().cpu(),
    ], dim=0)
    W_o = state[f"model.layers.{layer}.self_attn.o_proj.weight"].float().cpu()
    W_v = state[f"model.layers.{layer}.self_attn.v_proj.weight"].float().cpu()
    W_down = state[f"model.layers.{layer}.mlp.down_proj.weight"].float().cpu()
    mlp_up_read.append(right_svd_basis(up))
    mlp_gate_read.append(right_svd_basis(gate))
    attn_qkv_read.append(right_svd_basis(qkv))
    attn_ov_write.append(left_svd_basis(W_o @ expand_gqa_v_rows(W_v, W_o)))
    mlp_roundtrip.append(left_svd_basis(W_down @ up))
add("mlp_up_read", "W:read", mlp_up_read, "right singular vectors of W_up, i.e. MLP expansion read directions")
add("mlp_gate_read", "W:read", mlp_gate_read, "right singular vectors of W_gate")
add("attn_qkv_read", "W:read", attn_qkv_read, "right singular vectors of concatenated W_q/W_k/W_v")
add("attn_ov_write", "W:OV", attn_ov_write, "left singular vectors of W_o W_v")
add("mlp_roundtrip_write", "W:MLP", mlp_roundtrip, "left singular vectors of W_down W_up residual-to-residual map")

suppressed = pca(suppressed_features(hs_clean_fit), PCS)
amplified = pca(amplified_features(hs_clean_fit), PCS)
global_clean_pca = pca(hs_clean_fit.permute(1, 0, 2).reshape(-1, d_model), PCS)
global_persona_pca = pca(
    torch.cat([
        hs_persona_pos_fit.permute(1, 0, 2).reshape(-1, d_model),
        hs_persona_neg_fit.permute(1, 0, 2).reshape(-1, d_model),
    ]),
    PCS,
)
add("suppressed", "act:clean", [suppressed] * n_layers, "PCA of base-model magnitude turnover across layers")
add("amplified", "act:clean", [amplified] * n_layers, "PCA of base-model magnitudes that persist from first to last layer")
add("global_clean_resid_pca", "act:baseline", [global_clean_pca] * n_layers, "PCA of all clean base residual activations; generic anisotropy baseline")
add("global_persona_resid_pca", "act:baseline", [global_persona_pca] * n_layers, "PCA of persona+ and persona- residual activations without differencing")
add("layer_clean_resid_pca", "act:baseline", [pca(hs_clean_fit[layer], PCS) for layer in range(n_layers)], "per-layer PCA of clean base residual activations")
add("TaskDiff_contrast", "act:persona", [pca(hs_diff_A_fit[layer], PCS) for layer in range(n_layers)], "PCA of persona+ minus persona- residual activations")
add("up_proj_input_contrast", "act:up_proj", [pca(up_diff_A_fit[layer], PCS) for layer in range(n_layers)], "PCA of persona contrast in inputs to mlp.up_proj")
add("up_proj_output_written_contrast", "act:up_proj", [pca(up_written_diff_A_fit[layer], PCS) for layer in range(n_layers)], "PCA of persona contrast after W_up, mapped back to residual by W_down")
add("churn", "act:clean", [pca(hs_clean_fit[min(layer + 1, n_layers - 1)] - hs_clean_fit[layer], PCS) for layer in range(n_layers)], "PCA of signed clean residual change h_{l+1}-h_l")
add(
    "rotation_contrast",
    "act:rotation",
    [procrustes_rotation_basis(hs_persona_neg_fit[layer], hs_persona_pos_fit[layer]) for layer in range(n_layers)],
    "top directions of the skew generator from persona- to persona+ Procrustes rotation",
)
add(
    "WNR_union_TaskDiff",
    "compound",
    [orthonormal_union(write_not_downstream_read[layer], pca(hs_diff_A_fit[layer], PCS)) for layer in range(n_layers)],
    "rank-expanded union of write_not_downstream_read and TaskDiff_contrast",
)

ceiling = Candidate(
    "TaskDiff_lora_ceiling",
    "ceiling",
    [pca(hs_diff_B_fit[layer], PCS) for layer in range(n_layers)],
    "PCA of LoRA FIT-half label; not an A-side hypothesis",
)

logger.info(f"built {len(candidate_list)} A-side candidates + ceiling")


# %% [markdown]
# ## Score every candidate against the same held-out LoRA label

# %%
null_cache: dict[tuple[int, int], tuple[float, float]] = {}


def null_stats(layer: int, rank: int) -> tuple[float, float]:
    key = (layer, rank)
    if key in null_cache:
        return null_cache[key]
    samples = hs_diff_B[layer]
    d = samples.shape[1]
    total = samples.pow(2).sum(1) + 1e-12
    null = rank / d
    gen = torch.Generator(device=samples.device).manual_seed(10_000 + 97 * layer + rank)
    values = []
    for _ in range(N_NULL):
        rb, _ = torch.linalg.qr(torch.randn(d, rank, generator=gen, device=samples.device, dtype=samples.dtype))
        values.append(((samples @ rb).pow(2).sum(1) / total).mean().item() / null)
    arr = torch.tensor(values)
    stats = (float(arr.mean()), float(arr.std(unbiased=True)))
    null_cache[key] = stats
    return stats


def concentration(layer: int, basis: torch.Tensor) -> dict[str, float]:
    samples = hs_diff_B[layer]
    rank = basis.shape[1]
    if rank == 0:
        return {"conc": 0.0, "z": 0.0, "energy_frac": 0.0}
    total = samples.pow(2).sum(1) + 1e-12
    energy_frac = ((samples @ basis).pow(2).sum(1) / total).mean().item()
    conc = energy_frac / (rank / samples.shape[1])
    null_mean, null_std = null_stats(layer, rank)
    return {"conc": conc, "z": (conc - null_mean) / (null_std + 1e-12), "energy_frac": energy_frac}


def dw_left_basis(layer: int) -> torch.Tensor:
    cols = []
    for proj in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
        key = f"model.layers.{layer}.{proj}"
        if key in w:
            cols.append(w[key].float().cpu())
    if not cols:
        return torch.zeros(d_model, 0)
    return left_svd_basis(torch.cat(cols, dim=1))


all_candidates = [*candidate_list, ceiling]
dw_bases = [dw_left_basis(layer) for layer in range(n_layers)]
rows = []
for layer in range(n_layers):
    for candidate in all_candidates:
        basis = candidate.basis_by_layer[layer]
        score = concentration(layer, basis)
        rows.append({
            "layer": layer,
            "subspace": candidate.name,
            "family": candidate.family,
            "kind": "ceiling" if candidate.family == "ceiling" else "A-hypothesis",
            "rank": basis.shape[1],
            "conc_in_B": score["conc"],
            "energy_frac": score["energy_frac"],
            "z": score["z"],
            "cos_with_dW": principal_cos(basis, dw_bases[layer]),
        })

per_layer = pl.DataFrame(rows)
per_layer_path = OUT_DIR / "v5_hypothesis_sweep_per_layer.csv"
per_layer.write_csv(per_layer_path)


# %% [markdown]
# ## Specificity control: remove generic clean-residual PCs
#
# `layer_clean_resid_pca` is a deliberately boring baseline. If it wins the raw score,
# the raw score is partly measuring generic residual-stream anisotropy. The control below
# projects both the B-side label and every candidate away from the per-layer clean PCA,
# then reruns the same concentration score in the residual ambient dimension.

# %%
clean_basis_by_layer = {c.name: c.basis_by_layer for c in candidate_list}["layer_clean_resid_pca"]


def complement_basis(candidate_basis: torch.Tensor, baseline_basis: torch.Tensor) -> torch.Tensor:
    P0 = baseline_basis @ baseline_basis.T
    return orthonormalize((torch.eye(candidate_basis.shape[0]) - P0) @ candidate_basis)


specific_null_cache: dict[tuple[int, int, int], tuple[float, float]] = {}


def specific_null_stats(layer: int, rank: int, ambient_rank: int) -> tuple[float, float]:
    key = (layer, rank, ambient_rank)
    if key in specific_null_cache:
        return specific_null_cache[key]
    clean = clean_basis_by_layer[layer]
    P_clean = clean @ clean.T
    samples = hs_diff_B[layer] @ (torch.eye(d_model) - P_clean)
    total = samples.pow(2).sum(1) + 1e-12
    null = rank / ambient_rank
    gen = torch.Generator(device=samples.device).manual_seed(50_000 + 97 * layer + 13 * rank)
    values = []
    for _ in range(N_NULL):
        rb, _ = torch.linalg.qr(torch.randn(d_model, rank, generator=gen, device=samples.device, dtype=samples.dtype))
        rb = complement_basis(rb, clean)
        if rb.shape[1] != rank:
            raise ValueError(f"random residual rank collapsed: layer={layer}, rank={rank}, got={rb.shape[1]}")
        values.append(((samples @ rb).pow(2).sum(1) / total).mean().item() / null)
    arr = torch.tensor(values)
    stats = (float(arr.mean()), float(arr.std(unbiased=True)))
    specific_null_cache[key] = stats
    return stats


def specific_concentration(layer: int, basis: torch.Tensor) -> dict[str, float]:
    clean = clean_basis_by_layer[layer]
    P_clean = clean @ clean.T
    residual_basis = complement_basis(basis, clean)
    rank = residual_basis.shape[1]
    if rank == 0:
        return {"specific_conc": 0.0, "specific_z": 0.0, "specific_energy_frac": 0.0, "specific_rank": 0}
    samples = hs_diff_B[layer] @ (torch.eye(d_model) - P_clean)
    total = samples.pow(2).sum(1) + 1e-12
    ambient_rank = d_model - clean.shape[1]
    energy_frac = ((samples @ residual_basis).pow(2).sum(1) / total).mean().item()
    conc = energy_frac / (rank / ambient_rank)
    null_mean, null_std = specific_null_stats(layer, rank, ambient_rank)
    return {
        "specific_conc": conc,
        "specific_z": (conc - null_mean) / (null_std + 1e-12),
        "specific_energy_frac": energy_frac,
        "specific_rank": rank,
    }


specific_rows = []
for layer in range(n_layers):
    for candidate in all_candidates:
        score = specific_concentration(layer, candidate.basis_by_layer[layer])
        specific_rows.append({
            "layer": layer,
            "subspace": candidate.name,
            "family": candidate.family,
            "kind": "ceiling" if candidate.family == "ceiling" else "A-hypothesis",
            **score,
        })

specific_per_layer = pl.DataFrame(specific_rows)
specific_per_layer_path = OUT_DIR / "v5_hypothesis_sweep_specific_per_layer.csv"
specific_per_layer.write_csv(specific_per_layer_path)


# %%
active = per_layer.filter(pl.col("layer").is_in(list(LORA_LAYERS)))
summary = (
    active.group_by(["subspace", "family", "kind"])
    .agg(
        pl.col("conc_in_B").mean().alias("mean_conc_B"),
        pl.col("conc_in_B").median().alias("median_conc_B"),
        pl.col("conc_in_B").max().alias("max_conc_B"),
        pl.col("energy_frac").mean().alias("mean_energy_frac"),
        pl.col("z").mean().alias("mean_z"),
        pl.col("cos_with_dW").mean().alias("mean_cos_dW"),
        pl.col("rank").mean().alias("mean_rank"),
    )
    .sort("mean_conc_B", descending=True)
)

ceiling_mean = float(summary.filter(pl.col("kind") == "ceiling")["mean_conc_B"][0])
summary = summary.with_columns(pct_ceiling=100 * pl.col("mean_conc_B") / ceiling_mean)

a_summary = summary.filter(pl.col("kind") == "A-hypothesis")
candidate_names = a_summary["subspace"].to_list()
wide = active.select("layer", "subspace", "conc_in_B").pivot(
    index="layer", on="subspace", values="conc_in_B"
).sort("layer")
for name in candidate_names:
    if name not in wide.columns:
        raise ValueError(f"missing candidate in wide table: {name}")

winner = candidate_names[0]
runner_up = candidate_names[1]
winner_values = wide[winner].to_numpy()
runner_values = wide[runner_up].to_numpy()
layer_margins = np.log2(winner_values) - np.log2(runner_values)
boot_idx = RNG.integers(0, len(layer_margins), size=(BOOT, len(layer_margins)))
boot_means = layer_margins[boot_idx].mean(axis=1)
margin_mean = float(layer_margins.mean())
margin_low = float(np.quantile(boot_means, 0.025))
margin_high = float(np.quantile(boot_means, 0.975))
winner_layers = int((layer_margins > 0).sum())

layer_best = []
for row in wide.iter_rows(named=True):
    best_name = max(candidate_names, key=lambda name: row[name])
    layer_best.append({"layer": row["layer"], "best_subspace": best_name, "best_conc": row[best_name]})
layer_best_df = pl.DataFrame(layer_best)

summary_path = OUT_DIR / "v5_hypothesis_sweep_summary.tsv"
layer_best_path = OUT_DIR / "v5_hypothesis_sweep_layer_winners.tsv"
summary.write_csv(summary_path, separator="\t")
layer_best_df.write_csv(layer_best_path, separator="\t")

print("BLUF:")
print(
    f"winner={winner} | runner_up={runner_up} | margin_log2={margin_mean:+.2f} "
    f"[{margin_low:+.2f}, {margin_high:+.2f}] | layer_wins={winner_layers}/{len(list(LORA_LAYERS))}"
)
print(tabulate(summary.head(16).to_pandas(), headers="keys", tablefmt="github", floatfmt="+.3f"))


# %% [markdown]
# ## Diagnostics: which families mattered?

# %%
family_summary = (
    active.filter(pl.col("kind") == "A-hypothesis")
    .group_by("family")
    .agg(
        pl.col("conc_in_B").mean().alias("mean_conc_B"),
        pl.col("z").mean().alias("mean_z"),
        pl.col("cos_with_dW").mean().alias("mean_cos_dW"),
        pl.len().alias("n_layer_scores"),
    )
    .sort("mean_conc_B", descending=True)
)
family_path = OUT_DIR / "v5_hypothesis_sweep_family_summary.tsv"
family_summary.write_csv(family_path, separator="\t")
print(tabulate(family_summary.to_pandas(), headers="keys", tablefmt="github", floatfmt="+.3f"))

specific_active = specific_per_layer.filter(pl.col("layer").is_in(list(LORA_LAYERS)))
specific_summary = (
    specific_active.group_by(["subspace", "family", "kind"])
    .agg(
        pl.col("specific_conc").mean().alias("mean_specific_conc"),
        pl.col("specific_conc").median().alias("median_specific_conc"),
        pl.col("specific_conc").max().alias("max_specific_conc"),
        pl.col("specific_energy_frac").mean().alias("mean_specific_energy_frac"),
        pl.col("specific_z").mean().alias("mean_specific_z"),
        pl.col("specific_rank").mean().alias("mean_specific_rank"),
    )
    .sort("mean_specific_conc", descending=True)
)
specific_ceiling_mean = float(specific_summary.filter(pl.col("kind") == "ceiling")["mean_specific_conc"][0])
specific_summary = specific_summary.with_columns(
    pct_specific_ceiling=100 * pl.col("mean_specific_conc") / specific_ceiling_mean
)
specific_summary_path = OUT_DIR / "v5_hypothesis_sweep_specific_summary.tsv"
specific_summary.write_csv(specific_summary_path, separator="\t")

specific_a_names = specific_summary.filter(pl.col("kind") == "A-hypothesis")["subspace"].to_list()
specific_wide = specific_active.select("layer", "subspace", "specific_conc").pivot(
    index="layer", on="subspace", values="specific_conc"
).sort("layer")
specific_winner = specific_a_names[0]
specific_runner_up = specific_a_names[1]
specific_margins = np.log2(specific_wide[specific_winner].to_numpy()) - np.log2(
    specific_wide[specific_runner_up].to_numpy()
)
specific_boot_idx = RNG.integers(0, len(specific_margins), size=(BOOT, len(specific_margins)))
specific_boot_means = specific_margins[specific_boot_idx].mean(axis=1)
specific_margin_mean = float(specific_margins.mean())
specific_margin_low = float(np.quantile(specific_boot_means, 0.025))
specific_margin_high = float(np.quantile(specific_boot_means, 0.975))
specific_winner_layers = int((specific_margins > 0).sum())

print("specificity BLUF:")
print(
    f"winner={specific_winner} | runner_up={specific_runner_up} | "
    f"specific_margin_log2={specific_margin_mean:+.2f} "
    f"[{specific_margin_low:+.2f}, {specific_margin_high:+.2f}] | "
    f"layer_wins={specific_winner_layers}/{len(list(LORA_LAYERS))}"
)
print(tabulate(specific_summary.head(16).to_pandas(), headers="keys", tablefmt="github", floatfmt="+.3f"))


# %% [markdown]
# ## Figures

# %%
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 240,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
})

top_n = min(14, summary.height)
plot_df = summary.head(top_n).to_pandas()
colors = ["#1f77b4" if kind == "ceiling" else "#ff7f0e" if name == winner else "#8c8c8c" for name, kind in zip(plot_df["subspace"], plot_df["kind"])]

fig, (ax_bar, ax_layer) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.0, 1.15]})
y = np.arange(len(plot_df))
ax_bar.barh(y, plot_df["mean_conc_B"], color=colors, alpha=0.9)
ax_bar.axvline(1.0, color="black", linestyle="--", linewidth=1.0, label="random null")
ax_bar.set_yticks(y, plot_df["subspace"])
ax_bar.invert_yaxis()
ax_bar.set_xlabel("mean held-out recovery R over LoRA layers")
ax_bar.set_title("A. Expanded hypothesis sweep")
ax_bar.grid(axis="x", alpha=0.25)
for yi, row in enumerate(plot_df.itertuples(index=False)):
    suffix = "ceiling" if row.kind == "ceiling" else f"{row.pct_ceiling:.0f}% ceil, z={row.mean_z:.1f}"
    ax_bar.text(row.mean_conc_B + 0.25, yi, suffix, va="center", fontsize=8)

layers = wide["layer"].to_numpy()
ax_layer.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="random null")
for name, color, width, style in [
    ("TaskDiff_lora_ceiling", "#1f77b4", 2.4, "--"),
    (winner, "#ff7f0e", 2.4, "-"),
    (runner_up, "#2ca02c", 1.9, "-"),
]:
    ax_layer.plot(layers, wide[name].to_numpy(), marker="o", color=color, linewidth=width, linestyle=style, label=name)
ax_layer.set_yscale("log")
ax_layer.set_xlabel("layer ℓ")
ax_layer.set_ylabel("held-out recovery R")
ax_layer.set_title("B. Winner vs runner-up vs ceiling")
ax_layer.grid(alpha=0.25, which="both")
ax_layer.legend(frameon=True)
ax_layer.text(
    0.02,
    0.03,
    f"{winner} vs {runner_up}\nlog2 margin {margin_mean:+.2f} [{margin_low:+.2f}, {margin_high:+.2f}]\npositive on {winner_layers}/14 layers",
    transform=ax_layer.transAxes,
    ha="left",
    va="bottom",
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
)

fig.suptitle("Qwen3-0.6B sycophancy LoRA: many hypotheses, one held-out-label score", y=1.02, fontsize=14)
fig.tight_layout()
main_png = OUT_DIR / "v5_hypothesis_sweep_main.png"
main_pdf = OUT_DIR / "v5_hypothesis_sweep_main.pdf"
fig.savefig(main_png, bbox_inches="tight")
fig.savefig(main_pdf, bbox_inches="tight")
plt.close(fig)


# %%
pivot_top = active.filter(pl.col("subspace").is_in(summary.head(12)["subspace"].to_list())).select(
    "layer", "subspace", "conc_in_B"
).pivot(index="subspace", on="layer", values="conc_in_B")
row_order = summary.head(12)["subspace"].to_list()
pivot_top = pivot_top.with_columns(
    order=pl.col("subspace").replace_strict(row_order, list(range(len(row_order))), return_dtype=pl.Int64)
).sort("order").drop("order")
heat = np.log2(pivot_top.drop("subspace").to_numpy())

fig, ax = plt.subplots(figsize=(10.5, 5.5))
im = ax.imshow(heat, aspect="auto", cmap="coolwarm", vmin=-1, vmax=np.nanpercentile(heat, 95))
ax.set_yticks(np.arange(len(row_order)), row_order)
ax.set_xticks(np.arange(len(layers)), [str(int(layer)) for layer in layers])
ax.set_xlabel("layer ℓ")
ax.set_title("Appendix: log2 recovery by layer for top hypotheses")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("log2 R")
fig.tight_layout()
heat_png = OUT_DIR / "v5_hypothesis_sweep_heatmap.png"
heat_pdf = OUT_DIR / "v5_hypothesis_sweep_heatmap.pdf"
fig.savefig(heat_png, bbox_inches="tight")
fig.savefig(heat_pdf, bbox_inches="tight")
plt.close(fig)


# %%
specific_plot_df = specific_summary.head(top_n).to_pandas()
specific_colors = [
    "#1f77b4" if kind == "ceiling" else "#ff7f0e" if name == specific_winner else "#8c8c8c"
    for name, kind in zip(specific_plot_df["subspace"], specific_plot_df["kind"])
]

fig, (ax_bar, ax_layer) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.0, 1.15]})
y = np.arange(len(specific_plot_df))
ax_bar.barh(y, specific_plot_df["mean_specific_conc"], color=specific_colors, alpha=0.9)
ax_bar.axvline(1.0, color="black", linestyle="--", linewidth=1.0, label="random residual null")
ax_bar.set_yticks(y, specific_plot_df["subspace"])
ax_bar.invert_yaxis()
ax_bar.set_xlabel("mean residualized recovery R over LoRA layers")
ax_bar.set_title("A. Specificity after removing clean residual PCs")
ax_bar.grid(axis="x", alpha=0.25)
for yi, row in enumerate(specific_plot_df.itertuples(index=False)):
    suffix = "ceiling" if row.kind == "ceiling" else f"{row.pct_specific_ceiling:.0f}% ceil, z={row.mean_specific_z:.1f}"
    ax_bar.text(row.mean_specific_conc + 0.25, yi, suffix, va="center", fontsize=8)

specific_layers = specific_wide["layer"].to_numpy()
ax_layer.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="random residual null")
for name, color, width, style in [
    ("TaskDiff_lora_ceiling", "#1f77b4", 2.4, "--"),
    (specific_winner, "#ff7f0e", 2.4, "-"),
    (specific_runner_up, "#2ca02c", 1.9, "-"),
]:
    ax_layer.plot(
        specific_layers,
        specific_wide[name].to_numpy(),
        marker="o",
        color=color,
        linewidth=width,
        linestyle=style,
        label=name,
    )
ax_layer.set_yscale("log")
ax_layer.set_xlabel("layer ℓ")
ax_layer.set_ylabel("residualized held-out recovery R")
ax_layer.set_title("B. Specific winner vs runner-up vs ceiling")
ax_layer.grid(alpha=0.25, which="both")
ax_layer.legend(frameon=True)
ax_layer.text(
    0.02,
    0.03,
    f"{specific_winner} vs {specific_runner_up}\nlog2 margin {specific_margin_mean:+.2f} [{specific_margin_low:+.2f}, {specific_margin_high:+.2f}]\npositive on {specific_winner_layers}/14 layers",
    transform=ax_layer.transAxes,
    ha="left",
    va="bottom",
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
)

fig.suptitle("Specificity control: remove generic clean-residual PCs, then score hypotheses", y=1.02, fontsize=14)
fig.tight_layout()
specific_png = OUT_DIR / "v5_hypothesis_sweep_specificity.png"
specific_pdf = OUT_DIR / "v5_hypothesis_sweep_specificity.pdf"
fig.savefig(specific_png, bbox_inches="tight")
fig.savefig(specific_pdf, bbox_inches="tight")
plt.close(fig)


# %% [markdown]
# ## Write conclusion and method glossary

# %%
definitions_path = OUT_DIR / "v5_hypothesis_sweep_definitions.md"
definitions = [
    "# v5 hypothesis definitions",
    "",
    "All A-side hypotheses are built without the trained LoRA. The ceiling is marked separately.",
    "",
    "| name | family | definition |",
    "|---|---|---|",
]
for candidate in all_candidates:
    definitions.append(f"| `{candidate.name}` | {candidate.family} | {candidate.definition} |")
definitions_path.write_text("\n".join(definitions) + "\n")

winner_row = summary.filter(pl.col("subspace") == winner).row(0, named=True)
runner_row = summary.filter(pl.col("subspace") == runner_up).row(0, named=True)
ceiling_row = summary.filter(pl.col("kind") == "ceiling").row(0, named=True)
specific_winner_row = specific_summary.filter(pl.col("subspace") == specific_winner).row(0, named=True)
specific_runner_row = specific_summary.filter(pl.col("subspace") == specific_runner_up).row(0, named=True)
specific_ceiling_row = specific_summary.filter(pl.col("kind") == "ceiling").row(0, named=True)
conclusion_path = OUT_DIR / "v5_hypothesis_sweep_conclusion.md"
conclusion_path.write_text(f"""# v5 hypothesis sweep conclusion

## BLUF

Expanded sweep winner: `{winner}` with mean recovery R={winner_row['mean_conc_B']:.2f}, z={winner_row['mean_z']:.1f}, and {winner_row['pct_ceiling']:.1f}% of the LoRA-fitted ceiling.

Runner-up: `{runner_up}` with mean recovery R={runner_row['mean_conc_B']:.2f}, z={runner_row['mean_z']:.1f}, and {runner_row['pct_ceiling']:.1f}% of ceiling.

Paired layer margin: log2({winner}/{runner_up}) = {margin_mean:+.2f} [{margin_low:+.2f}, {margin_high:+.2f}], positive on {winner_layers}/14 LoRA layers.

Ceiling: `{ceiling_row['subspace']}` with mean recovery R={ceiling_row['mean_conc_B']:.2f}.

## Specificity control

The raw winner is a warning sign, not a final mechanism: `layer_clean_resid_pca` uses no task/persona information and still gets {winner_row['pct_ceiling']:.1f}% of ceiling. This means raw held-out recovery is heavily influenced by generic residual-stream anisotropy.

After projecting the label and all candidates away from per-layer clean residual PCs, the specific winner is `{specific_winner}` with residualized R={specific_winner_row['mean_specific_conc']:.2f}, z={specific_winner_row['mean_specific_z']:.1f}, and {specific_winner_row['pct_specific_ceiling']:.1f}% of residualized ceiling.

Specific runner-up: `{specific_runner_up}` with residualized R={specific_runner_row['mean_specific_conc']:.2f}, z={specific_runner_row['mean_specific_z']:.1f}, and {specific_runner_row['pct_specific_ceiling']:.1f}% of residualized ceiling.

Residualized paired margin: log2({specific_winner}/{specific_runner_up}) = {specific_margin_mean:+.2f} [{specific_margin_low:+.2f}, {specific_margin_high:+.2f}], positive on {specific_winner_layers}/14 LoRA layers. Residualized ceiling `{specific_ceiling_row['subspace']}` has R={specific_ceiling_row['mean_specific_conc']:.2f}.

## What this tests

The sweep adds the hypotheses the previous notebook was missing: churn, suppressed/amplified turnover, global write, global read, downstream write-not-read, attention OV write, MLP up/gate read spaces, up_proj-input activations, and a Procrustes rotation parameterization.

## Failure modes checked

- If the added hypotheses were noise, their R values would sit near the random null R=1 and z≈0.
- If broadening the search only rediscovered the old result, the best new candidates would stay below `write_not_lm_head_read` / old `write_not_read`.
- If the winner were a layer-noise artifact, the paired log-margin CI would include 0 and layer wins would split.

## Artifacts

- Main figure: `{main_png}` and `{main_pdf}`
- Specificity figure: `{specific_png}` and `{specific_pdf}`
- Heatmap: `{heat_png}` and `{heat_pdf}`
- Per-layer scores: `{per_layer_path}`
- Residualized per-layer scores: `{specific_per_layer_path}`
- Summary table: `{summary_path}`
- Residualized summary table: `{specific_summary_path}`
- Family table: `{family_path}`
- Layer winners: `{layer_best_path}`
- Definitions: `{definitions_path}`
""")

print("wrote:")
for path in [
    per_layer_path,
    specific_per_layer_path,
    summary_path,
    specific_summary_path,
    family_path,
    layer_best_path,
    definitions_path,
    conclusion_path,
    main_png,
    main_pdf,
    specific_png,
    specific_pdf,
    heat_png,
    heat_pdf,
]:
    print(f"  {path} ({path.stat().st_size} bytes)")

print(
    "SHOULD: winner has R well above 1, positive paired margin CI if decisive, and a clear family interpretation. "
    "ELSE: the broadened search did not improve the hypothesis beyond v4."
)