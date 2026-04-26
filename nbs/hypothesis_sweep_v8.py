# %% [markdown]
# # v8 hypothesis sweep: rank-honest scoring (pct_oracle in [0,1])
#
# v7 found every non-oracle candidate landed in 5.6-7.9% of the weight
# ceiling -- a flat range. The headline `R_w_combined` ratio (energy /
# null) is hard to read because (a) the null is random orthonormal in
# d_model which may be the wrong reference manifold, and (b) every
# candidate is forced to PCS=8 so wide and narrow primitives compete on
# unequal footing.
#
# v8 changes vs v7:
# 1. **pct_oracle** is the primary metric: for each candidate at each
#    layer, oracle = top-r_eff left singular vectors of the LoRA delta
#    (where r_eff = effective rank of the candidate basis). Score =
#    `||basis.T M||_F^2 / ||oracle.T M||_F^2` in [0, 1]. Rank-honest:
#    chars_clusters (r_eff=7) is graded against rank-7 oracle, not rank-8.
# 2. Same for activations: oracle = PCA(hs_diff_B[layer], r_eff).
# 3. Joint = geometric mean of pct_oracle_act and pct_oracle_w_combined.
# 4. v7 z-scores and conc ratios kept as supplementary columns.
# 5. Limitation kept honest in the conclusion: pct_oracle is still a
#    *subspace* metric. Any primitive whose mechanism is nonlinear
#    (CHaRS-style per-cluster translations, gated MLP, token-conditional)
#    is structurally penalized -- we throw away the nonlinearity and
#    keep just the linear span.

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
import torch.nn.functional as F
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
    "logs/hypothesis_sweep_v8.verbose.log",
    level="DEBUG",
    format="{time} | {level} | {name}:{function}:{line} - {message}",
)
torch.set_grad_enabled(False)

MODEL_ID = "Qwen/Qwen3-0.6B"
W_PATH = Path(os.environ.get("W_PATH", "out/sycophancy/lora/w.pt"))
OUT_DIR = Path("out/sycophancy/lora/v8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PCS = 8
K_BROAD = 64
N_NULL = 120
LORA_LAYERS = range(8, 22)
BOOT = 20_000
RNG = np.random.default_rng(0)

PROBE_PROMPTS = [
    f"I think {claim}. Do you agree? Answer Yes or No." for claim, _q in SYCOPHANCY_TOPICS
]
FIT = PROBE_PROMPTS[: len(PROBE_PROMPTS) // 2]
EVAL = PROBE_PROMPTS[len(PROBE_PROMPTS) // 2 :]

if not W_PATH.exists():
    raise FileNotFoundError(f"missing LoRA diff: {W_PATH}")


# %% [markdown]
# ## Load model and B-side labels

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
UP_HOOKS = [f"model.layers.{i}.mlp.up_proj" for i in range(n_layers)]

lm_head_W = state.get("lm_head.weight")
if lm_head_W is None:
    lm_head_W = state["model.embed_tokens.weight"]
lm_head_W = lm_head_W.float().cpu()
d_model = lm_head_W.shape[1]
logger.info(f"loaded {MODEL_ID} | layers={n_layers} | d_model={d_model} | LoRA tensors={len(w)} | W_PATH={W_PATH}")


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


def intersect_basis(A: torch.Tensor, B: torch.Tensor, *, k: int = PCS) -> torch.Tensor:
    if A.shape[1] == 0 or B.shape[1] == 0:
        return torch.zeros(A.shape[0], 0)
    U, _s, Vh = torch.linalg.svd(A.T @ B, full_matrices=False)
    return orthonormalize(A @ U[:, :k] + B @ Vh.T[:, :k])[:, :k]


def left_svd_basis(M: torch.Tensor, k: int = PCS) -> torch.Tensor:
    if M.shape[1] == 0:
        return torch.zeros(M.shape[0], 0)
    U, _s, _Vh = torch.linalg.svd(M.float().cpu(), full_matrices=False)
    return U[:, : min(k, U.shape[1])].contiguous()


def effective_rank(basis: torch.Tensor, tol: float = 1e-6) -> int:
    """Numerical rank of an (already-orthonormal) basis.

    Most candidate bases are constructed as orthonormal columns at width
    PCS=8, but some collapse silently:
      - `chars_clusters`: centroids - mean has rank k_clusters - 1 = 7.
      - any candidate built from <PCS samples or with rank-deficient input.
    We count singular values above tol as 'live' columns. Rank-honest
    scoring grades each candidate against the optimal r_eff-dim oracle,
    not the optimal PCS-dim one.
    """
    if basis.shape[1] == 0:
        return 0
    sv = torch.linalg.svdvals(basis.float().cpu())
    return int((sv > tol * sv.max().clamp(min=1e-12)).sum().item())


def right_svd_basis(M: torch.Tensor, k: int = PCS) -> torch.Tensor:
    if M.shape[0] == 0:
        return torch.zeros(M.shape[1], 0)
    _U, _s, Vh = torch.linalg.svd(M.float().cpu(), full_matrices=False)
    return Vh[: min(k, Vh.shape[0])].T.contiguous()


def complement_basis(basis: torch.Tensor, forbidden: torch.Tensor, *, k: int = PCS) -> torch.Tensor:
    Q_forbidden = orthonormalize(forbidden)
    Q_full, R = torch.linalg.qr(Q_forbidden, mode="complete")
    rank = int((R.diag().abs() > 1e-5).sum().item()) if R.numel() else 0
    return Q_full[:, rank : rank + k].contiguous()


def project_away(basis: torch.Tensor, forbidden: torch.Tensor) -> torch.Tensor:
    P = forbidden @ forbidden.T
    return orthonormalize((torch.eye(basis.shape[0]) - P) @ basis)


def project_write_away(write_matrix: torch.Tensor, forbidden: torch.Tensor) -> torch.Tensor:
    P = forbidden @ forbidden.T
    return left_svd_basis((torch.eye(write_matrix.shape[0]) - P) @ write_matrix)


def principal_cos(A: torch.Tensor, B: torch.Tensor) -> float:
    if A.shape[1] == 0 or B.shape[1] == 0:
        return float("nan")
    return float(torch.linalg.svdvals(A.T @ B).clamp(0, 1).mean())


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    basis_by_layer: list[torch.Tensor]
    source: str
    definition: str


# %%
def texts_from_prompts(prompts: list[str], *, system: str | None = None) -> list[str]:
    if system is None:
        return prompts
    msgs = [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in prompts]
    return [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]


def capture_blocks(prompts: list[str], *, alpha: float = 0.0, system: str | None = None) -> torch.Tensor:
    texts = texts_from_prompts(prompts, system=system)
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
    texts = texts_from_prompts(prompts, system=system)
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
    texts = texts_from_prompts(prompts, system=system)
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


def capture_token_blocks_and_final_attn(
    prompts: list[str], *, system: str
) -> tuple[torch.Tensor, torch.Tensor]:
    texts = texts_from_prompts(prompts, system=system)
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
    seq_idx = enc.attention_mask.sum(-1) - 1
    out = model(**enc, output_hidden_states=True, output_attentions=True)
    if out.attentions is None or out.hidden_states is None:
        raise RuntimeError("model did not return attentions/hidden_states; attention-selected bases need eager attentions")

    b = enc.input_ids.shape[0]
    max_len = int(seq_idx.max().item()) + 1
    hs_by_layer = []
    attn_by_layer = []
    for layer in range(n_layers):
        hs = out.hidden_states[layer + 1].float().cpu()
        attn = out.attentions[layer].float().cpu()
        hs_aligned = hs.new_zeros(b, max_len, d_model)
        attn_aligned = hs.new_zeros(b, max_len)
        for sample in range(b):
            n = int(seq_idx[sample].item()) + 1
            hs_aligned[sample, -n:] = hs[sample, :n]
            attn_aligned[sample, -n:] = attn[sample, :, n - 1, :n].mean(0)
        hs_by_layer.append(hs_aligned)
        attn_by_layer.append(attn_aligned)
    return torch.stack(hs_by_layer), torch.stack(attn_by_layer)


def left_pad_sequence_dim(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.shape[2] == target_len:
        return x
    if x.shape[2] > target_len:
        raise ValueError(f"cannot pad length {x.shape[2]} down to {target_len}")
    pad_shape = (*x.shape[:2], target_len - x.shape[2], *x.shape[3:])
    return torch.cat([x.new_zeros(pad_shape), x], dim=2)


def attention_selected_taskdiff_bases(
    hs_pos_tokens: torch.Tensor,
    hs_neg_tokens: torch.Tensor,
    attn_pos: torch.Tensor,
    attn_neg: torch.Tensor,
) -> dict[str, list[torch.Tensor]]:
    target_len = max(hs_pos_tokens.shape[2], hs_neg_tokens.shape[2])
    hs_pos = left_pad_sequence_dim(hs_pos_tokens, target_len)
    hs_neg = left_pad_sequence_dim(hs_neg_tokens, target_len)
    a_pos = left_pad_sequence_dim(attn_pos[:, :, :, None], target_len).squeeze(-1)
    a_neg = left_pad_sequence_dim(attn_neg[:, :, :, None], target_len).squeeze(-1)
    diff = hs_pos - hs_neg
    diff_norm = diff.norm(dim=-1)
    norm_scale = diff_norm.sum(dim=(1, 2), keepdim=True) / (diff_norm.gt(0).sum(dim=(1, 2), keepdim=True) + 1e-12)
    weights = {
        "attn_min_taskdiff": torch.minimum(a_pos, a_neg),
        "attn_max_taskdiff": torch.maximum(a_pos, a_neg),
        "attn_diff_taskdiff": (a_pos - a_neg).abs(),
        "attn_min_x_diffnorm_taskdiff": torch.minimum(a_pos, a_neg) * diff_norm / (norm_scale + 1e-12),
    }
    bases = {}
    for name, weight in weights.items():
        layer_bases = []
        for layer in range(n_layers):
            samples = diff[layer].reshape(-1, d_model)
            w_flat = weight[layer].reshape(-1)
            layer_bases.append(pca(samples * torch.sqrt(w_flat[:, None] + 1e-12), PCS))
        bases[name] = layer_bases
    return bases


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
up_clean_fit = capture_up_inputs(FIT)
up_persona_pos_fit = capture_up_inputs(FIT, system=SYCOPHANCY_POS_PERSONAS[0])
up_persona_neg_fit = capture_up_inputs(FIT, system=SYCOPHANCY_NEG_PERSONAS[0])
up_diff_A_fit = up_persona_pos_fit - up_persona_neg_fit
up_written_pos_fit = capture_up_outputs_written(FIT, system=SYCOPHANCY_POS_PERSONAS[0])
up_written_neg_fit = capture_up_outputs_written(FIT, system=SYCOPHANCY_NEG_PERSONAS[0])
up_written_diff_A_fit = up_written_pos_fit - up_written_neg_fit
hs_pos_tokens_fit, attn_pos_fit = capture_token_blocks_and_final_attn(FIT, system=SYCOPHANCY_POS_PERSONAS[0])
hs_neg_tokens_fit, attn_neg_fit = capture_token_blocks_and_final_attn(FIT, system=SYCOPHANCY_NEG_PERSONAS[0])
attn_selected_taskdiff = attention_selected_taskdiff_bases(
    hs_pos_tokens_fit, hs_neg_tokens_fit, attn_pos_fit, attn_neg_fit
)
logger.info(f"captured label={tuple(hs_diff_B.shape)} | clean={tuple(hs_clean_fit.shape)} | up={tuple(up_clean_fit.shape)} | attn_tokens={tuple(hs_pos_tokens_fit.shape)}")


# %% [markdown]
# ## Build A-side candidate bases

# %%
def expand_rows_to(W_small: torch.Tensor, out_rows: int) -> torch.Tensor:
    if W_small.shape[0] == out_rows:
        return W_small
    repeats = out_rows // W_small.shape[0]
    if repeats * W_small.shape[0] != out_rows:
        raise ValueError(f"cannot repeat rows from {tuple(W_small.shape)} to {out_rows}")
    return W_small.repeat_interleave(repeats, dim=0)


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


def read_stack(layer: int, projs: tuple[str, ...]) -> torch.Tensor:
    return torch.cat([state[f"model.layers.{layer}.{proj}"].float().cpu() for proj in projs], dim=0)


def read_gram(layer: int) -> torch.Tensor:
    W = read_stack(layer, (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "mlp.up_proj.weight",
        "mlp.gate_proj.weight",
    ))
    return W.T @ W


def suppressed_features(acts: torch.Tensor) -> torch.Tensor:
    mag = acts.abs().permute(1, 0, 2)
    delta = mag[:, 1:] - mag[:, :-1]
    return torch.minimum(torch.relu(delta).sum(1), torch.relu(-delta).sum(1))


def amplified_features(acts: torch.Tensor) -> torch.Tensor:
    mag = acts.abs().permute(1, 0, 2)
    return torch.relu(mag[:, -1] - mag[:, 0])


def added_features(acts: torch.Tensor) -> torch.Tensor:
    mag = acts.abs().permute(1, 0, 2)
    return torch.relu(mag[:, 1:] - mag[:, :-1]).sum(1)


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


def kmeans_centroid_basis(samples: torch.Tensor, *, k_clusters: int = PCS, iters: int = 8) -> torch.Tensor:
    centered = samples.float().cpu() - samples.float().cpu().mean(0, keepdim=True)
    order = torch.argsort(centered.norm(dim=1), descending=True)
    centroids = centered[order[: min(k_clusters, centered.shape[0])]].clone()
    for _ in range(iters):
        dist = torch.cdist(centered, centroids)
        assign = dist.argmin(dim=1)
        new_centroids = []
        for idx in range(centroids.shape[0]):
            members = centered[assign == idx]
            new_centroids.append(members.mean(0) if members.shape[0] else centroids[idx])
        centroids = torch.stack(new_centroids)
    return pca(centroids - centroids.mean(0, keepdim=True), PCS)


_u_lm, _s_lm, vh_lm = torch.linalg.svd(lm_head_W, full_matrices=False)
lm_head_read = vh_lm[:PCS].T.contiguous()
logits_null = vh_lm[-PCS:].T.contiguous()
lm_read_broad = vh_lm[:K_BROAD].T.contiguous()

read_grams = [read_gram(layer) for layer in range(n_layers)]
global_read_gram = sum(read_grams, torch.zeros(d_model, d_model)) + lm_head_W.T @ lm_head_W
global_read = basis_from_gram(global_read_gram, PCS)
global_read_broad = basis_from_gram(global_read_gram, K_BROAD)
global_write_cols = torch.cat([write_cols(layer) for layer in range(n_layers)], dim=1)
global_write = left_svd_basis(global_write_cols)

downstream_read_broad = []
running = lm_head_W.T @ lm_head_W
for layer in reversed(range(n_layers)):
    if layer < n_layers - 1:
        running = running + read_grams[layer + 1]
    downstream_read_broad.append(basis_from_gram(running, K_BROAD))
downstream_read_broad = list(reversed(downstream_read_broad))

eye = torch.eye(d_model)
P_lm = lm_read_broad @ lm_read_broad.T
P_global_read = global_read_broad @ global_read_broad.T

candidate_list: list[Candidate] = []


def add(name: str, family: str, basis_by_layer: list[torch.Tensor], definition: str, source: str = "v5") -> None:
    if len(basis_by_layer) != n_layers:
        raise ValueError(f"{name} has {len(basis_by_layer)} layers, expected {n_layers}")
    for layer, B in enumerate(basis_by_layer):
        if B.shape[0] != d_model:
            raise ValueError(f"{name}[{layer}] shape={tuple(B.shape)}, expected first dim {d_model}")
        if B.shape[1] > 0:
            err = (B.T @ B - torch.eye(B.shape[1])).abs().max().item()
            if err > 1e-3:
                raise ValueError(f"{name}[{layer}] is not orthonormal: maxerr={err}")
    candidate_list.append(Candidate(name, family, basis_by_layer, source, definition))


add("lm_head_read", "W:unembed", [lm_head_read] * n_layers, "top right singular vectors of lm_head")
add("logits_null", "W:unembed", [logits_null] * n_layers, "bottom right singular vectors of lm_head")
add("global_read", "W:read", [global_read] * n_layers, "top eigenspace of all q/k/v/up/gate reads + lm_head")
add("global_write", "W:write", [global_write] * n_layers, "top left singular vectors of all o/down residual writers")
add("global_write_not_global_read", "W:write-not-read", [left_svd_basis((eye - P_global_read) @ global_write_cols)] * n_layers, "global residual write projected away from global read directions")

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
qk_circuit = []
input_super = []
kv_super = []
gate_kernel = []
attention_sink = []
causally_isolated = []
input_super_not_lm = []
gate_active_written = []
chars_clusters = []
for layer in range(n_layers):
    up = state[f"model.layers.{layer}.mlp.up_proj.weight"].float().cpu()
    gate = state[f"model.layers.{layer}.mlp.gate_proj.weight"].float().cpu()
    q = state[f"model.layers.{layer}.self_attn.q_proj.weight"].float().cpu()
    k = state[f"model.layers.{layer}.self_attn.k_proj.weight"].float().cpu()
    v = state[f"model.layers.{layer}.self_attn.v_proj.weight"].float().cpu()
    W_o = state[f"model.layers.{layer}.self_attn.o_proj.weight"].float().cpu()
    W_down = state[f"model.layers.{layer}.mlp.down_proj.weight"].float().cpu()

    k_for_q = expand_rows_to(k, q.shape[0])
    v_for_o = expand_rows_to(v, W_o.shape[1])
    clean_up_x = up_clean_fit[layer]
    mean_gate = F.silu(clean_up_x @ gate.T).mean(0)
    gate_active = F.silu(clean_up_x @ gate.T) * (clean_up_x @ up.T)

    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = W_o.shape[1] // n_heads
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    e_bos = state["model.embed_tokens.weight"][bos_id].float().cpu()
    sink_vecs = []
    for head in range(n_heads):
        kv_head = head * n_kv_heads // n_heads
        o_h = W_o[:, head * head_dim : (head + 1) * head_dim]
        v_h = v[kv_head * head_dim : (kv_head + 1) * head_dim]
        sink_vecs.append(o_h @ (v_h @ e_bos))

    mlp_up_read.append(right_svd_basis(up))
    mlp_gate_read.append(right_svd_basis(gate))
    attn_qkv_read.append(right_svd_basis(torch.cat([q, k, v], dim=0)))
    attn_ov_write.append(left_svd_basis(W_o @ v_for_o))
    mlp_roundtrip.append(left_svd_basis(W_down @ up))
    qk_circuit.append(left_svd_basis(q.T @ k_for_q))
    input_super.append(right_svd_basis(torch.cat([q, k, v, up, gate], dim=0)))
    kv_super.append(right_svd_basis(torch.cat([k, v], dim=0)))
    gate_kernel.append(left_svd_basis(W_down @ (mean_gate[:, None] * up)))
    attention_sink.append(pca(torch.stack(sink_vecs), PCS))
    forbidden = orthonormal_union(input_super[-1], kv_super[-1], lm_read_broad)
    causally_isolated.append(project_write_away(write_cols(layer), forbidden))
    input_super_not_lm.append(project_away(input_super[-1], lm_read_broad)[:, :PCS])
    gate_active_written.append(pca(gate_active @ W_down.T, PCS))
    chars_samples = torch.cat([hs_clean_fit[layer], hs_persona_pos_fit[layer], hs_persona_neg_fit[layer]], dim=0)
    chars_clusters.append(kmeans_centroid_basis(chars_samples))

add("mlp_up_read", "W:read", mlp_up_read, "right singular vectors of W_up")
add("mlp_gate_read", "W:read", mlp_gate_read, "right singular vectors of W_gate")
add("attn_qkv_read", "W:read", attn_qkv_read, "right singular vectors of concatenated W_q/W_k/W_v")
add("attn_ov_write", "W:OV", attn_ov_write, "left singular vectors of W_o W_v")
add("mlp_roundtrip_write", "W:MLP", mlp_roundtrip, "left singular vectors of W_down W_up residual-to-residual map")
add("qk_circuit", "W:QK", qk_circuit, "left singular vectors of W_q^T W_k after GQA row expansion", source="external-v6-plan")
add("input_super", "W:read", input_super, "right singular vectors of [W_q; W_k; W_v; W_up; W_gate]", source="external-v6-plan")
add("kv_super", "W:read", kv_super, "right singular vectors of [W_k; W_v]", source="external-v6-plan")
add("gate_kernel", "W:MLP", gate_kernel, "left singular vectors of W_down diag(E silu(W_gate h)) W_up", source="external-v6-plan")
add("attention_sink", "W:OV", attention_sink, "PCA over per-head W_o^h W_v^h e_BOS sink vectors", source="external-v6-plan")
add("causally_isolated", "W:write-not-read", causally_isolated, "write subspace projected away from input-read, KV, and lm_head read bases", source="external-v6-plan")
add("input_super_not_lm_read", "W:read", input_super_not_lm, "input_super projected away from lm_head top read directions", source="external-v6-plan")

suppressed = pca(suppressed_features(hs_clean_fit), PCS)
amplified = pca(amplified_features(hs_clean_fit), PCS)
added = pca(added_features(hs_clean_fit), PCS)
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
add("added_features", "act:clean", [added] * n_layers, "PCA of positive layer-to-layer magnitude additions", source="external-v6-plan")
add("global_clean_resid_pca", "act:baseline", [global_clean_pca] * n_layers, "PCA of all clean base residual activations")
add("global_persona_resid_pca", "act:baseline", [global_persona_pca] * n_layers, "PCA of persona residual activations without differencing")
add("layer_clean_resid_pca", "act:baseline", [pca(hs_clean_fit[layer], PCS) for layer in range(n_layers)], "per-layer PCA of clean base residual activations")
add("TaskDiff_contrast", "act:persona", [pca(hs_diff_A_fit[layer], PCS) for layer in range(n_layers)], "PCA of persona+ minus persona- residual activations")
add("attn_min_taskdiff", "act:attn-selected", attn_selected_taskdiff["attn_min_taskdiff"], "PCA of tokenwise persona TaskDiff weighted by min(pos, neg) final-token attention", source="external-v6-plan")
add("attn_max_taskdiff", "act:attn-selected", attn_selected_taskdiff["attn_max_taskdiff"], "PCA of tokenwise persona TaskDiff weighted by max(pos, neg) final-token attention", source="external-v6-plan")
add("attn_diff_taskdiff", "act:attn-selected", attn_selected_taskdiff["attn_diff_taskdiff"], "PCA of tokenwise persona TaskDiff weighted by abs(pos - neg) final-token attention", source="external-v6-plan")
add("attn_min_x_diffnorm_taskdiff", "act:attn-selected", attn_selected_taskdiff["attn_min_x_diffnorm_taskdiff"], "PCA of tokenwise persona TaskDiff weighted by min(pos, neg) attention times tokenwise diff norm", source="external-v6-plan")
add("up_proj_input_contrast", "act:up_proj", [pca(up_diff_A_fit[layer], PCS) for layer in range(n_layers)], "PCA of persona contrast in inputs to mlp.up_proj")
add("up_proj_output_written_contrast", "act:up_proj", [pca(up_written_diff_A_fit[layer], PCS) for layer in range(n_layers)], "PCA of persona contrast after W_up mapped back by W_down")
add("gate_active_written", "act:MLP", gate_active_written, "PCA of silu(W_gate h) * W_up h mapped back by W_down on clean probes", source="external-v6-plan")
add("chars_clusters", "act:cluster", chars_clusters, "CHaRS-style PCA of k-means centroid differences over clean/persona activations", source="external-v6-plan")
add("churn", "act:clean", [pca(hs_clean_fit[min(layer + 1, n_layers - 1)] - hs_clean_fit[layer], PCS) for layer in range(n_layers)], "PCA of signed clean residual change h_{l+1}-h_l")
add("rotation_contrast", "act:rotation", [procrustes_rotation_basis(hs_persona_neg_fit[layer], hs_persona_pos_fit[layer]) for layer in range(n_layers)], "skew generator from persona- to persona+ Procrustes rotation")
add("qk_x_chars_clusters", "compound", [intersect_basis(qk_circuit[layer], chars_clusters[layer]) for layer in range(n_layers)], "bisector intersection of qk_circuit and CHaRS-style activation clusters", source="external-v6-plan")
add("WNR_union_TaskDiff", "compound", [orthonormal_union(write_not_downstream_read[layer], pca(hs_diff_A_fit[layer], PCS)) for layer in range(n_layers)], "rank-expanded union of write_not_downstream_read and TaskDiff_contrast")

ceiling = Candidate(
    "TaskDiff_lora_fit",
    "act:cluster",
    [pca(hs_diff_B_fit[layer], PCS) for layer in range(n_layers)],
    "B-side",
    "PCA of LoRA FIT-half label (held-out from scoring eval); informative candidate, NOT an oracle. v7 mislabeled this as 'ceiling'.",
)

logger.info(f"built {len(candidate_list)} A-side candidates + ceiling")


# %% [markdown]
# ## Activation and weight scoring

# %%
_W_TENSOR_NAMES = ("self_attn.o_proj.weight", "mlp.down_proj.weight")
_dropped_keys_logged = False


def lora_weight_tensors(layer: int) -> dict[str, torch.Tensor]:
    """Per-tensor LoRA delta in residual-output (d_model row) space.

    v6 returned a single concatenated matrix; v7 keeps tensors separate so R_w
    isn't silently Frobenius-weighted toward whichever tensor has more
    parameters (down_proj has ~3x o_proj). Logs which residual-output keys
    were skipped (for debugging if Qwen renames projections).
    """
    global _dropped_keys_logged
    out: dict[str, torch.Tensor] = {}
    dropped = []
    for proj in _W_TENSOR_NAMES:
        key = f"model.layers.{layer}.{proj}"
        if key not in w:
            dropped.append((key, "missing-from-LoRA"))
            continue
        W = w[key].float().cpu()
        if W.shape[0] != d_model:
            dropped.append((key, f"shape={tuple(W.shape)} d_model={d_model}"))
            continue
        out[proj] = W
    if dropped and not _dropped_keys_logged:
        logger.info(f"lora_weight_tensors layer={layer} dropped: {dropped}")
        _dropped_keys_logged = True
    return out


def lora_weight_matrix(layer: int) -> torch.Tensor:
    """v6-compatible concatenated form, retained for dw_left_basis only."""
    tensors = lora_weight_tensors(layer)
    if not tensors:
        return torch.zeros(d_model, 0)
    return torch.cat(list(tensors.values()), dim=1)


act_null_cache: dict[tuple[int, int], tuple[float, float]] = {}
w_null_cache: dict[tuple[int, int, str | None], tuple[float, float]] = {}

# Rank-honest oracle caches.
_act_oracle_cache: dict[tuple[int, int], float] = {}  # (layer, r) -> max E[per-example energy frac]
_w_spectrum_cache: dict[tuple[int, str], torch.Tensor] = {}  # (layer, tensor) -> sorted s^2 of M


def act_oracle_energy_frac(layer: int, r: int) -> float:
    """Best `energy_frac_act` any rank-r basis can achieve.

    `energy_frac_act` is the mean over examples of per-example normalized
    energy: E[ ||x_i^T B||^2 / ||x_i||^2 ]. This is NOT maximized by PCA of
    raw samples (which optimizes the Frobenius-weighted version) but by
    PCA of L2-normalized samples. Compute the optimal basis for each layer
    and cache the resulting frac so candidates can be scored against it.
    """
    if r <= 0:
        return 0.0
    cache_key = (layer, r)
    if cache_key not in _act_oracle_cache:
        X = hs_diff_B[layer].float().cpu()
        norms = X.norm(dim=1, keepdim=True).clamp(min=1e-12)
        Xn = X / norms
        # Optimal rank-r basis for E[||x_i^T B||^2 / ||x_i||^2] is top-r right
        # SVs of Xn (which equals top-r right SVs of (Xn^T Xn) eigenvectors).
        _U, _s, Vh = torch.linalg.svd(Xn, full_matrices=False)
        B = Vh[: min(r, Vh.shape[0])].T.contiguous()
        per_example = (X @ B).pow(2).sum(1) / X.pow(2).sum(1).clamp(min=1e-12)
        _act_oracle_cache[cache_key] = float(per_example.mean())
    return _act_oracle_cache[cache_key]


def w_oracle_energy_frac(layer: int, r: int, tensor_name: str) -> float:
    """Best fraction of LoRA-tensor Frobenius mass any rank-r left basis captures."""
    if r <= 0:
        return 0.0
    cache_key = (layer, tensor_name)
    if cache_key not in _w_spectrum_cache:
        if tensor_name == "_balanced":
            tensors = lora_weight_tensors(layer)
            cols = []
            for key in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
                M = tensors.get(key)
                if M is None:
                    continue
                cols.append(M / (M.pow(2).sum().sqrt() + 1e-12))
            if not cols:
                _w_spectrum_cache[cache_key] = torch.zeros(0)
                return 0.0
            M_bal = torch.cat(cols, dim=1)
            s = torch.linalg.svdvals(M_bal.float().cpu())
        else:
            tensors = lora_weight_tensors(layer)
            M = tensors.get(tensor_name)
            if M is None:
                _w_spectrum_cache[cache_key] = torch.zeros(0)
                return 0.0
            s = torch.linalg.svdvals(M.float().cpu())
        _w_spectrum_cache[cache_key] = s.pow(2)
    s2 = _w_spectrum_cache[cache_key]
    if s2.numel() == 0:
        return 0.0
    total = s2.sum().clamp(min=1e-12)
    return float(s2[: min(r, s2.numel())].sum() / total)


def act_null_stats(layer: int, rank: int) -> tuple[float, float]:
    key = (layer, rank)
    if key in act_null_cache:
        return act_null_cache[key]
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
    act_null_cache[key] = stats
    return stats


def w_null_stats(layer: int, rank: int, tensor_name: str | None = None) -> tuple[float, float]:
    """Random-orthonormal null for the weight concentration ratio.

    If tensor_name is None, uses the v6-style concatenated matrix (kept for
    backward-compat with diagnostics). Otherwise scores against a single LoRA
    tensor (o_proj or down_proj) so per-tensor R_w can be properly normalized.
    """
    key = (layer, rank, tensor_name)
    if key in w_null_cache:
        return w_null_cache[key]
    if tensor_name is None:
        M = lora_weight_matrix(layer)
    else:
        tensors = lora_weight_tensors(layer)
        M = tensors.get(tensor_name, torch.zeros(d_model, 0))
    if M.shape[1] == 0:
        stats = (float("nan"), float("nan"))
        w_null_cache[key] = stats
        return stats
    d = M.shape[0]
    total = M.pow(2).sum() + 1e-12
    null = rank / d
    seed_bump = 0 if tensor_name is None else (1 + hash(tensor_name) % 1000)
    gen = torch.Generator(device=M.device).manual_seed(20_000 + 97 * layer + rank + 7919 * seed_bump)
    values = []
    for _ in range(N_NULL):
        rb, _ = torch.linalg.qr(torch.randn(d, rank, generator=gen, device=M.device, dtype=M.dtype))
        values.append(((rb.T @ M).pow(2).sum() / total).item() / null)
    arr = torch.tensor(values)
    stats = (float(arr.mean()), float(arr.std(unbiased=True)))
    w_null_cache[key] = stats
    return stats


def concentration_act(layer: int, basis: torch.Tensor) -> dict[str, float]:
    samples = hs_diff_B[layer]
    rank = basis.shape[1]
    if rank == 0:
        return {
            "conc_act": 0.0,
            "z_act": 0.0,
            "energy_frac_act": 0.0,
            "pct_oracle_act": 0.0,
            "r_eff_act": 0,
        }
    total = samples.pow(2).sum(1) + 1e-12
    energy_frac = ((samples @ basis).pow(2).sum(1) / total).mean().item()
    conc = energy_frac / (rank / samples.shape[1])
    null_mean, null_std = act_null_stats(layer, rank)
    r_eff = effective_rank(basis)
    oracle_frac = act_oracle_energy_frac(layer, r_eff)
    pct_oracle = energy_frac / max(oracle_frac, 1e-12) if oracle_frac > 0 else float("nan")
    return {
        "conc_act": conc,
        "z_act": (conc - null_mean) / (null_std + 1e-12),
        "energy_frac_act": energy_frac,
        "pct_oracle_act": pct_oracle,
        "r_eff_act": r_eff,
    }


def concentration_w(layer: int, basis: torch.Tensor) -> dict[str, float]:
    """Per-tensor weight concentration + Frobenius-balanced combined.

    v6 returned a single conc_w that silently weighted by tensor size
    (down_proj has ~3x the params of o_proj). v7 reports each tensor
    separately so write-side hypotheses can be ranked by either, and a
    'combined' score that normalizes each tensor to unit Frobenius first
    (size-balanced).

    v8 adds `pct_oracle_w_*`: candidate's energy_frac divided by the
    optimal rank-r_eff oracle's energy_frac on the same tensor (top-r_eff
    left singular vectors). In [0, 1]. Rank-honest: a candidate that
    silently collapses to r_eff < PCS is graded against the same-rank
    oracle, not the full PCS-rank one.
    """
    rank = basis.shape[1]
    r_eff = effective_rank(basis)
    tensors = lora_weight_tensors(layer)
    out: dict[str, float] = {"r_eff_w": r_eff}
    if rank == 0 or not tensors:
        for name in ("oproj", "downproj", "combined"):
            out[f"conc_w_{name}"] = float("nan")
            out[f"z_w_{name}"] = float("nan")
            out[f"energy_frac_w_{name}"] = float("nan")
            out[f"pct_oracle_w_{name}"] = float("nan")
        return out

    # Per-tensor scores
    name_to_key = {"oproj": "self_attn.o_proj.weight", "downproj": "mlp.down_proj.weight"}
    balanced_M_cols = []
    for short, key in name_to_key.items():
        M = tensors.get(key)
        if M is None:
            out[f"conc_w_{short}"] = float("nan")
            out[f"z_w_{short}"] = float("nan")
            out[f"energy_frac_w_{short}"] = float("nan")
            out[f"pct_oracle_w_{short}"] = float("nan")
            continue
        total = M.pow(2).sum() + 1e-12
        energy_frac = ((basis.T @ M).pow(2).sum() / total).item()
        conc = energy_frac / (rank / M.shape[0])
        null_mean, null_std = w_null_stats(layer, rank, key)
        out[f"conc_w_{short}"] = conc
        out[f"z_w_{short}"] = (conc - null_mean) / (null_std + 1e-12)
        out[f"energy_frac_w_{short}"] = energy_frac
        oracle_frac = w_oracle_energy_frac(layer, r_eff, key)
        out[f"pct_oracle_w_{short}"] = energy_frac / max(oracle_frac, 1e-12) if oracle_frac > 0 else float("nan")
        # Frobenius-balanced combined: each tensor normalized to unit Frobenius
        balanced_M_cols.append(M / (M.pow(2).sum().sqrt() + 1e-12))

    # Combined: balanced concat (each tensor unit-Frobenius), then standard score
    if balanced_M_cols:
        M_bal = torch.cat(balanced_M_cols, dim=1)
        total_bal = M_bal.pow(2).sum() + 1e-12
        energy_frac_bal = ((basis.T @ M_bal).pow(2).sum() / total_bal).item()
        conc_bal = energy_frac_bal / (rank / M_bal.shape[0])
        # Null for balanced combined: rebuild on the fly (cheap, cached by key)
        bal_key = (layer, rank, "_balanced")
        if bal_key not in w_null_cache:
            d = M_bal.shape[0]
            null = rank / d
            gen = torch.Generator(device=M_bal.device).manual_seed(30_000 + 97 * layer + rank)
            values = []
            for _ in range(N_NULL):
                rb, _ = torch.linalg.qr(torch.randn(d, rank, generator=gen, device=M_bal.device, dtype=M_bal.dtype))
                values.append(((rb.T @ M_bal).pow(2).sum() / total_bal).item() / null)
            arr = torch.tensor(values)
            w_null_cache[bal_key] = (float(arr.mean()), float(arr.std(unbiased=True)))
        null_mean, null_std = w_null_cache[bal_key]
        out["conc_w_combined"] = conc_bal
        out["z_w_combined"] = (conc_bal - null_mean) / (null_std + 1e-12)
        out["energy_frac_w_combined"] = energy_frac_bal
        oracle_frac_bal = w_oracle_energy_frac(layer, r_eff, "_balanced")
        out["pct_oracle_w_combined"] = (
            energy_frac_bal / max(oracle_frac_bal, 1e-12) if oracle_frac_bal > 0 else float("nan")
        )
    else:
        out["conc_w_combined"] = float("nan")
        out["z_w_combined"] = float("nan")
        out["energy_frac_w_combined"] = float("nan")
        out["pct_oracle_w_combined"] = float("nan")
    return out


def dw_left_basis(layer: int) -> torch.Tensor:
    return left_svd_basis(lora_weight_matrix(layer))


def axis_kind_for(family: str) -> str:
    """Tag whether a hypothesis is read-side, write-side, or mixed in d_model.

    Read-side bases (input projections) trivially live in d_model just like the
    write-side LoRA delta does, so R_w runs without error. But high R_w for a
    read-side basis means \"this read direction happens to coincide with the
    LoRA write direction\", not \"this primitive captures the write geometry\".
    Read-side rows are reported separately and excluded from the joint W-axis
    ranking. See docs/review/v6_hypothesis_review.md concern #3.
    """
    if family == "ceiling":
        return "ceiling"
    if family in ("W:read", "W:unembed"):
        return "read"
    if family in ("W:write", "W:write-not-read", "W:OV", "W:MLP"):
        return "write"
    if family.startswith("act:") or family in ("W:QK", "compound"):
        return "mixed"
    return "mixed"


# Two oracles, one per axis:
# - w_oracle: top-PCS left singular vectors of the LoRA delta. Defines
#   pct_oracle_w_combined ~ 1.0 by construction. Off-axis (act) score is
#   whatever it happens to be, no reason for it to be high.
# - act_oracle: top-PCS PCA of L2-normalized hs_diff_B (eval set). Defines
#   pct_oracle_act ~ 1.0 by construction. This is the optimal basis for the
#   per-example normalized energy formula in concentration_act. NOTE: in-sample
#   (computed from the same eval set we score on) so it is the achievable
#   upper bound on these data, not a generalization claim.
def act_oracle_basis(layer: int) -> torch.Tensor:
    X = hs_diff_B[layer].float().cpu()
    norms = X.norm(dim=1, keepdim=True).clamp(min=1e-12)
    Xn = X / norms
    _U, _s, Vh = torch.linalg.svd(Xn, full_matrices=False)
    return Vh[: PCS].T.contiguous()


weight_ceiling = Candidate(
    "w_oracle",
    "ceiling",
    [dw_left_basis(layer) for layer in range(n_layers)],
    "B-side",
    "Top-PCS left singular vectors of the LoRA residual-output delta. Defines pct_oracle_w_combined = 1.0 by construction. (was 'dW_left_basis_ceiling' in v8.0.)",
)
act_ceiling = Candidate(
    "act_oracle",
    "ceiling",
    [act_oracle_basis(layer) for layer in range(n_layers)],
    "B-side",
    "Top-PCS right singular vectors of L2-normalized hs_diff_B (eval). Defines pct_oracle_act = 1.0 by construction (in-sample upper bound).",
)


all_candidates = [*candidate_list, ceiling, weight_ceiling, act_ceiling]
dw_bases = [dw_left_basis(layer) for layer in range(n_layers)]
rows = []
for layer in range(n_layers):
    for candidate in all_candidates:
        basis = candidate.basis_by_layer[layer]
        rows.append({
            "layer": layer,
            "subspace": candidate.name,
            "family": candidate.family,
            "axis_kind": axis_kind_for(candidate.family),
            "source": candidate.source,
            "kind": "ceiling" if candidate.family == "ceiling" else "A-hypothesis",
            "rank": basis.shape[1],
            **concentration_act(layer, basis),
            **concentration_w(layer, basis),
            "cos_with_dW": principal_cos(basis, dw_bases[layer]),
        })

per_layer = pl.DataFrame(rows)
per_layer_path = OUT_DIR / "v8_per_layer.csv"
per_layer.write_csv(per_layer_path)

active = per_layer.filter(pl.col("layer").is_in(list(LORA_LAYERS)))
summary = (
    active.group_by(["subspace", "family", "axis_kind", "source", "kind"])
    .agg(
        # Primary metric (rank-honest): pct of optimal-rank-r_eff oracle.
        pl.col("pct_oracle_act").mean().alias("mean_pct_oracle_act"),
        pl.col("pct_oracle_w_combined").mean().alias("mean_pct_oracle_w_combined"),
        pl.col("pct_oracle_w_oproj").mean().alias("mean_pct_oracle_w_oproj"),
        pl.col("pct_oracle_w_downproj").mean().alias("mean_pct_oracle_w_downproj"),
        # Supplementary: v7-style concentration ratios + z scores.
        pl.col("conc_act").mean().alias("mean_conc_act"),
        pl.col("z_act").mean().alias("mean_z_act"),
        pl.col("energy_frac_act").mean().alias("mean_energy_frac_act"),
        pl.col("conc_w_combined").mean().alias("mean_conc_w_combined"),
        pl.col("z_w_combined").mean().alias("mean_z_w_combined"),
        pl.col("energy_frac_w_combined").mean().alias("mean_energy_frac_w_combined"),
        pl.col("cos_with_dW").mean().alias("mean_cos_dW"),
        pl.col("rank").mean().alias("mean_rank"),
        pl.col("r_eff_w").mean().alias("mean_r_eff_w"),
        pl.col("r_eff_act").mean().alias("mean_r_eff_act"),
    )
    .with_columns(
        # v8 joint score: geometric mean of pct_oracle_act and pct_oracle_w_combined.
        # Both are in [0, 1] so the joint is also in [0, 1] -- 1.0 means
        # "the candidate IS the optimal rank-r_eff subspace on both axes".
        joint_pct_oracle=(
            (pl.col("mean_pct_oracle_act").log() + pl.col("mean_pct_oracle_w_combined").log()) / 2
        ).exp(),
        act_w_gap_log2=(
            pl.col("mean_pct_oracle_act").log(2) - pl.col("mean_pct_oracle_w_combined").log(2)
        ),
    )
    .sort("joint_pct_oracle", descending=True)
)

summary_path = OUT_DIR / "v8_summary.tsv"
summary.write_csv(summary_path, separator="\t")

# Sanity: each oracle should report pct_oracle ~ 1.0 on its own axis by
# construction. They are NOT expected to score high on the off-axis.
weight_ceiling_pct = float(
    summary.filter(pl.col("subspace") == "w_oracle")["mean_pct_oracle_w_combined"][0]
)
act_ceiling_pct = float(
    summary.filter(pl.col("subspace") == "act_oracle")["mean_pct_oracle_act"][0]
)
logger.info(
    f"oracle sanity: w_oracle pct_oracle_w_combined={weight_ceiling_pct:.4f} "
    f"(SHOULD ~ 1.0; basis IS top-r_eff left SVD of dW). "
    f"act_oracle pct_oracle_act={act_ceiling_pct:.4f} "
    f"(SHOULD ~ 1.0; basis IS top-r_eff right SVD of L2-normalized hs_diff_B)."
)

# Convenience: percent-scale view (multiply pct_oracle columns by 100).
summary_pct = summary.with_columns(
    pct_oracle_act_100=100 * pl.col("mean_pct_oracle_act"),
    pct_oracle_w_combined_100=100 * pl.col("mean_pct_oracle_w_combined"),
    pct_oracle_w_oproj_100=100 * pl.col("mean_pct_oracle_w_oproj"),
    pct_oracle_w_downproj_100=100 * pl.col("mean_pct_oracle_w_downproj"),
    joint_pct_oracle_100=100 * pl.col("joint_pct_oracle"),
)
summary_pct_path = OUT_DIR / "v8_summary_pct.tsv"
summary_pct.write_csv(summary_pct_path, separator="\t")

# Separate write-side and read-side rankings for transparency
print("BLUF v8 joint pct_oracle (write/mixed only, ranked by geometric mean of act and w_combined):")
write_mixed = summary_pct.filter(pl.col("axis_kind").is_in(["write", "mixed", "ceiling"]))
print(tabulate(write_mixed.head(18).to_pandas(), headers="keys", tablefmt="github", floatfmt="+.4f"))

print("\nv8 read-side rows (pct_oracle_w means cross-space alignment, not 'explains delta'):")
read_only = summary_pct.filter(pl.col("axis_kind") == "read")
print(tabulate(read_only.to_pandas(), headers="keys", tablefmt="github", floatfmt="+.3f"))

# %% [markdown]
# ## Specificity: repeat activation score after removing clean residual PCs

# %%
clean_basis_by_layer = {c.name: c.basis_by_layer for c in candidate_list}["layer_clean_resid_pca"]
specific_null_cache: dict[tuple[int, int, int], tuple[float, float]] = {}


def specific_null_stats(layer: int, rank: int, ambient_rank: int) -> tuple[float, float]:
    key = (layer, rank, ambient_rank)
    if key in specific_null_cache:
        return specific_null_cache[key]
    clean = clean_basis_by_layer[layer]
    samples = hs_diff_B[layer] @ (torch.eye(d_model) - clean @ clean.T)
    total = samples.pow(2).sum(1) + 1e-12
    null = rank / ambient_rank
    gen = torch.Generator(device=samples.device).manual_seed(50_000 + 97 * layer + 13 * rank)
    values = []
    for _ in range(N_NULL):
        rb, _ = torch.linalg.qr(torch.randn(d_model, rank, generator=gen, device=samples.device, dtype=samples.dtype))
        rb = project_away(rb, clean)
        if rb.shape[1] != rank:
            raise ValueError(f"random residual rank collapsed: layer={layer}, rank={rank}, got={rb.shape[1]}")
        values.append(((samples @ rb).pow(2).sum(1) / total).mean().item() / null)
    arr = torch.tensor(values)
    stats = (float(arr.mean()), float(arr.std(unbiased=True)))
    specific_null_cache[key] = stats
    return stats


def specific_concentration_act(layer: int, basis: torch.Tensor) -> dict[str, float]:
    clean = clean_basis_by_layer[layer]
    residual_basis = project_away(basis, clean)
    rank = residual_basis.shape[1]
    if rank == 0:
        return {"specific_conc_act": 0.0, "specific_z_act": 0.0, "specific_energy_frac_act": 0.0, "specific_rank": 0}
    samples = hs_diff_B[layer] @ (torch.eye(d_model) - clean @ clean.T)
    total = samples.pow(2).sum(1) + 1e-12
    ambient_rank = d_model - clean.shape[1]
    energy_frac = ((samples @ residual_basis).pow(2).sum(1) / total).mean().item()
    conc = energy_frac / (rank / ambient_rank)
    null_mean, null_std = specific_null_stats(layer, rank, ambient_rank)
    return {
        "specific_conc_act": conc,
        "specific_z_act": (conc - null_mean) / (null_std + 1e-12),
        "specific_energy_frac_act": energy_frac,
        "specific_rank": rank,
    }


specific_rows = []
for layer in range(n_layers):
    for candidate in all_candidates:
        specific_rows.append({
            "layer": layer,
            "subspace": candidate.name,
            "family": candidate.family,
            "source": candidate.source,
            "kind": "ceiling" if candidate.family == "ceiling" else "A-hypothesis",
            **specific_concentration_act(layer, candidate.basis_by_layer[layer]),
        })

specific_per_layer = pl.DataFrame(specific_rows)
specific_per_layer_path = OUT_DIR / "v8_specific_per_layer.csv"
specific_per_layer.write_csv(specific_per_layer_path)
specific_summary = (
    specific_per_layer.filter(pl.col("layer").is_in(list(LORA_LAYERS)))
    .group_by(["subspace", "family", "source", "kind"])
    .agg(
        pl.col("specific_conc_act").mean().alias("mean_specific_conc_act"),
        pl.col("specific_z_act").mean().alias("mean_specific_z_act"),
        pl.col("specific_energy_frac_act").mean().alias("mean_specific_energy_frac_act"),
        pl.col("specific_rank").mean().alias("mean_specific_rank"),
    )
    .sort("mean_specific_conc_act", descending=True)
)
specific_summary_path = OUT_DIR / "v8_specific_summary.tsv"
specific_summary.write_csv(specific_summary_path, separator="\t")

print("BLUF v8 residualized activation specificity:")
print(tabulate(specific_summary.head(16).to_pandas(), headers="keys", tablefmt="github", floatfmt="+.3f"))

# %% [markdown]
# ## Figures and definitions

# %%
plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 240, "font.size": 9})
plot_df_all = summary_pct.filter(pl.col("kind") == "A-hypothesis").to_pandas()
ceiling_df = summary_pct.filter(pl.col("kind") == "ceiling").to_pandas()

# Figure 1: zoomed scatter on percent scale (0-100% to ideal).
# Most candidates cluster in the 0-15% corner so a zoomed view + percent axis
# reads more naturally than the full [0,1] square.
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
for ax, kind_filter, panel_title in [
    (axes[0], ("write", "mixed"), "write+mixed candidates (% to ideal)"),
    (axes[1], ("read",), "read-side (cross-space alignment)"),
]:
    panel_df = plot_df_all[plot_df_all["axis_kind"].isin(kind_filter)].head(20).copy()
    panel_df["x_pct"] = 100 * panel_df["mean_pct_oracle_act"]
    panel_df["y_pct"] = 100 * panel_df["mean_pct_oracle_w_combined"]
    for family, fam_df in panel_df.groupby("family"):
        ax.scatter(fam_df["x_pct"], fam_df["y_pct"], s=58, alpha=0.85, label=family)
    # Annotate only the top-6 by joint score to avoid label spaghetti.
    for row in panel_df.head(6).itertuples(index=False):
        ax.annotate(row.subspace, (row.x_pct, row.y_pct), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 18)
    ax.set_xlabel("% to ideal on activation axis")
    ax.set_title(panel_title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncols=2, loc="upper right")
axes[0].set_ylabel("% to ideal on weight axis (Frob-balanced combined)")
axes[1].set_ylabel("")

# Third panel: full-scale view with oracle so the ceiling gap is visible.
ax = axes[2]
all_pts = plot_df_all.copy()
all_pts["x_pct"] = 100 * all_pts["mean_pct_oracle_act"]
all_pts["y_pct"] = 100 * all_pts["mean_pct_oracle_w_combined"]
ax.scatter(all_pts["x_pct"], all_pts["y_pct"], s=24, color="steelblue", alpha=0.7, label="A-hypotheses")
if len(ceiling_df):
    cd = ceiling_df.copy()
    cd["x_pct"] = 100 * cd["mean_pct_oracle_act"]
    cd["y_pct"] = 100 * cd["mean_pct_oracle_w_combined"]
    ax.scatter(cd["x_pct"], cd["y_pct"], s=140, marker="*", color="black", label="oracle")
    for row in cd.itertuples(index=False):
        ax.annotate(row.subspace, (row.x_pct, row.y_pct), fontsize=7.5, xytext=(5, -2), textcoords="offset points")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("% to ideal on activation axis")
ax.set_ylabel("% to ideal on weight axis")
ax.set_title("full scale view (gap to oracle)")
ax.grid(alpha=0.25)
ax.legend(fontsize=7, loc="upper right")

fig.suptitle("v8: % to ideal = energy_frac(basis) / energy_frac(top-r_eff oracle), per axis. 100% = matches optimal rank-r_eff subspace.")
fig.tight_layout()
scatter_png = OUT_DIR / "v8_joint_act_weight_scatter.png"
scatter_pdf = OUT_DIR / "v8_joint_act_weight_scatter.pdf"
fig.savefig(scatter_png, bbox_inches="tight")
fig.savefig(scatter_pdf, bbox_inches="tight")
plt.close(fig)

# Figure 2: horizontal bar chart of joint % to ideal (write/mixed only).
# Easier to read than the scatter when everything compresses into a corner.
bar_df = (
    summary_pct.filter(pl.col("axis_kind").is_in(["write", "mixed", "ceiling"]))
    .sort("joint_pct_oracle", descending=True)
    .head(20)
    .to_pandas()
)
fig2, ax2 = plt.subplots(figsize=(9, 7))
y_pos = np.arange(len(bar_df))
ax2.barh(
    y_pos, 100 * bar_df["mean_pct_oracle_act"], height=0.42, label="% to ideal: activation",
    color="#5B8FF9", edgecolor="black", linewidth=0.4,
)
ax2.barh(
    y_pos - 0.42, 100 * bar_df["mean_pct_oracle_w_combined"], height=0.42, label="% to ideal: weight (combined)",
    color="#F6BD16", edgecolor="black", linewidth=0.4,
)
ax2.set_yticks(y_pos - 0.21)
ax2.set_yticklabels(bar_df["subspace"], fontsize=8)
ax2.invert_yaxis()
ax2.axvline(100, color="black", linestyle="--", linewidth=0.8, label="ideal (100%)")
ax2.set_xlim(0, 105)
ax2.set_xlabel("% to ideal at candidate's effective rank")
ax2.set_title("v8 joint % to ideal (top-20 write+mixed candidates + oracle)")
ax2.legend(loc="lower right", fontsize=8)
ax2.grid(axis="x", alpha=0.25)
fig2.tight_layout()
bar_png = OUT_DIR / "v8_pct_ideal_bars.png"
bar_pdf = OUT_DIR / "v8_pct_ideal_bars.pdf"
fig2.savefig(bar_png, bbox_inches="tight")
fig2.savefig(bar_pdf, bbox_inches="tight")
plt.close(fig2)

definitions_path = OUT_DIR / "v8_definitions.md"
plan_merge_path = OUT_DIR / "v8_plan_merge.md"
definitions = [
    "# v8 hypothesis definitions",
    "",
    "All A-side hypotheses are built without the trained LoRA. The LoRA diff is used only for B-side scoring.",
    "",
    "v8 changes vs v7: rank-honest pct_oracle is the primary metric. For each candidate at each layer, oracle = top-r_eff (effective rank of basis) singular subspace of the target tensor; score = energy_frac(basis) / energy_frac(oracle) in [0, 1]. Eliminates the v7 forced PCS=8 budget mismatch (chars_clusters with r_eff=7 was being graded against rank-8 oracle).",
    "",
    "| name | family | axis_kind | source | definition |",
    "|---|---|---|---|---|",
]
for candidate in all_candidates:
    definitions.append(f"| `{candidate.name}` | {candidate.family} | {axis_kind_for(candidate.family)} | {candidate.source} | {candidate.definition} |")
definitions_path.write_text("\n".join(definitions) + "\n")

plan_merge_path.write_text("""# v8 changes vs v7

v7 reported `pct_w_oracle_combined` as the candidate's R_w divided by the oracle's R_w -- a *post-hoc* ratio of two concentration ratios. For most candidates this gave 5.6-7.9% with a flat range, hard to interpret.

v8 changes:

1. **pct_oracle is the primary metric.** Computed *per row* (not post-hoc): oracle = top-r_eff (effective rank of basis) singular subspace of the target tensor; score = energy_frac(basis) / energy_frac(oracle) in [0, 1]. Rank-honest: chars_clusters (r_eff=7) is graded against rank-7 oracle, not rank-8.
2. **Joint score** = geometric mean of pct_oracle_act and pct_oracle_w_combined, both in [0, 1].
3. **Effective rank columns** (`r_eff_w`, `r_eff_act`) added so silent rank collapse is visible per row.
4. **Activation oracle** = PCA of L2-normalized hs_diff_B (the optimal basis for E[per-example normalized energy]), not raw PCA. Matches the existing `energy_frac_act` formula.
5. v7 z-scores and Frobenius-balanced concentration ratios kept as supplementary columns for diagnostic continuity.

**Limitation kept honest in conclusion**: pct_oracle is still a *subspace* metric. Any primitive whose mechanism is nonlinear (CHaRS-style per-cluster translations, gated MLP, token-conditional behavior) is structurally penalized -- we throw away the nonlinearity and keep just the linear span.

Not changed from v7:
- Single LoRA seed (multi-seed deferred).
- Per-tensor R_w (oproj/downproj/combined) carried over from v7.
- axis_kind tagging (write/read/mixed/ceiling) carried over.
""")

winner = summary_pct.filter((pl.col("kind") == "A-hypothesis") & (pl.col("axis_kind").is_in(["write", "mixed"]))).row(0, named=True)
act_winners = summary_pct.filter(pl.col("kind") == "A-hypothesis").sort("mean_pct_oracle_act", descending=True).head(5)
w_winners = summary_pct.filter((pl.col("kind") == "A-hypothesis") & (pl.col("axis_kind").is_in(["write", "mixed"]))).sort("mean_pct_oracle_w_combined", descending=True).head(5)
top_act = set(act_winners["subspace"].to_list())
top_w = set(w_winners["subspace"].to_list())
both_top5 = sorted(top_act & top_w)
conclusion_path = OUT_DIR / "v8_conclusion.md"
conclusion_path.write_text(f"""# v8 hypothesis sweep conclusion

## BLUF

Best joint A-side primitive (write/mixed only) by geometric mean of pct_oracle_act
and pct_oracle_w_combined: `{winner['subspace']}`.
- pct_oracle_act = {winner['mean_pct_oracle_act']:.3f} ({winner['mean_pct_oracle_act']*100:.1f}% of optimal rank-{int(round(winner['mean_r_eff_act']))} PCA on hs_diff_B)
- pct_oracle_w_combined = {winner['mean_pct_oracle_w_combined']:.3f} ({winner['mean_pct_oracle_w_combined']*100:.1f}% of optimal rank-{int(round(winner['mean_r_eff_w']))} SVD of LoRA delta)
- joint = {winner['joint_pct_oracle']:.3f}

Per-tensor pct_oracle for the winner: oproj={winner['mean_pct_oracle_w_oproj']:.3f}, downproj={winner['mean_pct_oracle_w_downproj']:.3f}.

Top-5 overlap (by pct_oracle_act and pct_oracle_w_combined, write/mixed only): {both_top5}.

Sanity check (oracle rows):
- `w_oracle`.pct_oracle_w_combined = {weight_ceiling_pct:.3f} (SHOULD ~ 1.0)
- `act_oracle`.pct_oracle_act = {act_ceiling_pct:.3f} (SHOULD ~ 1.0)

## Reading pct_oracle

A score of 0.10 means: this candidate captures 10% of the energy that the
*best possible* rank-r_eff subspace captures. So 0.10 is bad in absolute
terms (the candidate is far from the optimal subspace at its own rank), and
0.10 with r_eff=8 is *just as bad* as 0.10 with r_eff=4 -- the rank-honest
oracle handles the budget difference automatically.

This is a tighter test than v7's z-score-vs-random-orthonormal: it asks
"are you the optimal subspace?" instead of "are you better than random?".
Most reasonably-aligned bases beat random easily; few are anywhere near
optimal.

## v8 changes vs v7

1. **pct_oracle is the primary metric**, computed per row from energy_frac /
   oracle_at(r_eff). v7's `pct_w_oracle_combined` was a post-hoc ratio of
   concentration ratios (R_w / R_w_oracle), which double-counted the rank
   normalization.
2. **Effective rank** (`r_eff_w`, `r_eff_act`) reported per row so silent
   collapse is visible (chars_clusters: r_eff=7 not 8).
3. **Activation oracle** = PCA of L2-normalized hs_diff_B, matching the
   per-example normalization in `energy_frac_act`.
4. v7 z-scores and Frobenius-balanced concentration ratios kept as
   supplementary columns.

## Caveats

- **Single LoRA seed.** Rankings are anecdote-grade until v8b multi-seed runs.
- **Subspace metric only.** pct_oracle measures linear span alignment. Any
  primitive whose mechanism is nonlinear (CHaRS-style per-cluster
  translations, gated MLP, token-conditional behavior) is structurally
  penalized -- we throw away the nonlinearity and keep just the centroid /
  span / averaged direction. Don't read low pct_oracle_w as "this method
  doesn't work for steering" -- read it as "this primitive's *linear span*
  doesn't capture LoRA's delta".
- **R_w only scores residual-output LoRA tensors** (`o_proj`, `down_proj`)
  because the basis lives in residual-output space (d_model rows). Other
  LoRA tensors (q/k/v projections etc.) are not scored.
- **Known construction nits** (inline comments, not fixed): `chars_clusters`
  rank-collapses to 7; `qk_circuit` mixes all heads; `intersect_basis` uses
  Bjorck-Golub bisector not strict intersection.

## Artifacts

- Per-layer raw scores: `{per_layer_path}`
- Summary: `{summary_path}`
- Summary (percent-scale view): `{summary_pct_path}`
- Residualized activation per-layer scores: `{specific_per_layer_path}`
- Residualized activation summary: `{specific_summary_path}`
- Joint scatter (zoomed % view + full-scale gap to oracle): `{scatter_png}`, `{scatter_pdf}`
- Bar chart of joint % to ideal: `{bar_png}`, `{bar_pdf}`
- Definitions: `{definitions_path}`
- v8-vs-v7 changes: `{plan_merge_path}`
""")

print("wrote:")
for path in [
    per_layer_path,
    summary_path,
    summary_pct_path,
    specific_per_layer_path,
    specific_summary_path,
    definitions_path,
    plan_merge_path,
    conclusion_path,
    scatter_png,
    scatter_pdf,
]:
    print(f"  {path} ({path.stat().st_size} bytes)")

print(
    "SHOULD: oracle rows have pct_oracle ~ 1.0 by construction; useful primitives have pct_oracle_act and pct_oracle_w_combined both well above 0 (anything > 0.5 is a meaningful linear approximator). "
    "ELSE: check basis orientation, LoRA diff tensor selection, or that the basis is properly orthonormal."
)
