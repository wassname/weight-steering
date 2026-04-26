# %% [markdown]
# # v7 hypothesis sweep: per-tensor R_w, true weight ceiling, axis-kind tagging
#
# v6 found that R_w was Frobenius-dominated by mlp.down_proj (3M params)
# vs self_attn.o_proj (1M), used PCA(hs_diff_B_fit) as the "weight ceiling"
# (which is not a ceiling on weights), and silently scored read-side bases on
# the write-side LoRA delta as if it meant "explains delta".
#
# v7 fixes:
# 1. R_w split into R_w_oproj, R_w_downproj, plus a Frobenius-balanced combined.
# 2. dw_left_basis is the true weight ceiling (R_w / R_w(dw_left_basis) ~ 1.0
#    for the oracle row by construction).
# 3. axis_kind tag (write/read/mixed/ceiling) on every hypothesis; read-side
#    rows are reported separately and excluded from the joint W-axis ranking.
# 4. (multi-seed loop deferred to v7b once single-seed validation passes.)

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
    "logs/hypothesis_sweep_v7.verbose.log",
    level="DEBUG",
    format="{time} | {level} | {name}:{function}:{line} - {message}",
)
torch.set_grad_enabled(False)

MODEL_ID = "Qwen/Qwen3-0.6B"
W_PATH = Path(os.environ.get("W_PATH", "out/sycophancy/lora/w.pt"))
OUT_DIR = Path("out/sycophancy/lora/v7")
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
    "TaskDiff_lora_ceiling",
    "ceiling",
    [pca(hs_diff_B_fit[layer], PCS) for layer in range(n_layers)],
    "B-side",
    "PCA of LoRA FIT-half label; not an A-side hypothesis",
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
        return {"conc_act": 0.0, "z_act": 0.0, "energy_frac_act": 0.0}
    total = samples.pow(2).sum(1) + 1e-12
    energy_frac = ((samples @ basis).pow(2).sum(1) / total).mean().item()
    conc = energy_frac / (rank / samples.shape[1])
    null_mean, null_std = act_null_stats(layer, rank)
    return {"conc_act": conc, "z_act": (conc - null_mean) / (null_std + 1e-12), "energy_frac_act": energy_frac}


def concentration_w(layer: int, basis: torch.Tensor) -> dict[str, float]:
    """Per-tensor weight concentration + Frobenius-balanced combined.

    v6 returned a single conc_w that silently weighted by tensor size
    (down_proj has ~3x the params of o_proj). v7 reports each tensor
    separately so write-side hypotheses can be ranked by either, and a
    'combined' score that normalizes each tensor to unit Frobenius first
    (size-balanced).
    """
    rank = basis.shape[1]
    tensors = lora_weight_tensors(layer)
    out: dict[str, float] = {}
    if rank == 0 or not tensors:
        for name in ("oproj", "downproj", "combined"):
            out[f"conc_w_{name}"] = float("nan")
            out[f"z_w_{name}"] = float("nan")
            out[f"energy_frac_w_{name}"] = float("nan")
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
            continue
        total = M.pow(2).sum() + 1e-12
        energy_frac = ((basis.T @ M).pow(2).sum() / total).item()
        conc = energy_frac / (rank / M.shape[0])
        null_mean, null_std = w_null_stats(layer, rank, key)
        out[f"conc_w_{short}"] = conc
        out[f"z_w_{short}"] = (conc - null_mean) / (null_std + 1e-12)
        out[f"energy_frac_w_{short}"] = energy_frac
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
    else:
        out["conc_w_combined"] = float("nan")
        out["z_w_combined"] = float("nan")
        out["energy_frac_w_combined"] = float("nan")
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


# Build the true weight ceiling: top-PCS left singular vectors of the LoRA
# delta itself, per layer. This is the natural R_w oracle: scoring it gives
# R_w / R_w_ceiling ~ 1.0 for any properly-implemented per-tensor split.
weight_ceiling = Candidate(
    "dW_left_basis_ceiling",
    "ceiling",
    [dw_left_basis(layer) for layer in range(n_layers)],
    "B-side",
    "Top-PCS left singular vectors of the LoRA residual-output delta itself; defines R_w = 1.0 by construction",
)


all_candidates = [*candidate_list, ceiling, weight_ceiling]
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
per_layer_path = OUT_DIR / "v7_per_layer.csv"
per_layer.write_csv(per_layer_path)

active = per_layer.filter(pl.col("layer").is_in(list(LORA_LAYERS)))
summary = (
    active.group_by(["subspace", "family", "axis_kind", "source", "kind"])
    .agg(
        pl.col("conc_act").mean().alias("mean_conc_act"),
        pl.col("z_act").mean().alias("mean_z_act"),
        pl.col("energy_frac_act").mean().alias("mean_energy_frac_act"),
        pl.col("conc_w_oproj").mean().alias("mean_conc_w_oproj"),
        pl.col("conc_w_downproj").mean().alias("mean_conc_w_downproj"),
        pl.col("conc_w_combined").mean().alias("mean_conc_w_combined"),
        pl.col("z_w_oproj").mean().alias("mean_z_w_oproj"),
        pl.col("z_w_downproj").mean().alias("mean_z_w_downproj"),
        pl.col("z_w_combined").mean().alias("mean_z_w_combined"),
        pl.col("cos_with_dW").mean().alias("mean_cos_dW"),
        pl.col("rank").mean().alias("mean_rank"),
    )
    .with_columns(
        # Joint score uses the size-balanced combined R_w to be fair across hypotheses
        joint_score=((pl.col("mean_conc_act").log() + pl.col("mean_conc_w_combined").log()) / 2).exp(),
        act_w_gap_log2=(pl.col("mean_conc_act").log(2) - pl.col("mean_conc_w_combined").log(2)),
    )
    .sort("joint_score", descending=True)
)

summary_path = OUT_DIR / "v7_summary.tsv"
summary.write_csv(summary_path, separator="\t")

ceiling_act = float(summary.filter(pl.col("subspace") == "TaskDiff_lora_ceiling")["mean_conc_act"][0])
# True weight ceiling: dW_left_basis_ceiling. Reports as ~1.0 by construction
# (the basis IS the top singular subspace of the weight diff).
weight_ceiling_combined = float(
    summary.filter(pl.col("subspace") == "dW_left_basis_ceiling")["mean_conc_w_combined"][0]
)
weight_ceiling_oproj = float(
    summary.filter(pl.col("subspace") == "dW_left_basis_ceiling")["mean_conc_w_oproj"][0]
)
weight_ceiling_downproj = float(
    summary.filter(pl.col("subspace") == "dW_left_basis_ceiling")["mean_conc_w_downproj"][0]
)
logger.info(
    f"weight ceiling (dW_left_basis): combined={weight_ceiling_combined:.3f} "
    f"oproj={weight_ceiling_oproj:.3f} downproj={weight_ceiling_downproj:.3f} "
    "SHOULD: all > 1.0 (basis IS top singular subspace, so concentrates >> null); "
    "oproj vs downproj differ because top-PCS captures different fractions of each "
    "tensor's Frobenius energy (square-ish o_proj concentrates better than wide down_proj). "
    "ELSE per-tensor split or null normalization is wrong."
)
summary_pct = summary.with_columns(
    pct_act_ceiling=100 * pl.col("mean_conc_act") / ceiling_act,
    pct_w_oracle_combined=100 * pl.col("mean_conc_w_combined") / weight_ceiling_combined,
    pct_w_oracle_oproj=100 * pl.col("mean_conc_w_oproj") / weight_ceiling_oproj,
    pct_w_oracle_downproj=100 * pl.col("mean_conc_w_downproj") / weight_ceiling_downproj,
)
summary_pct_path = OUT_DIR / "v7_summary_pct.tsv"
summary_pct.write_csv(summary_pct_path, separator="\t")

# Separate write-side and read-side rankings for transparency
print("BLUF v7 joint act+weight (write/mixed only, ranked by joint_score):")
write_mixed = summary_pct.filter(pl.col("axis_kind").is_in(["write", "mixed", "ceiling"]))
print(tabulate(write_mixed.head(18).to_pandas(), headers="keys", tablefmt="github", floatfmt="+.3f"))

print("\nv7 read-side rows (R_w means cross-space alignment, not 'explains delta'):")
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
specific_per_layer_path = OUT_DIR / "v7_specific_per_layer.csv"
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
specific_summary_path = OUT_DIR / "v7_specific_summary.tsv"
specific_summary.write_csv(specific_summary_path, separator="\t")

print("BLUF v7 residualized activation specificity:")
print(tabulate(specific_summary.head(16).to_pandas(), headers="keys", tablefmt="github", floatfmt="+.3f"))

# %% [markdown]
# ## Figures and definitions

# %%
plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 240, "font.size": 9})
plot_df_all = summary_pct.filter(pl.col("kind") == "A-hypothesis").to_pandas()
# Two-panel scatter: write/mixed (joint ranking) and read-side (cross-space alignment)
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
for ax, kind_filter, panel_title in [
    (axes[0], ("write", "mixed"), "write+mixed (R_w = explains delta)"),
    (axes[1], ("read",), "read-side (R_w = cross-space alignment)"),
]:
    panel_df = plot_df_all[plot_df_all["axis_kind"].isin(kind_filter)].head(20)
    for family, fam_df in panel_df.groupby("family"):
        ax.scatter(fam_df["mean_conc_act"], fam_df["mean_conc_w_combined"], s=52, alpha=0.82, label=family)
    for row in panel_df.head(10).itertuples(index=False):
        ax.annotate(row.subspace, (row.mean_conc_act, row.mean_conc_w_combined), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("activation recovery R_act")
    ax.set_title(panel_title)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, ncols=2)
axes[0].set_ylabel("weight recovery R_w (Frobenius-balanced combined)")
ceiling_df = summary_pct.filter(pl.col("kind") == "ceiling").to_pandas()
for ax in axes:
    if len(ceiling_df):
        ax.scatter(ceiling_df["mean_conc_act"], ceiling_df["mean_conc_w_combined"], s=85, marker="*", color="black", label="ceiling")
fig.suptitle("v7: read-side R_w is cross-space alignment, not 'explains delta'")
fig.tight_layout()
scatter_png = OUT_DIR / "v7_joint_act_weight_scatter.png"
scatter_pdf = OUT_DIR / "v7_joint_act_weight_scatter.pdf"
fig.savefig(scatter_png, bbox_inches="tight")
fig.savefig(scatter_pdf, bbox_inches="tight")
plt.close(fig)

definitions_path = OUT_DIR / "v7_definitions.md"
plan_merge_path = OUT_DIR / "v7_plan_merge.md"
definitions = [
    "# v7 hypothesis definitions",
    "",
    "All A-side hypotheses are built without the trained LoRA. The LoRA diff is used only for B-side scoring.",
    "",
    "v7 changes vs v6: per-tensor R_w (oproj/downproj/combined), dW_left_basis_ceiling as the true weight ceiling, axis_kind tag (write/read/mixed/ceiling) so read-side cross-space scores aren't conflated with 'explains delta'.",
    "",
    "| name | family | axis_kind | source | definition |",
    "|---|---|---|---|---|",
]
for candidate in all_candidates:
    definitions.append(f"| `{candidate.name}` | {candidate.family} | {axis_kind_for(candidate.family)} | {candidate.source} | {candidate.definition} |")
definitions_path.write_text("\n".join(definitions) + "\n")

plan_merge_path.write_text("""# v7 changes vs v6

Addresses three real concerns from `docs/review/v6_hypothesis_review.md`:

1. **Per-tensor R_w.** `lora_weight_tensors(layer)` returns a dict {o_proj, down_proj}; `concentration_w` reports `R_w_oproj`, `R_w_downproj`, and a Frobenius-balanced `R_w_combined`. Joint score uses combined; per-tensor are reported for inspection. Eliminates the silent down_proj domination (down_proj has ~3x the params of o_proj in this model).

2. **True weight ceiling.** Added `dW_left_basis_ceiling` candidate: top-PCS left singular vectors of the LoRA delta itself. By construction `R_w(combined) ~ d_model/PCS = 128` for that row, so `pct_w_oracle_combined` is on a true 0-100 scale (oracle = 100). The v6 column `pct_w_taskdiff_basis` was relative to `PCA(hs_diff_B_fit)` -- an activation basis, not a weight oracle.

3. **axis_kind tag.** Each candidate is tagged write / read / mixed / ceiling. Read-side bases (mlp_up_read, mlp_gate_read, attn_qkv_read, kv_super, input_super, lm_head_read, logits_null, input_super_not_lm_read) are reported in a separate sub-table and a separate scatter panel. High R_w on a read-side basis means "this read direction happens to coincide with LoRA write directions", not "this primitive captures the LoRA write geometry".

Deferred to v7b (multi-seed): currently single-LoRA-seed; rankings are anecdote-grade until run on >=3 LoRA seeds with stability filtering.

Not fixed (left as known-limitations comments only):
- `chars_clusters` PCA collapses to rank 7 because centroids - mean has rank k_clusters - 1 = 7 < PCS=8.
- `qk_circuit` mixes all heads in one d_model x d_model matrix.
- `intersect_basis` uses Bjorck-Golub bisector, not strict subspace intersection (returns directions even at low principal-angle alignment).
""")

winner = summary_pct.filter((pl.col("kind") == "A-hypothesis") & (pl.col("axis_kind").is_in(["write", "mixed"]))).row(0, named=True)
act_winners = summary_pct.filter(pl.col("kind") == "A-hypothesis").sort("mean_conc_act", descending=True).head(5)
w_winners = summary_pct.filter((pl.col("kind") == "A-hypothesis") & (pl.col("axis_kind").is_in(["write", "mixed"]))).sort("mean_conc_w_combined", descending=True).head(5)
top_act = set(act_winners["subspace"].to_list())
top_w = set(w_winners["subspace"].to_list())
both_top5 = sorted(top_act & top_w)
conclusion_path = OUT_DIR / "v7_conclusion.md"
conclusion_path.write_text(f"""# v7 hypothesis sweep conclusion

## BLUF

Best joint A-side primitive (write/mixed only) by geometric mean of activation
and Frobenius-balanced weight recovery: `{winner['subspace']}`. R_act={winner['mean_conc_act']:.2f},
R_w_combined={winner['mean_conc_w_combined']:.2f} (oracle={weight_ceiling_combined:.2f}, so
{winner['pct_w_oracle_combined']:.1f}% of weight ceiling), joint={winner['joint_score']:.2f}.

Per-tensor R_w for the winner: oproj={winner['mean_conc_w_oproj']:.2f} ({winner['pct_w_oracle_oproj']:.1f}% of oracle), downproj={winner['mean_conc_w_downproj']:.2f} ({winner['pct_w_oracle_downproj']:.1f}% of oracle).

Top-5 overlap between activation winners and weight winners (write/mixed only): {both_top5}.

## v7 changes vs v6

1. R_w split per LoRA tensor (o_proj vs down_proj) plus a Frobenius-balanced combined; v6's single conc_w was silently dominated by down_proj (~3x the params).
2. dW_left_basis_ceiling row gives `R_w_combined~={weight_ceiling_combined:.2f}` (oracle); `pct_w_oracle_combined` is now percent-of-oracle, not percent-of-PCA(hs_diff_B_fit).
3. Read-side hypotheses (input projections) are tagged axis_kind='read' and reported in a separate sub-table. A high R_w there means cross-space alignment between the read subspace and the write-side LoRA delta -- not 'this primitive explains the delta'.

## Caveats

- Single LoRA seed; rankings are anecdote-grade until v7b multi-seed runs.
- R_w only scores residual-output LoRA tensors (`o_proj`, `down_proj`) because the basis lives in residual-output space (d_model rows).
- `chars_clusters` silently rank-collapses to 7 (centroids - mean has rank k-1); `qk_circuit` mixes all heads; `intersect_basis` is the Bjorck-Golub bisector not strict intersection. Inline comments only; not fixed in v7.

## Artifacts

- Per-layer raw scores: `{per_layer_path}`
- Summary: `{summary_path}`
- Summary with oracle-relative percentages: `{summary_pct_path}`
- Residualized activation per-layer scores: `{specific_per_layer_path}`
- Residualized activation summary: `{specific_summary_path}`
- Joint scatter (write+mixed | read sub-panel): `{scatter_png}`, `{scatter_pdf}`
- Definitions: `{definitions_path}`
- v7-vs-v6 changes: `{plan_merge_path}`
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
    "SHOULD: useful subspaces have R_act>1 and R_w>1; generic activation artifacts show high R_act but weak R_w. "
    "ELSE: check basis orientation and LoRA diff tensor selection."
)
