# %% [markdown]
# # v6 hypothesis sweep: activation score + weight score
#
# v5 asked which LoRA-free basis recovers the held-out LoRA activation label.
# v6 adds the cheap missing correctness check: does the same basis also recover the
# residual-output LoRA weight diff?
#
# A-side bases still use only pretrained weights and base-model activations. B-side
# labels are the trained LoRA activation difference and LoRA weight diff.

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
    "logs/hypothesis_sweep_v6.verbose.log",
    level="DEBUG",
    format="{time} | {level} | {name}:{function}:{line} - {message}",
)
torch.set_grad_enabled(False)

MODEL_ID = "Qwen/Qwen3-0.6B"
W_PATH = Path(os.environ.get("W_PATH", "out/sycophancy/lora/w.pt"))
OUT_DIR = Path("out/sycophancy/lora/v6")
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
def lora_weight_matrix(layer: int) -> torch.Tensor:
    cols = []
    for proj in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
        key = f"model.layers.{layer}.{proj}"
        if key in w:
            W = w[key].float().cpu()
            if W.shape[0] == d_model:
                cols.append(W)
    if not cols:
        return torch.zeros(d_model, 0)
    return torch.cat(cols, dim=1)


act_null_cache: dict[tuple[int, int], tuple[float, float]] = {}
w_null_cache: dict[tuple[int, int], tuple[float, float]] = {}


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


def w_null_stats(layer: int, rank: int) -> tuple[float, float]:
    key = (layer, rank)
    if key in w_null_cache:
        return w_null_cache[key]
    M = lora_weight_matrix(layer)
    if M.shape[1] == 0:
        stats = (float("nan"), float("nan"))
        w_null_cache[key] = stats
        return stats
    d = M.shape[0]
    total = M.pow(2).sum() + 1e-12
    null = rank / d
    gen = torch.Generator(device=M.device).manual_seed(20_000 + 97 * layer + rank)
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
    M = lora_weight_matrix(layer)
    rank = basis.shape[1]
    if rank == 0 or M.shape[1] == 0:
        return {"conc_w": float("nan"), "z_w": float("nan"), "energy_frac_w": float("nan")}
    total = M.pow(2).sum() + 1e-12
    energy_frac = ((basis.T @ M).pow(2).sum() / total).item()
    conc = energy_frac / (rank / M.shape[0])
    null_mean, null_std = w_null_stats(layer, rank)
    return {"conc_w": conc, "z_w": (conc - null_mean) / (null_std + 1e-12), "energy_frac_w": energy_frac}


def dw_left_basis(layer: int) -> torch.Tensor:
    return left_svd_basis(lora_weight_matrix(layer))


all_candidates = [*candidate_list, ceiling]
dw_bases = [dw_left_basis(layer) for layer in range(n_layers)]
rows = []
for layer in range(n_layers):
    for candidate in all_candidates:
        basis = candidate.basis_by_layer[layer]
        rows.append({
            "layer": layer,
            "subspace": candidate.name,
            "family": candidate.family,
            "source": candidate.source,
            "kind": "ceiling" if candidate.family == "ceiling" else "A-hypothesis",
            "rank": basis.shape[1],
            **concentration_act(layer, basis),
            **concentration_w(layer, basis),
            "cos_with_dW": principal_cos(basis, dw_bases[layer]),
        })

per_layer = pl.DataFrame(rows)
per_layer_path = OUT_DIR / "v6_per_layer.csv"
per_layer.write_csv(per_layer_path)

active = per_layer.filter(pl.col("layer").is_in(list(LORA_LAYERS)))
summary = (
    active.group_by(["subspace", "family", "source", "kind"])
    .agg(
        pl.col("conc_act").mean().alias("mean_conc_act"),
        pl.col("z_act").mean().alias("mean_z_act"),
        pl.col("energy_frac_act").mean().alias("mean_energy_frac_act"),
        pl.col("conc_w").mean().alias("mean_conc_w"),
        pl.col("z_w").mean().alias("mean_z_w"),
        pl.col("energy_frac_w").mean().alias("mean_energy_frac_w"),
        pl.col("cos_with_dW").mean().alias("mean_cos_dW"),
        pl.col("rank").mean().alias("mean_rank"),
    )
    .with_columns(
        joint_score=((pl.col("mean_conc_act").log() + pl.col("mean_conc_w").log()) / 2).exp(),
        act_w_gap_log2=(pl.col("mean_conc_act").log(2) - pl.col("mean_conc_w").log(2)),
    )
    .sort("joint_score", descending=True)
)

summary_path = OUT_DIR / "v6_summary.tsv"
summary.write_csv(summary_path, separator="\t")

ceiling_act = float(summary.filter(pl.col("kind") == "ceiling")["mean_conc_act"][0])
taskdiff_basis_w = float(summary.filter(pl.col("kind") == "ceiling")["mean_conc_w"][0])
summary_pct = summary.with_columns(
    pct_act_ceiling=100 * pl.col("mean_conc_act") / ceiling_act,
    pct_w_taskdiff_basis=100 * pl.col("mean_conc_w") / taskdiff_basis_w,
)
summary_pct_path = OUT_DIR / "v6_summary_pct.tsv"
summary_pct.write_csv(summary_pct_path, separator="\t")

print("BLUF v6 joint activation+weight score:")
print(tabulate(summary_pct.head(18).to_pandas(), headers="keys", tablefmt="github", floatfmt="+.3f"))

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
specific_per_layer_path = OUT_DIR / "v6_specific_per_layer.csv"
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
specific_summary_path = OUT_DIR / "v6_specific_summary.tsv"
specific_summary.write_csv(specific_summary_path, separator="\t")

print("BLUF v6 residualized activation specificity:")
print(tabulate(specific_summary.head(16).to_pandas(), headers="keys", tablefmt="github", floatfmt="+.3f"))

# %% [markdown]
# ## Figures and definitions

# %%
plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 240, "font.size": 9})
plot_df = summary_pct.filter(pl.col("kind") == "A-hypothesis").head(18).to_pandas()
ceiling_df = summary_pct.filter(pl.col("kind") == "ceiling").to_pandas()
fig, ax = plt.subplots(figsize=(8.5, 6.2))
for family, fam_df in plot_df.groupby("family"):
    ax.scatter(fam_df["mean_conc_act"], fam_df["mean_conc_w"], s=52, alpha=0.82, label=family)
for row in plot_df.head(10).itertuples(index=False):
    ax.annotate(row.subspace, (row.mean_conc_act, row.mean_conc_w), fontsize=7, xytext=(3, 3), textcoords="offset points")
if len(ceiling_df):
    ax.scatter(ceiling_df["mean_conc_act"], ceiling_df["mean_conc_w"], s=85, marker="*", color="black", label="ceiling")
ax.axvline(1.0, color="black", linestyle="--", linewidth=0.9)
ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("activation recovery R_act")
ax.set_ylabel("weight recovery R_w")
ax.set_title("v6: a useful primitive should beat random on both axes")
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=7, ncols=2)
fig.tight_layout()
scatter_png = OUT_DIR / "v6_joint_act_weight_scatter.png"
scatter_pdf = OUT_DIR / "v6_joint_act_weight_scatter.pdf"
fig.savefig(scatter_png, bbox_inches="tight")
fig.savefig(scatter_pdf, bbox_inches="tight")
plt.close(fig)

definitions_path = OUT_DIR / "v6_definitions.md"
plan_merge_path = OUT_DIR / "v6_plan_merge.md"
definitions = [
    "# v6 hypothesis definitions",
    "",
    "All A-side hypotheses are built without the trained LoRA. The LoRA diff is used only for B-side scoring.",
    "",
    "| name | family | source | definition |",
    "|---|---|---|---|",
]
for candidate in all_candidates:
    definitions.append(f"| `{candidate.name}` | {candidate.family} | {candidate.source} | {candidate.definition} |")
definitions_path.write_text("\n".join(definitions) + "\n")

plan_merge_path.write_text("""# v6 external-plan merge

Accepted into v6:

- Two-axis scoring: activation recovery `R_act` plus residual-output LoRA weight recovery `R_w`.
- W-only primitives: `qk_circuit`, `input_super`, `kv_super`, `gate_kernel`, `attention_sink`, `causally_isolated`, `input_super_not_lm_read`.
- Activation primitives: `added_features`, `gate_active_written`, `chars_clusters`, plus attention-selected TaskDiff variants `attn_min_taskdiff`, `attn_max_taskdiff`, `attn_diff_taskdiff`, `attn_min_x_diffnorm_taskdiff`.
- Compound primitive: `qk_x_chars_clusters`.
- Output isolation: all v6 artifacts write under `out/sycophancy/lora/v6/`.

Deferred deliberately:

- `polar_skew`: most relevant Qwen matrices here are rectangular due MLP/GQA shapes; forcing a square surrogate would add interpretation debt.

Correction made while merging:

- `causally_isolated` is now a write-constrained basis: residual write directions projected away from input-read, KV, and lm_head read bases. The crash-state version accidentally returned an arbitrary complement of the forbidden basis, not the isolated part of write.
""")

winner = summary_pct.filter(pl.col("kind") == "A-hypothesis").row(0, named=True)
act_winners = summary_pct.filter(pl.col("kind") == "A-hypothesis").sort("mean_conc_act", descending=True).head(5)
w_winners = summary_pct.filter(pl.col("kind") == "A-hypothesis").sort("mean_conc_w", descending=True).head(5)
top_act = set(act_winners["subspace"].to_list())
top_w = set(w_winners["subspace"].to_list())
both_top5 = sorted(top_act & top_w)
conclusion_path = OUT_DIR / "v6_conclusion.md"
conclusion_path.write_text(f"""# v6 hypothesis sweep conclusion

## BLUF

Best joint A-side primitive by geometric mean of activation and weight recovery: `{winner['subspace']}` with activation R={winner['mean_conc_act']:.2f}, weight R={winner['mean_conc_w']:.2f}, joint={winner['joint_score']:.2f}.

Top-5 overlap between activation winners and weight winners: {both_top5}.

The weight axis is weak: most activation winners have `R_w` near the random null, and even the LoRA-fitted activation basis has `R_w={taskdiff_basis_w:.2f}`. So v6 mostly says which hypotheses retain activation evidence after stronger controls; only top weight-overlap rows are plausible two-axis leads.

## Caveats

- `R_w` only scores residual-output LoRA tensors (`o_proj`, `down_proj`) because the basis lives in residual-output space.
- The LoRA-fitted activation ceiling is not a weight ceiling. Columns named `pct_w_taskdiff_basis` are relative to that basis, not to an oracle upper bound.
- If no candidate is strong on both axes, that is a negative result for these hand-written structural primitives, not evidence that no structure exists.

## Artifacts

- Per-layer raw scores: `{per_layer_path}`
- Summary: `{summary_path}`
- Summary with reference percentages: `{summary_pct_path}`
- Residualized activation per-layer scores: `{specific_per_layer_path}`
- Residualized activation summary: `{specific_summary_path}`
- Joint scatter: `{scatter_png}`, `{scatter_pdf}`
- Definitions: `{definitions_path}`
- External-plan merge notes: `{plan_merge_path}`
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
