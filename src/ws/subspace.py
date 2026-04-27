"""Subspace alignment for the diff vector w.

We test whether the steering direction `w_layer = θ+_layer - θ-_layer` lies
preferentially in interpretable subspaces of the pretrained model, rather
than in random directions.

Two weight-only tests (no activations needed):

1. SVD-of-W (per layer)
   For W = U @ diag(S) @ V^T, project w into the SVD basis:
       proj = U^T @ w @ V       # shape (r, r)
   Energy in the top-k×k block / total energy.
   Null (uniform random matrix): (k/r)² because a random direction has
   equal energy in any orthonormal pair of k-dim subspaces of size k×k.
   ratio > 1 ⇒ w concentrates in W's principal components (PiSSA-aligned).

2. Weak-readout (for params whose output is the residual stream: o_proj, down_proj)
   SVD of lm_head: lm_head = U_lm @ diag(S_lm) @ V_lm^T, V_lm rows live in residual space.
   Bottom-frac rows of V_lm = "weak-readout" directions (logits read them weakly).
   Energy of w's row-space (output side) in those directions / total ||w||².
   Null = frac.
   ratio > 1 ⇒ w writes into directions the unembedding ignores.

Two activation-based subspaces (Suppressed, Stenographic) are deferred:
they need a probe set of forward passes through the base model.

Returns polars DataFrames so we can group by param-kind (q_proj / o_proj / ...)
and see which families of weights carry the steering signal.
"""

from __future__ import annotations

import re

import polars as pl
import torch
from jaxtyping import Float
from torch import Tensor


@torch.no_grad()
def svd_alignment(
    w: Float[Tensor, "d_out d_in"],
    W: Float[Tensor, "d_out d_in"],
    k_frac: float = 0.1,
) -> dict:
    """Energy fraction of w in the top-k SVD block of W."""
    Wf = W.float()
    wf = w.float()
    U, S, Vt = torch.linalg.svd(Wf, full_matrices=False)
    r = S.numel()
    k = max(1, int(round(k_frac * r)))
    proj = U.T @ wf @ Vt.T  # (r, r) in W's SVD basis
    e_total = (proj * proj).sum()
    e_top = (proj[:k, :k] * proj[:k, :k]).sum()
    e_bot = (proj[k:, k:] * proj[k:, k:]).sum()
    return {
        "k": k,
        "r": r,
        "energy_top": float(e_top / e_total),
        "energy_bot": float(e_bot / e_total),
        "null_top": (k * k) / (r * r),
    }


@torch.no_grad()
def weak_readout_alignment(
    w: Float[Tensor, "d_resid d_in"],
    lm_head_W: Float[Tensor, "vocab d_resid"],
    frac: float = 0.01,
) -> dict:
    """Energy of w's output (d_resid) basis in the bottom-frac of lm_head's input side."""
    Wlm = lm_head_W.float()
    wf = w.float()
    U, S, Vt = torch.linalg.svd(Wlm, full_matrices=False)
    r = S.numel()
    k_weak = max(1, int(round(frac * r)))
    weak = Vt[r - k_weak :]  # (k_weak, d_resid), bottom singulars on input side
    # project rows of w (output side = residual side) onto weak basis
    proj = weak @ wf  # (k_weak, d_in)
    e_total = (wf * wf).sum()
    e_weak = (proj * proj).sum()
    return {
        "energy_weak": float(e_weak / e_total),
        "null_weak": k_weak / r,
        "k_weak": k_weak,
    }


_PROJ_RE = re.compile(r"\.([a-z_]+_proj)\.weight$")
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
# o_proj and down_proj write into residual stream (output dim = d_resid).
RESID_OUT_KINDS = {"o_proj", "down_proj"}


def _kind(name: str) -> str | None:
    m = _PROJ_RE.search(name)
    return m.group(1) if m else None


def _layer_idx(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def alignment_table(
    w: dict[str, Tensor],
    base_state: dict[str, Tensor],
    k_frac: float = 0.1,
    weak_frac: float = 0.01,
) -> pl.DataFrame:
    """Per-layer per-kind alignment table.

    Columns: layer_idx, kind, name, shape, norm, energy_top, null_top, ratio_top,
             energy_weak (if applicable), null_weak, ratio_weak.
    """
    # find lm_head (or tied embed) for weak-readout
    lm_head_W = base_state.get("lm_head.weight")
    if lm_head_W is None:
        lm_head_W = base_state.get("model.embed_tokens.weight")  # tied case

    rows = []
    for name, dw in w.items():
        if dw.dim() != 2:
            continue  # skip biases / norm gains
        W = base_state.get(name)
        if W is None or W.shape != dw.shape:
            continue
        svd = svd_alignment(dw, W, k_frac=k_frac)
        row = {
            "name": name,
            "layer_idx": _layer_idx(name),
            "kind": _kind(name),
            "shape": str(tuple(dw.shape)),
            "norm": float(dw.float().norm()),
            "k": svd["k"],
            "r": svd["r"],
            "energy_top": svd["energy_top"],
            "null_top": svd["null_top"],
            "ratio_top": svd["energy_top"] / svd["null_top"],
        }
        if (
            lm_head_W is not None
            and _kind(name) in RESID_OUT_KINDS
            and dw.shape[0] == lm_head_W.shape[1]
        ):
            wr = weak_readout_alignment(dw, lm_head_W, frac=weak_frac)
            row["energy_weak"] = wr["energy_weak"]
            row["null_weak"] = wr["null_weak"]
            row["ratio_weak"] = wr["energy_weak"] / wr["null_weak"]
        else:
            row["energy_weak"] = None
            row["null_weak"] = None
            row["ratio_weak"] = None
        rows.append(row)
    return pl.DataFrame(rows)


def summarize_by_kind(df: pl.DataFrame) -> pl.DataFrame:
    """Group by kind (q_proj / o_proj / ...): mean ± std of alignment ratios."""
    if df.is_empty():
        return pl.DataFrame(schema={"kind": pl.Utf8, "mean_ratio_top": pl.Float64,
                                    "std_ratio_top": pl.Float64, "mean_ratio_weak": pl.Float64,
                                    "std_ratio_weak": pl.Float64, "mean_norm": pl.Float64,
                                    "n": pl.UInt32})
    return (
        df.group_by("kind")
        .agg(
            pl.col("ratio_top").mean().alias("mean_ratio_top"),
            pl.col("ratio_top").std().alias("std_ratio_top"),
            pl.col("ratio_weak").mean().alias("mean_ratio_weak"),
            pl.col("ratio_weak").std().alias("std_ratio_weak"),
            pl.col("norm").mean().alias("mean_norm"),
            pl.len().alias("n"),
        )
        .sort("kind")
    )
