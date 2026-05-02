"""Weight-space diff: w = θ+ - θ-.

Functional replacement for the original TaskVector class.

Each adapter (LoRA / DoRA / PiSSA-init / DeLoRA) is merged into a delta over the
base model: delta = merged_W - base_W.  The behavior direction is then

    w_layer = delta_pos[layer] - delta_neg[layer]

Working in delta-W space (rather than diffing raw A/B factors) makes the four
adapter families directly comparable: every adapter produces a delta living
in the same ambient space as W.
"""

from pathlib import Path

import torch
from jaxtyping import Float
from loguru import logger
from peft import PeftModel
from torch import Tensor
from transformers import AutoModelForCausalLM


DIFF_FILENAME = "w.pt"


def load_base_state(model_id: str, dtype=torch.bfloat16) -> dict[str, Float[Tensor, "..."]]:
    """Return CPU state dict of the pretrained base model. Snapshot once, reuse."""
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    sd = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    del base
    return sd


def load_delta(
    model_id: str,
    adapter_path: Path,
    base_state: dict[str, Tensor] | None = None,
    dtype=torch.bfloat16,
) -> dict[str, Float[Tensor, "..."]]:
    """Merge an adapter into base and return the per-key delta (merged - base).

    Only returns keys whose delta is non-zero (i.e. parameters the adapter touched).
    """
    if base_state is None:
        base_state = load_base_state(model_id, dtype=dtype)

    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    peft_model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = peft_model.merge_and_unload()
    merged_state = merged.state_dict()

    delta = {}
    for k, v in merged_state.items():
        if k not in base_state:
            continue
        d = (v.detach().cpu() - base_state[k]).to(dtype)
        if d.abs().sum() > 0:
            delta[k] = d

    logger.info(f"delta from {adapter_path}: {len(delta)} touched params")
    del base, peft_model, merged
    return delta


def compute_diff(
    delta_pos: dict[str, Tensor],
    delta_neg: dict[str, Tensor],
    mode: str = "dw",
) -> dict[str, Float[Tensor, "..."]]:
    """Behavior direction in delta-W space.

    mode='dw'        : w = τ⁺ − τ⁻  (paper's contrastive task vector)
    mode='bisector'  : w ∝ τ̂⁺ − τ̂⁻, length-normalize each side then subtract,
                       rescale to ‖dW‖. Treats each adapter as a direction so a
                       louder fine-tune doesn't dominate. Coefficient sweeps stay
                       comparable across modes because of the rescale.
    """
    keys = set(delta_pos) & set(delta_neg)
    if not keys:
        logger.warning("compute_diff: no overlapping keys -- both deltas may be zero "
                       "(e.g. IA3 with too few training steps). Returning empty diff.")
        return {}

    pos_norm_sq = sum(float((delta_pos[k].float() ** 2).sum()) for k in keys)
    neg_norm_sq = sum(float((delta_neg[k].float() ** 2).sum()) for k in keys)
    pos_norm, neg_norm = pos_norm_sq ** 0.5, neg_norm_sq ** 0.5

    if mode == "dw":
        w = {k: delta_pos[k] - delta_neg[k] for k in keys}
    elif mode == "bisector":
        pn = sum(float((delta_pos[k].float() * delta_neg[k].float()).sum()) for k in keys)
        dW_norm = (pos_norm_sq - 2 * pn + neg_norm_sq) ** 0.5
        raw = {k: delta_pos[k].float() / pos_norm - delta_neg[k].float() / neg_norm
               for k in keys}
        raw_norm = sum(float((v ** 2).sum()) for v in raw.values()) ** 0.5
        scale = dW_norm / raw_norm if raw_norm > 0 else 1.0
        w = {k: (v * scale).to(delta_pos[k].dtype) for k, v in raw.items()}
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected 'dw' or 'bisector')")

    norm = float(sum((v.float() ** 2).sum() for v in w.values()) ** 0.5)
    logger.info(
        f"diff w (mode={mode}): {len(w)} keys, {sum(v.numel() for v in w.values()):,} params, "
        f"||w||={norm:.4g}, ||θ+||={pos_norm:.4g}, ||θ-||={neg_norm:.4g}"
    )
    if norm == 0:
        logger.warning("||w|| == 0: pos and neg adapters are identical; steering will be a no-op")
    return w


def diagnostics(
    delta_pos: dict[str, Tensor], delta_neg: dict[str, Tensor]
) -> dict[str, float]:
    """Geometric diagnostics on (τ⁺, τ⁻) before forming a steering vector.

    cos_anti     = cos(τ⁺, −τ⁻). →1 means τ⁺ and τ⁻ are antipodal (clean contrast).
    asymmetry    = ‖τ⁺‖/‖τ⁻‖. ≠1 means one fine-tune is louder than the other.
    drift_ratio  = ‖M‖/‖b‖ with M=(τ⁺+τ⁻)/2 (common drift), b=(τ⁺−τ⁻)/2 (behavior).
                   ≫1 means common-mode dominates differential; bisector might help.
    cos_dW_M     = |cos(dW, M)|. Fraction of paper's dW that points along common drift.
                   ≪1 means dW already sits in M⊥ (drop-midpoint would be a no-op).
    """
    keys = set(delta_pos) & set(delta_neg)
    p2 = sum(float((delta_pos[k].float() ** 2).sum()) for k in keys)
    n2 = sum(float((delta_neg[k].float() ** 2).sum()) for k in keys)
    pn = sum(float((delta_pos[k].float() * delta_neg[k].float()).sum()) for k in keys)

    p_norm, n_norm = p2 ** 0.5, n2 ** 0.5
    cos_anti = -pn / (p_norm * n_norm) if p_norm * n_norm > 0 else 0.0
    dW_norm = (p2 - 2 * pn + n2) ** 0.5
    M_norm = ((p2 + 2 * pn + n2) / 4) ** 0.5
    b_norm = dW_norm / 2
    drift_ratio = M_norm / b_norm if b_norm > 0 else 0.0
    cos_dW_M = abs((p2 - n2) / 2) / (dW_norm * M_norm) if dW_norm * M_norm > 0 else 0.0

    d = {
        "norm_pos": p_norm, "norm_neg": n_norm,
        "asymmetry": p_norm / n_norm if n_norm > 0 else float("inf"),
        "cos_anti": cos_anti,
        "norm_dW": dW_norm, "norm_M": M_norm,
        "drift_ratio": drift_ratio,
        "cos_dW_M": cos_dW_M,
    }
    logger.info(
        "geometry: cos(τ⁺,-τ⁻)={cos_anti:+.3f}  ‖τ⁺‖/‖τ⁻‖={asymmetry:.3f}  "
        "‖M‖/‖b‖={drift_ratio:.3f}  |cos(dW,M)|={cos_dW_M:.3f}".format(**d)
    )
    logger.info(
        "SHOULD: cos_anti→+1 (antipodal). drift_ratio≪1 (clean). "
        "|cos(dW,M)|≪1 means paper's dW already orthogonal to drift."
    )
    return d


def save_diff(w: dict[str, Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(w, path)
    logger.info(f"saved diff to {path}")


def load_diff(path: Path) -> dict[str, Float[Tensor, "..."]]:
    return torch.load(path, map_location="cpu")
