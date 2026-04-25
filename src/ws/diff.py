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
    delta_pos: dict[str, Tensor], delta_neg: dict[str, Tensor]
) -> dict[str, Float[Tensor, "..."]]:
    """w = delta_pos - delta_neg, only over keys present in both."""
    keys = set(delta_pos) & set(delta_neg)
    if not keys:
        raise ValueError("no overlapping keys between pos and neg deltas")
    w = {k: delta_pos[k] - delta_neg[k] for k in keys}
    norm = float(sum((v.float() ** 2).sum() for v in w.values()) ** 0.5)
    pos_norm = float(sum((v.float() ** 2).sum() for v in delta_pos.values()) ** 0.5)
    neg_norm = float(sum((v.float() ** 2).sum() for v in delta_neg.values()) ** 0.5)
    logger.info(
        f"diff w: {len(w)} keys, {sum(v.numel() for v in w.values()):,} params, "
        f"||w||={norm:.4g}, ||θ+||={pos_norm:.4g}, ||θ-||={neg_norm:.4g}"
    )
    if norm == 0:
        logger.warning("||w|| == 0: pos and neg adapters are identical; steering will be a no-op")
    return w


def save_diff(w: dict[str, Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(w, path)
    logger.info(f"saved diff to {path}")


def load_diff(path: Path) -> dict[str, Float[Tensor, "..."]]:
    return torch.load(path, map_location="cpu")
