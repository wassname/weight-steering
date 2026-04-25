"""Apply scaled weight diff at inference: W <- W + alpha * w.

Weight steering (this file) modifies model parameters in place inside a context
manager and restores them on exit. This matches the original paper's "additive
weight steering": (W + alpha * w) @ x + b, equivalent to adding alpha * w to W
before the forward pass.

We use direct in-place parameter edits rather than activation hooks because
we are perturbing W itself, not the activations. (baukit / TraceDict is the
right primitive when you instead want to steer hidden states; not used here.)
"""

from contextlib import contextmanager

import torch
from loguru import logger
from torch import Tensor, nn


@contextmanager
def weight_steer(model: nn.Module, w: dict[str, Tensor], alpha: float):
    """Add alpha * w to model weights for the scope of the context.

    On exit, restores original weights exactly. Safe to nest as long as keys
    don't overlap.
    """
    if alpha == 0.0 or not w:
        yield
        return

    params = dict(model.named_parameters())
    backup: dict[str, Tensor] = {}
    missing: list[str] = []

    for k, dw in w.items():
        if k not in params:
            missing.append(k)
            continue
        p = params[k]
        backup[k] = p.data.detach().clone()
        p.data.add_(dw.to(p.device, p.dtype), alpha=alpha)

    if missing:
        logger.warning(f"weight_steer: {len(missing)} keys not in model (e.g. {missing[:3]})")

    try:
        yield
    finally:
        for k, orig in backup.items():
            params[k].data.copy_(orig)


def apply_diff_permanent(model: nn.Module, w: dict[str, Tensor], alpha: float) -> None:
    """Same math as weight_steer, but no restore. Use for save-after-merge."""
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, dw in w.items():
            if k in params:
                params[k].data.add_(dw.to(params[k].device, params[k].dtype), alpha=alpha)
