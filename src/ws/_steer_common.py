"""Shared steering primitives used by KL calibration.

Provides:
  - chat-template builders (text + ids)
  - steering_context: dW / base under one with-block
  - greedy_generate_under_steering: greedy-roll n_new_tokens with dW steering
  - teacher_force_logp: forward fixed ids, return log-probs at last n positions
  - log_sample_prompt: dumps the full chat-templated string with special tokens
    visible so prompt-formatting bugs surface in logs
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from loguru import logger
from torch import Tensor

from ws._tok_extras import chat_template_extras  # noqa: F401 (re-export)
from ws.steer import weight_steer


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def build_chat_text(tok, system: str, user: str, assistant_prefix: str,
                    *, thinking: bool = False) -> str:
    """Render [sys?, user, assistant=prefix] through the model's chat template.

    `continue_final_message=True` means the assistant turn stays open, so the
    next-token distribution is over the *continuation* of `assistant_prefix`,
    not over a fresh assistant turn header.

    If `thinking=True`, post-process the rendered text so the assistant turn
    ends inside an *open* `<think>` block — Qwen3's chat template auto-injects
    `<think>\\n\\n</think>\\n\\n` when the prefix doesn't start with `<think>`.
    We snip everything after the last `<think>` so the next-token distribution
    is over reasoning tokens, matching the gist's "20 thinking tokens" budget.
    """
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": assistant_prefix})
    text = tok.apply_chat_template(
        msgs, tokenize=False,
        continue_final_message=True, add_generation_prompt=False,
        **chat_template_extras(tok),
    )
    if thinking:
        idx = text.rfind(THINK_OPEN)
        if idx >= 0:
            text = text[: idx + len(THINK_OPEN)] + "\n"
    return text


def build_chat_ids(tok, system: str, user: str, assistant_prefix: str,
                   max_total: int = 512, *, thinking: bool = False) -> Tensor:
    text = build_chat_text(tok, system, user, assistant_prefix, thinking=thinking)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_total)
    return enc.input_ids.squeeze(0)


@contextmanager
def steering_context(method: str, alpha: float, *, model, w=None):
    """Steering context for dW: methods (or base for unsteered pass)."""
    if method.startswith("dW:"):
        with weight_steer(model, w, alpha):
            yield
    elif method == "base":
        yield
    else:
        raise ValueError(f"unknown method: {method}")


@torch.no_grad()
def greedy_generate_under_steering(
    model, tok, input_ids: Tensor, *, method: str, alpha: float,
    n_new_tokens: int, w=None,
) -> tuple[Tensor, Tensor]:
    """Greedy-generate n_new_tokens under dW steering. Returns (gen_ids[T], logp_steered[T,V])."""
    with steering_context(method, alpha, model=model, w=w):
        out = model.generate(
            input_ids.unsqueeze(0).to(model.device),
            max_new_tokens=n_new_tokens, do_sample=False, temperature=1.0,
            return_dict_in_generate=True, output_scores=True,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        )
    new_ids = out.sequences[0, input_ids.shape[0]:].cpu()
    # output_scores: tuple of [B, V] tensors, one per generated step
    logp_steered = torch.stack(
        [s[0].float().log_softmax(-1) for s in out.scores], dim=0
    ).cpu()
    # If gen stopped early on EOS, scores has one extra step than new_ids; trim
    logp_steered = logp_steered[: new_ids.shape[0]]
    return new_ids, logp_steered


@torch.no_grad()
def teacher_force_logp(model, full_ids: Tensor, n_tokens: int) -> Tensor:
    """Forward `full_ids` once, return log-probs at the last n_tokens positions.

    Specifically: returns log-probs of distributions that *predict* the last
    n_tokens of `full_ids` (i.e. positions [-n_tokens-1 : -1] of the logits).
    """
    out = model(input_ids=full_ids.unsqueeze(0).to(model.device))
    logits = out.logits[0, -n_tokens - 1:-1]
    return logits.float().log_softmax(-1).cpu()


def log_sample_prompt(tok, text: str, *, generated_ids: Tensor | None = None,
                      label: str = "sample", max_chars: int = 1200) -> None:
    """Log the full chat-templated prompt with special tokens visible.

    Use this once per method/per script run. The point is: if the chat
    template silently changes between calib and eval, you see it in the log
    before debugging metrics.
    """
    snippet = text if len(text) <= max_chars else (text[:max_chars] + f"…[+{len(text) - max_chars} chars]")
    logger.info(f"[{label}] full prompt (special tokens included):\n{snippet}")
    ids = tok(text, return_tensors="pt").input_ids[0]
    first = tok.convert_ids_to_tokens(ids[: min(8, len(ids))].tolist())
    last = tok.convert_ids_to_tokens(ids[-min(8, len(ids)):].tolist())
    logger.info(f"[{label}] n_input_tokens={ids.shape[0]} first8={first} last8={last}")
    if generated_ids is not None and len(generated_ids) > 0:
        gen_text = tok.decode(generated_ids, skip_special_tokens=False)
        gen_toks = tok.convert_ids_to_tokens(generated_ids.tolist())
        logger.info(f"[{label}] greedy gen ({len(generated_ids)} toks): {gen_text!r}")
        logger.info(f"[{label}] greedy gen tokens: {gen_toks}")
