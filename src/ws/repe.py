"""Reusable RepE-style activation helpers for steering and calibration."""

from __future__ import annotations

import torch
from baukit import TraceDict
from torch import Tensor

from ws.data import (
    HONESTY_NEG_PERSONAS,
    HONESTY_POS_PERSONAS,
    HONESTY_PROMPT,
    SYCOPHANCY_NEG_PERSONAS,
    SYCOPHANCY_POS_PERSONAS,
    TRAD_CARE_NEG_PERSONAS,
    TRAD_CARE_POS_PERSONAS,
    TRAD_CARE_PROMPT,
    _load_suffixes,
    train_topics,
)
from ws.eval.sycophancy import EVAL_HEADER as SYC_EVAL_HEADER


def _chat_text(tok, *, user: str, system: str = "", assistant_prefix: str | None = None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    if assistant_prefix is not None:
        msgs.append({"role": "assistant", "content": assistant_prefix})
        return tok.apply_chat_template(
            msgs,
            tokenize=False,
            continue_final_message=True,
            add_generation_prompt=False,
        )
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _block_output(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _replace_block_output(output, x: Tensor):
    if isinstance(output, tuple):
        return (x, *output[1:])
    return x


@torch.no_grad()
def _capture_last_token_blocks(
    model,
    tok,
    prompts: list[str],
    *,
    system: str,
    assistant_prefixes: list[str] | None = None,
) -> Tensor:
    hooks = [f"model.layers.{layer}" for layer in range(model.config.num_hidden_layers)]
    if assistant_prefixes is None:
        texts = [_chat_text(tok, user=prompt, system=system) for prompt in prompts]
    else:
        assert len(assistant_prefixes) == len(prompts)
        texts = [
            _chat_text(tok, user=prompt, system=system, assistant_prefix=prefix)
            for prompt, prefix in zip(prompts, assistant_prefixes, strict=True)
        ]
    old_padding_side = tok.padding_side
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
    tok.padding_side = old_padding_side
    seq_idx = torch.full((enc.input_ids.shape[0],), enc.input_ids.shape[1] - 1, device=model.device)
    with TraceDict(model, hooks, retain_output=True) as ret:
        _ = model(**enc)
    rows = []
    for hook in hooks:
        x = _block_output(ret[hook].output)
        b, _s, d = x.shape
        rows.append(x.gather(1, seq_idx.view(b, 1, 1).expand(b, 1, d)).squeeze(1).float().cpu())
    return torch.stack(rows, 0)


def fit_repe_directions(model, tok, n_train_topics: int, behavior: str) -> Tensor:
    """Fit one PCA direction per layer from persona contrasts."""
    if behavior == "sycophancy":
        prompts = [f"I think {claim}. {question}" for claim, question in train_topics()[:n_train_topics]]
        sys_pos = SYCOPHANCY_POS_PERSONAS[0]
        sys_neg = SYCOPHANCY_NEG_PERSONAS[0]
        assistant_prefixes = None
    elif behavior == "honesty":
        entries = _load_suffixes(thinking=False)[:n_train_topics]
        prompts = [entry["user_msg"] for entry in entries]
        assistant_prefixes = [entry["suffix"] for entry in entries]
        sys_pos = HONESTY_PROMPT.format(persona=HONESTY_POS_PERSONAS[0])
        sys_neg = HONESTY_PROMPT.format(persona=HONESTY_NEG_PERSONAS[0])
    elif behavior == "trad_care":
        entries = _load_suffixes(thinking=False)[:n_train_topics]
        prompts = [entry["user_msg"] for entry in entries]
        assistant_prefixes = [entry["suffix"] for entry in entries]
        sys_pos = TRAD_CARE_PROMPT.format(persona=TRAD_CARE_POS_PERSONAS[0])
        sys_neg = TRAD_CARE_PROMPT.format(persona=TRAD_CARE_NEG_PERSONAS[0])
    else:
        raise ValueError(f"unknown behavior: {behavior}")

    hs_pos = _capture_last_token_blocks(
        model, tok, prompts, system=sys_pos, assistant_prefixes=assistant_prefixes
    ).float()
    hs_neg = _capture_last_token_blocks(
        model, tok, prompts, system=sys_neg, assistant_prefixes=assistant_prefixes
    ).float()
    diffs = hs_pos - hs_neg
    diffs_centered = diffs - diffs.mean(dim=1, keepdim=True)
    _u, _s, vh = torch.linalg.svd(diffs_centered, full_matrices=False)
    directions = vh[:, 0, :]
    proj_pos = torch.einsum("lpd,ld->lp", hs_pos, directions).mean(dim=1)
    proj_neg = torch.einsum("lpd,ld->lp", hs_neg, directions).mean(dim=1)
    flip = (proj_pos < proj_neg).float() * -2 + 1
    return directions * flip.unsqueeze(-1)


def edit_all_tokens_per_layer(directions: Tensor, layer_indices: list[int], coeff: float):
    """Canonical RepE edit: add coeff * direction at every token for each hooked layer."""
    layer_to_dir = {f"model.layers.{layer}": directions[layer] for layer in layer_indices}

    def edit(output, layer_name):
        direction = layer_to_dir[layer_name]
        x0 = _block_output(output)
        x = x0.clone()
        d = x.shape[-1]
        delta = direction.to(device=x.device, dtype=x.dtype).view(1, 1, d)
        x = x + coeff * delta
        return _replace_block_output(output, x)

    return edit
