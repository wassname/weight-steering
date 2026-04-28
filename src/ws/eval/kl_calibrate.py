"""KL-budget calibration: pick α per method to match a prompt's distribution shift.

Why: comparing methods at α=1 is unfair — α=1 means very different things across
LoRA / PiSSA / DeLoRA / OFT / IA3 / RepE / prompt. The principled budget is the
KL footprint of a strong prompt baseline (here: engineered_prompt_honest). For
each method, Newton-search α so that p95 per-token KL(steered ‖ base) over 20
continuation positions matches the prompt's p95 KL.

Pipeline:
  1. Build 4 diverse calibration prompts (code/dialogue/encyclopedia/reasoning).
  2. Forward base model on each → base log-probs at last 20 positions.
  3. Forward steered (sys-prompt-engineered) → measure p95 KL = T (the budget).
  4. For each of {6 adapters, RepE}:
       α ← 1.0 (initial)
       for k in range(n_iters):
         M ← p95 KL(steered_at_α ‖ base) over 4 prompts × 20 tokens
         if M ≈ T: break
         α ← α · sqrt(T / M)            # 1-step Newton for KL ~ α²·F
  5. Audit: at calibrated α per method, recompute p95 on 150 held-out prompts.

KL direction: KL(steered ‖ base) — mode-seeking, captures where steered puts
mass base wouldn't. Standard RL/policy choice. Single-seed: methods are
deterministic; "noise" comes only from prompt sampling, so 4×20 = 80 token
positions is the MC budget for p95.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
import tyro
from baukit import TraceDict
from loguru import logger
from tabulate import tabulate
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from ws._log import final_summary, get_argv, setup_logging
from ws.data import _load_suffixes
from ws.diff import DIFF_FILENAME, load_diff
from ws.eval.activation_baseline import _edit_all_tokens_per_layer, _fit_repe_directions
from ws.eval.prompt_baseline import PROMPTS as PROMPT_TEXTS
from ws.steer import weight_steer

# Diverse calibration: stratified by category. Prompts perturb conditionally
# on topic, so n=4 underestimates p95 wildly. n=50 across all cats gives stable
# p95 (audit on 100 disjoint prompts validated this empirically).
CALIB_CATS = (
    "code", "dialogue", "encyclopedia", "reasoning",
    "ethics", "fact", "stories", "general", "email", "tech",
)


@dataclass
class KLCalibrateCfg:
    model: str = "Qwen/Qwen3-0.6B"
    behavior: str = "honesty"
    out: Path = Path("out")
    adapters: tuple[str, ...] = ("lora", "pissa", "dora", "delora", "oft", "ia3")
    include_repe: bool = True
    n_calib_prompts: int = 50
    n_audit_prompts: int = 100
    n_tokens: int = 20
    target_pct: float = 95.0
    target_prompt: str = "engineered_prompt_honest"
    alpha_init: float = 1.0
    n_newton_iters: int = 4
    convergence_band: float = 0.20  # stop if |M-T|/T < band
    repe_layers: tuple[int, ...] = field(default_factory=lambda: tuple(range(8, 22)))
    n_repe_train: int = 20
    seed: int = 0


def _select_prompts(n_calib: int, n_audit: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Round-robin across CALIB_CATS for stratified calib; random disjoint audit."""
    entries = _load_suffixes(thinking=False)
    by_cat: dict[str, list[dict]] = {}
    for e in entries:
        by_cat.setdefault(e.get("cat", "?"), []).append(e)

    rng = np.random.default_rng(seed)
    for cat in by_cat:
        rng.shuffle(by_cat[cat])

    calib: list[dict] = []
    used_keys: set = set()
    cat_cursors = {cat: 0 for cat in CALIB_CATS}
    while len(calib) < n_calib:
        added_in_round = 0
        for cat in CALIB_CATS:
            if len(calib) >= n_calib:
                break
            if cat not in by_cat:
                continue
            i = cat_cursors[cat]
            if i >= len(by_cat[cat]):
                continue
            e = by_cat[cat][i]
            cat_cursors[cat] += 1
            calib.append(e)
            used_keys.add((e["user_msg"], e["suffix"]))
            added_in_round += 1
        if added_in_round == 0:
            break

    pool = [e for e in entries if (e["user_msg"], e["suffix"]) not in used_keys]
    rng.shuffle(pool)
    audit = pool[:n_audit]
    return calib, audit


def _build_input_ids(tok, system: str, user: str, assistant_prefix: str,
                     n_tokens: int, max_total: int = 256) -> Tensor:
    """Tokenize chat: [sys?, user, assistant=prefix]. Truncate prefix from the end
    so suffix tail (the next-token prediction targets) survives."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": assistant_prefix})
    text = tok.apply_chat_template(
        msgs, tokenize=False,
        continue_final_message=True, add_generation_prompt=False,
    )
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_total)
    ids = enc.input_ids.squeeze(0)
    # Need at least n_tokens+2 to take logits at [-n_tokens-1:-1]
    if ids.shape[0] < n_tokens + 2:
        raise ValueError(f"input too short ({ids.shape[0]}) for n_tokens={n_tokens}")
    return ids


@torch.no_grad()
def _forward_logp(model, input_ids: Tensor, n_tokens: int) -> Tensor:
    """Returns log-softmax at last n_tokens positions of next-token preds → [n_tokens, V]."""
    out = model(input_ids=input_ids.unsqueeze(0).to(model.device))
    logits = out.logits[0, -n_tokens - 1:-1]  # predicts last n_tokens of input_ids
    return logits.float().log_softmax(-1).cpu()


def _kl_per_token(logp_steered: Tensor, logp_base: Tensor) -> Tensor:
    """KL(p_steered ‖ p_base) per position. Both [T, V] log-prob tensors."""
    p = logp_steered.exp()
    return (p * (logp_steered - logp_base)).sum(-1)  # [T]


@torch.no_grad()
def _measure_kl(method: str, alpha: float, *, model, tok, prompts, n_tokens,
                w=None, repe_dirs=None, repe_layers=None) -> dict:
    """Return per-token KL stats for given method+α across all prompts."""
    # Build base inputs (no system prompt).
    base_ids_list = [
        _build_input_ids(tok, "", p["user_msg"], p["suffix"], n_tokens)
        for p in prompts
    ]

    # Base log-probs (vanilla model, no steering, no sys prompt).
    base_logps = [_forward_logp(model, ids, n_tokens) for ids in base_ids_list]

    # Build steered inputs and forward under steering.
    steered_logps: list[Tensor] = []
    if method.startswith("prompt:"):
        sys_prompt = PROMPT_TEXTS[method.split(":", 1)[1]]
        steered_ids_list = [
            _build_input_ids(tok, sys_prompt, p["user_msg"], p["suffix"], n_tokens)
            for p in prompts
        ]
        for ids in steered_ids_list:
            steered_logps.append(_forward_logp(model, ids, n_tokens))
    elif method.startswith("dW:"):
        with weight_steer(model, w, alpha):
            for ids in base_ids_list:
                steered_logps.append(_forward_logp(model, ids, n_tokens))
    elif method == "repe":
        hooks = [f"model.layers.{L}" for L in repe_layers]
        layer_list = list(repe_layers)
        edit = _edit_all_tokens_per_layer(repe_dirs, layer_list, alpha)
        for ids in base_ids_list:
            with TraceDict(model, hooks, edit_output=edit):
                out = model(input_ids=ids.unsqueeze(0).to(model.device))
            logits = out.logits[0, -n_tokens - 1:-1]
            steered_logps.append(logits.float().log_softmax(-1).cpu())
    else:
        raise ValueError(f"unknown method: {method}")

    # Per-token KL across all (prompt, position) pairs.
    kls = torch.cat([_kl_per_token(s, b) for s, b in zip(steered_logps, base_logps)])
    arr = kls.numpy()
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "n": int(arr.shape[0]),
    }


def _newton_calibrate(method: str, target: float, *, model, tok, prompts, cfg,
                      w=None, repe_dirs=None) -> dict:
    """Newton iterations: α_next = α · sqrt(T / M). Stops when within band or n iters."""
    alpha = float(cfg.alpha_init)
    history = []
    for k in range(cfg.n_newton_iters):
        m = _measure_kl(
            method, alpha, model=model, tok=tok, prompts=prompts,
            n_tokens=cfg.n_tokens, w=w, repe_dirs=repe_dirs,
            repe_layers=cfg.repe_layers,
        )
        ratio = m["p95"] / target if target > 0 else 1.0
        history.append({"iter": k, "alpha": alpha, **m, "ratio": ratio})
        logger.info(
            f"  [{method}] iter={k} α={alpha:+.4f} p95={m['p95']:.4g} "
            f"mean={m['mean']:.4g} max={m['max']:.4g} ratio={ratio:.3f}"
        )
        if abs(ratio - 1.0) < cfg.convergence_band:
            break
        # Newton step (KL ~ α²·F): α_next = α · sqrt(T/M)
        if m["p95"] <= 0:
            alpha *= 2.0
            continue
        alpha = alpha * float(np.sqrt(target / m["p95"]))

    final = history[-1]
    converged = abs(final["ratio"] - 1.0) < cfg.convergence_band
    return {
        "method": method,
        "calibrated_alpha": final["alpha"],
        "p95_at_calib": final["p95"],
        "mean_at_calib": final["mean"],
        "max_at_calib": final["max"],
        "ratio_at_calib": final["ratio"],
        "iterations": len(history),
        "converged": converged,
        "history": history,
    }


def main(cfg: KLCalibrateCfg) -> None:
    setup_logging("kl_calibrate")
    out_dir = cfg.out / cfg.behavior / "kl_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    calib_prompts, audit_prompts = _select_prompts(cfg.n_calib_prompts, cfg.n_audit_prompts, cfg.seed)
    logger.info(f"calibration prompts: {[p.get('cat') for p in calib_prompts]}")
    logger.info(f"audit prompts: n={len(audit_prompts)}")

    # 1. Establish target T from the prompt anchor.
    target_method = f"prompt:{cfg.target_prompt}"
    logger.info(f"\n=== ANCHOR: {target_method} ===")
    anchor = _measure_kl(
        target_method, alpha=1.0, model=model, tok=tok, prompts=calib_prompts,
        n_tokens=cfg.n_tokens,
    )
    target = anchor["p95"]
    logger.info(f"target p95 KL = {target:.4g} (mean={anchor['mean']:.4g}, max={anchor['max']:.4g})")

    # Also measure simple_honest_prompt for reference.
    ref_methods = ["simple_honest_prompt", "engineered_prompt_dishonest", "simple_dishonest_prompt"]
    prompt_refs = {target_method: anchor}
    for name in ref_methods:
        if name in PROMPT_TEXTS and name != cfg.target_prompt:
            m = _measure_kl(
                f"prompt:{name}", alpha=1.0, model=model, tok=tok,
                prompts=calib_prompts, n_tokens=cfg.n_tokens,
            )
            prompt_refs[f"prompt:{name}"] = m
            logger.info(f"  ref prompt:{name} p95={m['p95']:.4g} mean={m['mean']:.4g}")

    # 2. Fit RepE directions once (used only if include_repe).
    repe_dirs = None
    if cfg.include_repe:
        logger.info("\n=== fit RepE directions ===")
        repe_dirs = _fit_repe_directions(model, tok, cfg.n_repe_train, cfg.behavior)

    # 3. Newton-calibrate each adapter and (optionally) RepE.
    results = []
    for adapter in cfg.adapters:
        logger.info(f"\n=== calibrate dW:{adapter} ===")
        w = load_diff(cfg.out / cfg.behavior / adapter / DIFF_FILENAME)
        r = _newton_calibrate(
            f"dW:{adapter}", target, model=model, tok=tok,
            prompts=calib_prompts, cfg=cfg, w=w,
        )
        results.append(r)

    if cfg.include_repe:
        logger.info("\n=== calibrate repe ===")
        r = _newton_calibrate(
            "repe", target, model=model, tok=tok,
            prompts=calib_prompts, cfg=cfg, repe_dirs=repe_dirs,
        )
        results.append(r)

    # 4. Audit: at calibrated α, recompute on n_audit prompts.
    logger.info(f"\n=== AUDIT (n={len(audit_prompts)} prompts) ===")
    # Anchor audit
    anchor_audit = _measure_kl(
        target_method, alpha=1.0, model=model, tok=tok,
        prompts=audit_prompts, n_tokens=cfg.n_tokens,
    )
    logger.info(f"  anchor audit p95={anchor_audit['p95']:.4g} (calib was {target:.4g})")

    audit_rows = [{
        "method": target_method,
        "alpha": 1.0,
        "p95_calib": target,
        "mean_calib": anchor["mean"],
        "p95_audit": anchor_audit["p95"],
        "mean_audit": anchor_audit["mean"],
        "max_audit": anchor_audit["max"],
        "calib_audit_ratio": anchor_audit["p95"] / target if target > 0 else float("nan"),
    }]

    for r in results:
        method = r["method"]
        alpha = r["calibrated_alpha"]
        if method.startswith("dW:"):
            adapter = method.split(":", 1)[1]
            w = load_diff(cfg.out / cfg.behavior / adapter / DIFF_FILENAME)
            m_audit = _measure_kl(
                method, alpha, model=model, tok=tok, prompts=audit_prompts,
                n_tokens=cfg.n_tokens, w=w,
            )
        elif method == "repe":
            m_audit = _measure_kl(
                method, alpha, model=model, tok=tok, prompts=audit_prompts,
                n_tokens=cfg.n_tokens, repe_dirs=repe_dirs,
                repe_layers=cfg.repe_layers,
            )
        else:
            raise ValueError(method)
        logger.info(
            f"  {method} α={alpha:+.3f} audit p95={m_audit['p95']:.4g} "
            f"(calib was {r['p95_at_calib']:.4g}, target {target:.4g})"
        )
        audit_rows.append({
            "method": method,
            "alpha": alpha,
            "p95_calib": r["p95_at_calib"],
            "mean_calib": r["mean_at_calib"],
            "p95_audit": m_audit["p95"],
            "mean_audit": m_audit["mean"],
            "max_audit": m_audit["max"],
            "calib_audit_ratio": m_audit["p95"] / r["p95_at_calib"] if r["p95_at_calib"] > 0 else float("nan"),
        })

    audit_df = pl.DataFrame(audit_rows)
    audit_path = out_dir / "audit.csv"
    audit_df.write_csv(audit_path)

    # Per-method summary (from results).
    summary_rows = []
    for r in results:
        summary_rows.append({
            "method": r["method"],
            "calibrated_alpha": r["calibrated_alpha"],
            "p95_at_calib": r["p95_at_calib"],
            "mean_at_calib": r["mean_at_calib"],
            "max_at_calib": r["max_at_calib"],
            "ratio_at_calib": r["ratio_at_calib"],
            "iterations": r["iterations"],
            "converged": r["converged"],
        })
    summary_df = pl.DataFrame(summary_rows).sort("calibrated_alpha")
    summary_df = summary_df.with_columns(pl.lit(target).alias("target_p95"))
    summary_path = out_dir / "summary.csv"
    summary_df.write_csv(summary_path)

    # Per-iteration history (Newton trace, for diagnostics).
    history_rows = []
    for r in results:
        for h in r["history"]:
            history_rows.append({"method": r["method"], **h})
    pl.DataFrame(history_rows).write_csv(out_dir / "newton_history.csv")

    # Prompt-anchor reference table.
    pl.DataFrame([
        {"method": k, **v} for k, v in prompt_refs.items()
    ]).write_csv(out_dir / "prompt_refs.csv")

    print("\n=== KL calibration summary ===")
    print(f"target p95 KL (anchor={target_method}) = {target:.4g} nats")
    print(tabulate(summary_df.to_pandas(), headers="keys", tablefmt="tsv",
                   floatfmt="+.4g", showindex=False))
    print("\naudit (held-out 150 prompts):")
    print(tabulate(audit_df.to_pandas(), headers="keys", tablefmt="tsv",
                   floatfmt="+.4g", showindex=False))

    cue = "🟢" if all(r["converged"] for r in results) else "🟡"
    final_summary(
        out=summary_path,
        argv=get_argv(),
        main_metric=f"target_p95={target:.4g} converged={sum(r['converged'] for r in results)}/{len(results)}",
        cue=cue,
        table_rows=summary_df.select("method", "calibrated_alpha", "p95_at_calib",
                                      "ratio_at_calib", "iterations", "converged").rows(),
        headers=["method", "alpha", "p95", "ratio", "iters", "ok"],
        floatfmt="",
    )


if __name__ == "__main__":
    main(tyro.cli(KLCalibrateCfg))
