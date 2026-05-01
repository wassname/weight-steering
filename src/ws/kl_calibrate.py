"""KL-budget calibration: pick α per method to match a prompt's distribution shift.

Why: comparing methods at α=1 is unfair — α=1 means very different things across
LoRA / PiSSA / DeLoRA / OFT / IA3 / RepE / prompt. The principled budget is the
KL footprint of a strong prompt baseline (here: engineered_prompt_honest). For
each method, Newton-search α so that p95 per-token KL(steered ‖ base) over the
greedy-generated trajectory matches the prompt's p95 KL.

Methodology (matches the gist
https://gist.github.com/wassname/6c11cf30b43d8c228bc114795f1019c7):

  For each prompt:
    1. Greedy-generate `n_tokens` continuation tokens under the *steered* model.
       This gives the trajectory the steered policy actually walks, plus the
       per-step steered log-probs from generate(output_scores=True).
    2. Append those generated tokens to the *base* prompt (no system prompt,
       no steering) and teacher-force one forward to score them under base.
    3. Per-position KL(steered ‖ base) = Σ p_s · (logp_s − logp_b) along
       the steered trajectory.

  This is mode-seeking KL on the *generated* path — captures cumulative drift
  that fixed-continuation KL misses. p95 over (prompts × positions) is the
  "no-spike" stat we calibrate against.

Search: exponential bracket on α, then Illinois regula-falsi in log-(α, p95).
Plain bisection is linear; stat(α) is roughly p95 ~ α^k near root, which is
linear in (log α, log p95), so log-space false-position usually converges in
3-4 iters. Illinois rule (halve the stuck side's f when same bracket end is
kept twice in a row) breaks the stuck-endpoint failure mode of pure regula
falsi. Generalises the gist's bisection — same bracket, faster inner loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
import tyro
from loguru import logger
from tabulate import tabulate
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from ws._log import final_summary, get_argv, setup_logging
from ws.data import _load_suffixes
from ws.diff import DIFF_FILENAME, load_diff
from ws._steer_common import (
    build_chat_ids,
    build_chat_text,
    greedy_generate_under_steering,
    log_sample_prompt,
    teacher_force_logp,
)
from ws.prompt_texts import PROMPTS as PROMPT_TEXTS
from ws.repe import fit_repe_directions

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
    n_tokens: int = 50
    target_pct: float = 95.0
    # "Side of the road" = 1 nat per-token KL (gist):
    # https://gist.github.com/wassname/6c11cf30b43d8c228bc114795f1019c7
    # Newton residual is 1 − p95(KL); we search a global coefficient C such
    # that p95 KL = target_kl at α=1.
    target_kl: float = 0.5
    target_prompt: str = "engineered_prompt_honest"  # logged as a reference, not the target
    # Bracket guard (lo, hi) on the global coefficient. KL ~ α²·F near root, so
    # below ~0.05 nothing happens; above ~16 we'd be deep in collapse-land.
    bracket_lo: float = 0.05
    bracket_hi: float = 16.0
    n_root_iters: int = 12  # Illinois inner loop; usually converges in 3-5
    convergence_tol: float = 0.05  # |p95 - target| < tol (absolute, in nats)
    repe_layers: tuple[int, ...] = field(default_factory=lambda: tuple(range(8, 22)))
    n_repe_train: int = 50
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


def _system_prompts_for(method: str) -> tuple[str, str]:
    """Return (sys_for_steered_pass, sys_for_base_pass).

    For prompt: methods, the "steering" is the system prompt; base has none.
    For dW / repe / base, both passes use the same (empty) system prompt and
    steering is applied at runtime.
    """
    if method.startswith("prompt:"):
        return PROMPT_TEXTS[method.split(":", 1)[1]], ""
    return "", ""


@torch.no_grad()
def _measure_kl_along_trajectory(
    method: str, alpha: float, *, model, tok, prompts, n_tokens,
    w=None, repe_dirs=None, repe_layers=None,
    log_first_sample: bool = False, sample_label: str = "",
) -> dict:
    """KL(steered ‖ base) per token along the steered greedy trajectory.

    For each prompt:
      1. Build steered_ids (with sys prompt if method=prompt:).
      2. Greedy-generate n_tokens under steering -> (gen_ids, logp_steered[T,V]).
      3. Build base_ids (no sys prompt) + gen_ids; teacher-force base -> logp_base[T,V].
      4. KL_t = Σ_v p_steered_t(v) · (logp_steered_t(v) − logp_base_t(v)).
    """
    sys_steered, sys_base = _system_prompts_for(method)

    all_kls: list[Tensor] = []
    for i, p in enumerate(prompts):
        # thinking=True: assistant turn ends in open `<think>\n` so the 20
        # greedy tokens are reasoning, not answer continuation. The suffix
        # field is unused here — the gist's protocol is "20 thinking tokens
        # under steering on a question prompt", not "complete this answer".
        steered_input_ids = build_chat_ids(
            tok, sys_steered, p["user_msg"], "", thinking=True,
        )
        if sys_steered == sys_base:
            base_input_ids = steered_input_ids
        else:
            base_input_ids = build_chat_ids(
                tok, sys_base, p["user_msg"], "", thinking=True,
            )

        gen_ids, logp_steered = greedy_generate_under_steering(
            model, tok, steered_input_ids,
            method=method, alpha=alpha, n_new_tokens=n_tokens,
            w=w, repe_dirs=repe_dirs, repe_layers=repe_layers,
        )
        T = gen_ids.shape[0]
        if T == 0:
            continue

        full_base_ids = torch.cat([base_input_ids, gen_ids])
        logp_base = teacher_force_logp(model, full_base_ids, T)

        p_s = logp_steered.exp()
        kl = (p_s * (logp_steered - logp_base)).sum(-1)  # [T]
        all_kls.append(kl)

        if log_first_sample and i == 0:
            text = build_chat_text(tok, sys_steered, p["user_msg"], "", thinking=True)
            label = sample_label or f"calib method={method} α={alpha:+.3f}"
            log_sample_prompt(tok, text, generated_ids=gen_ids, label=label)
            logger.info(
                f"[{label}] kl per pos: {[f'{k:.3f}' for k in kl.tolist()]} "
                f"sum={float(kl.sum()):.3f} max={float(kl.max()):.3f}"
            )

    if not all_kls:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "n": 0}

    arr = torch.cat(all_kls).numpy()
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "n": int(arr.shape[0]),
    }


def _illinois_calibrate(
    method: str,
    target: float,
    *,
    model,
    tok,
    prompts,
    cfg,
    alpha_sign: float = 1.0,
    sign_label: str = "pos",
    w=None,
    repe_dirs=None,
) -> dict:
    """Exponential bracket within (bracket_lo, bracket_hi) then log-log Illinois
    regula falsi. Mirrors steering-lite's validated `calibrate_iso_kl`.

    Geometry: KL ~ α²·F near α=0, saturates at large α → log-log curve concave.
    Plain secant chord lies below the curve, root estimate overshoots, one
    endpoint goes stale. Illinois halves the stale endpoint's log-stat
    (equivalent to dividing v by 2) once it's stuck for 2+ iters, giving
    superlinear convergence on concave segments. Bracket bounds always
    preserved; bisection fallback if interpolation lands outside.
    """
    history: list[dict] = []
    iter_idx = [0]

    def _result(final: dict, converged: bool) -> dict:
        return {
            "method": method,
            "sign": sign_label,
            "alpha_sign": alpha_sign,
            "alpha_mag": abs(final["alpha"]),
            "calibrated_alpha": final["alpha"],
            "p95_at_calib": final["p95"],
            "mean_at_calib": final["mean"],
            "max_at_calib": final["max"],
            "ratio_at_calib": final["ratio"],
            "iterations": len(history),
            "converged": converged,
            "history": history,
        }

    def stat(alpha_mag: float) -> float:
        alpha = alpha_sign * alpha_mag
        m = _measure_kl_along_trajectory(
            method, alpha, model=model, tok=tok, prompts=prompts,
            n_tokens=cfg.n_tokens, w=w, repe_dirs=repe_dirs,
            repe_layers=cfg.repe_layers,
            log_first_sample=(iter_idx[0] == 0),
            sample_label=f"calib iter=0 method={method} sign={sign_label} α={alpha:+.3f}",
        )
        ratio = m["p95"] / target if target > 0 else 1.0
        history.append({
            "iter": iter_idx[0],
            "sign": sign_label,
            "alpha": alpha,
            "alpha_mag": alpha_mag,
            **m,
            "ratio": ratio,
        })
        logger.info(
            f"  [{method}:{sign_label}] iter={iter_idx[0]} α={alpha:+.4f} p95={m['p95']:.4g} "
            f"mean={m['mean']:.4g} max={m['max']:.4g} ratio={ratio:.3f}"
        )
        iter_idx[0] += 1
        return m["p95"]

    lo, hi = float(cfg.bracket_lo), float(cfg.bracket_hi)
    log_target = float(np.log(target))

    # 1. Exponential bracket from geometric mid of (lo, hi)
    mid = float(np.sqrt(lo * hi))
    v_mid = stat(mid)
    if abs(v_mid - target) < cfg.convergence_tol:
        return _result(history[-1], True)

    if v_mid < target:
        c_lo, v_lo = mid, v_mid
        c_hi, v_hi = hi, None
        c = mid
        while c < hi:
            c *= 2.0
            v = stat(c)
            if v >= target:
                c_hi, v_hi = c, v
                break
            c_lo, v_lo = c, v
    else:
        c_hi, v_hi = mid, v_mid
        c_lo, v_lo = lo, None
        c = mid
        while c > lo:
            c /= 2.0
            v = stat(c)
            if v <= target:
                c_lo, v_lo = c, v
                break
            c_hi, v_hi = c, v

    if v_lo is None or v_hi is None:
        return _result(history[-1], False)

    # 2. Log-log Illinois regula-falsi inside the bracket.
    converged = False
    stale_lo = stale_hi = 0
    log2 = float(np.log(2))
    for _ in range(cfg.n_root_iters):
        if v_lo > 0 and v_hi > 0:
            log_c_lo, log_c_hi = float(np.log(c_lo)), float(np.log(c_hi))
            log_v_lo = float(np.log(v_lo)) - (log2 if stale_lo >= 2 else 0.0)
            log_v_hi = float(np.log(v_hi)) - (log2 if stale_hi >= 2 else 0.0)
            t = (log_target - log_v_lo) / (log_v_hi - log_v_lo)
            log_c_new = log_c_lo + t * (log_c_hi - log_c_lo)
            c_new = float(np.exp(log_c_new))
            if not (c_lo < c_new < c_hi):  # bisection fallback
                c_new = float(np.sqrt(c_lo * c_hi))
        else:
            c_new = float(np.sqrt(c_lo * c_hi))

        v_new = stat(c_new)
        if abs(v_new - target) < cfg.convergence_tol:
            converged = True
            break
        if v_new < target:
            c_lo, v_lo = c_new, v_new
            stale_lo = 0
            stale_hi += 1
        else:
            c_hi, v_hi = c_new, v_new
            stale_hi = 0
            stale_lo += 1

    # If we exhausted iters without hitting tol, pick the closest point seen.
    if not converged:
        return _result(min(history, key=lambda h: abs(h["p95"] - target)), False)
    return _result(history[-1], True)


def main(cfg: KLCalibrateCfg) -> None:
    setup_logging("kl_calibrate")
    out_dir = cfg.out / cfg.behavior / "kl_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    calib_prompts, audit_prompts = _select_prompts(cfg.n_calib_prompts, cfg.n_audit_prompts, cfg.seed)
    logger.info(f"calibration prompts (n={len(calib_prompts)}): cats={[p.get('cat') for p in calib_prompts[:10]]}…")
    logger.info(f"audit prompts: n={len(audit_prompts)}")

    # Sanity-print one full prompt + greedy sample under base BEFORE any
    # method runs. This is the "did the chat template render correctly?" gate.
    p0 = calib_prompts[0]
    base_text = build_chat_text(tok, "", p0["user_msg"], "", thinking=True)
    base_ids = build_chat_ids(tok, "", p0["user_msg"], "", thinking=True)
    gen0, _ = greedy_generate_under_steering(
        model, tok, base_ids, method="base", alpha=0.0, n_new_tokens=cfg.n_tokens,
    )
    log_sample_prompt(tok, base_text, generated_ids=gen0,
                      label="format-check base (open <think>, no steering)")

    # 1. Target is the constant "side of the road" budget (gist: 1 nat).
    target = float(cfg.target_kl)
    logger.info(f"\ntarget p95 KL = {target:.4g} nats (constant; gist 'side of the road')")

    # Measure prompt baselines at α=1 for diagnostics — these are the
    # *uncalibrated* prompts (no continuous coefficient to scale), reported
    # alongside the calibrated adapter/repe results.
    logger.info(f"\n=== reference prompts (α=1, no calibration) ===")
    ref_method_names = [cfg.target_prompt, "simple_honest_prompt",
                        "engineered_prompt_dishonest", "simple_dishonest_prompt"]
    prompt_refs = {}
    for ji, name in enumerate(ref_method_names):
        if name not in PROMPT_TEXTS:
            continue
        m = _measure_kl_along_trajectory(
            f"prompt:{name}", alpha=1.0, model=model, tok=tok,
            prompts=calib_prompts, n_tokens=cfg.n_tokens,
            log_first_sample=(ji == 0),
            sample_label=f"reference prompt:{name} α=+1.000",
        )
        prompt_refs[f"prompt:{name}"] = m
        logger.info(f"  prompt:{name} p95={m['p95']:.4g} mean={m['mean']:.4g} max={m['max']:.4g}")

    # 2. Fit RepE directions once (used only if include_repe).
    repe_dirs = None
    if cfg.include_repe:
        logger.info("\n=== fit RepE directions ===")
        repe_dirs = fit_repe_directions(model, tok, cfg.n_repe_train, cfg.behavior)

    # 3. Illinois regula-falsi calibrate each adapter and (optionally) RepE.
    results_by_method: dict[str, dict[str, dict]] = {}
    for adapter in cfg.adapters:
        logger.info(f"\n=== calibrate dW:{adapter} ===")
        w = load_diff(cfg.out / cfg.behavior / adapter / DIFF_FILENAME)
        results_by_method[f"dW:{adapter}"] = {
            "pos": _illinois_calibrate(
                f"dW:{adapter}", target, model=model, tok=tok,
                prompts=calib_prompts, cfg=cfg, alpha_sign=1.0, sign_label="pos", w=w,
            ),
            "neg": _illinois_calibrate(
                f"dW:{adapter}", target, model=model, tok=tok,
                prompts=calib_prompts, cfg=cfg, alpha_sign=-1.0, sign_label="neg", w=w,
            ),
        }

    if cfg.include_repe:
        logger.info("\n=== calibrate repe ===")
        results_by_method["repe"] = {
            "pos": _illinois_calibrate(
                "repe", target, model=model, tok=tok,
                prompts=calib_prompts, cfg=cfg, alpha_sign=1.0, sign_label="pos", repe_dirs=repe_dirs,
            ),
            "neg": _illinois_calibrate(
                "repe", target, model=model, tok=tok,
                prompts=calib_prompts, cfg=cfg, alpha_sign=-1.0, sign_label="neg", repe_dirs=repe_dirs,
            ),
        }

    # 4. Audit: at calibrated α, recompute on n_audit prompts.
    logger.info(f"\n=== AUDIT (n={len(audit_prompts)} prompts) ===")

    audit_rows = []
    # Reference prompts: re-measure on audit set (no calibration; α=1).
    for name, m_calib in prompt_refs.items():
        m_audit = _measure_kl_along_trajectory(
            name, alpha=1.0, model=model, tok=tok,
            prompts=audit_prompts, n_tokens=cfg.n_tokens,
        )
        logger.info(f"  {name} α=+1 audit p95={m_audit['p95']:.4g} (calib was {m_calib['p95']:.4g})")
        audit_rows.append({
            "method": name,
            "alpha": 1.0,
            "p95_calib": m_calib["p95"],
            "mean_calib": m_calib["mean"],
            "p95_audit": m_audit["p95"],
            "mean_audit": m_audit["mean"],
            "max_audit": m_audit["max"],
            "calib_audit_ratio": m_audit["p95"] / m_calib["p95"] if m_calib["p95"] > 0 else float("nan"),
        })

    logger.info(
        "SHOULD: pos and neg p95 each match the target independently. "
        "Asymmetric alpha_pos/alpha_neg means the steering direction has asymmetric KL footprint, not failure."
    )
    for method, signs in results_by_method.items():
        if method.startswith("dW:"):
            adapter = method.split(":", 1)[1]
            w = load_diff(cfg.out / cfg.behavior / adapter / DIFF_FILENAME)
        else:
            w = None
        for sign_label, r in signs.items():
            alpha = r["calibrated_alpha"]
            if method.startswith("dW:"):
                m_audit = _measure_kl_along_trajectory(
                    method, alpha, model=model, tok=tok, prompts=audit_prompts,
                    n_tokens=cfg.n_tokens, w=w,
                )
            elif method == "repe":
                m_audit = _measure_kl_along_trajectory(
                    method, alpha, model=model, tok=tok, prompts=audit_prompts,
                    n_tokens=cfg.n_tokens, repe_dirs=repe_dirs,
                    repe_layers=cfg.repe_layers,
                )
            else:
                raise ValueError(method)
            logger.info(
                f"  {method}:{sign_label} α={alpha:+.3f} audit p95={m_audit['p95']:.4g} "
                f"(calib was {r['p95_at_calib']:.4g}, target {target:.4g})"
            )
            audit_rows.append({
                "method": method,
                "sign": sign_label,
                "alpha": alpha,
                "alpha_mag": r["alpha_mag"],
                "p95_calib": r["p95_at_calib"],
                "mean_calib": r["mean_at_calib"],
                "p95_audit": m_audit["p95"],
                "mean_audit": m_audit["mean"],
                "max_audit": m_audit["max"],
                "calib_audit_ratio": m_audit["p95"] / r["p95_at_calib"] if r["p95_at_calib"] > 0 else float("nan"),
            })

    audit_df = pl.DataFrame(audit_rows)
    audit_df.write_csv(out_dir / "audit.csv")

    summary_rows = []
    for method, signs in results_by_method.items():
        pos = signs["pos"]
        neg = signs["neg"]
        summary_rows.append({
            "method": method,
            "alpha_pos": pos["alpha_mag"],
            "alpha_neg": neg["alpha_mag"],
            "calibrated_alpha": pos["alpha_mag"],
            "p95_at_pos": pos["p95_at_calib"],
            "p95_at_neg": neg["p95_at_calib"],
            "mean_at_pos": pos["mean_at_calib"],
            "mean_at_neg": neg["mean_at_calib"],
            "max_at_pos": pos["max_at_calib"],
            "max_at_neg": neg["max_at_calib"],
            "ratio_at_pos": pos["ratio_at_calib"],
            "ratio_at_neg": neg["ratio_at_calib"],
            "iterations_pos": pos["iterations"],
            "iterations_neg": neg["iterations"],
            "converged_pos": pos["converged"],
            "converged_neg": neg["converged"],
        })
    summary_df = pl.DataFrame(summary_rows).sort("alpha_pos")
    summary_df = summary_df.with_columns(pl.lit(target).alias("target_p95"))
    summary_path = out_dir / "summary.csv"
    summary_df.write_csv(summary_path)

    history_rows = []
    for method, signs in results_by_method.items():
        for sign_label, r in signs.items():
            for h in r["history"]:
                history_rows.append({"method": method, "sign": sign_label, **h})
    pl.DataFrame(history_rows).write_csv(out_dir / "root_history.csv")

    pl.DataFrame([{"method": k, **v} for k, v in prompt_refs.items()]).write_csv(out_dir / "prompt_refs.csv")

    print("\n=== KL calibration summary (gist-faithful: greedy trajectory KL) ===")
    print(f"target p95 KL = {target:.4g} nats (constant; gist 'side of the road')")
    print(tabulate(summary_df.to_pandas(), headers="keys", tablefmt="tsv",
                   floatfmt="+.4g", showindex=False))
    print(f"\naudit (held-out {len(audit_prompts)} prompts):")
    print(tabulate(audit_df.to_pandas(), headers="keys", tablefmt="tsv",
                   floatfmt="+.4g", showindex=False))

    n_converged = sum(
        int(r["converged"])
        for signs in results_by_method.values()
        for r in signs.values()
    )
    n_total = sum(len(signs) for signs in results_by_method.values())
    cue = "🟢" if n_converged == n_total else "🟡"
    final_summary(
        out=summary_path,
        argv=get_argv(),
        main_metric=f"target_p95={target:.4g} converged={n_converged}/{n_total}",
        cue=cue,
        table_rows=summary_df.select(
            "method", "alpha_neg", "alpha_pos", "p95_at_neg", "p95_at_pos",
            "iterations_neg", "iterations_pos", "converged_neg", "converged_pos"
        ).rows(),
        headers=["method", "alpha_neg", "alpha_pos", "p95_neg", "p95_pos", "iters_neg", "iters_pos", "ok_neg", "ok_pos"],
        floatfmt="",
    )


if __name__ == "__main__":
    main(tyro.cli(KLCalibrateCfg))
