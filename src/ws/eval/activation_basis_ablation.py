"""Activation-basis ablation: SVD trained dW in the realized output-energy basis.

Hypothesis (H1 in nbs/ablation_analysis.py): own-SVD of `w_l` ranks output
directions by `sigma_i(w_l)` -- the operator norm under a *uniform* input
distribution. Real activations live on a low-dim manifold; the operator-norm
basis often misses it. So cropping by own-SVD throws away signal even when
the steering effect is genuinely low-rank in the basis that activations
actually populate.

Test: build the basis from *realized* output energy under DD-prompt activations.

For each trained tensor `w_l` of shape (d_out, d_in):

    Σ_x  = E_x [ x x^T ]                    # input cov on DD prompts (base model)
    C    = w_l Σ_x w_l^T                    # output-side cov under real x distribution
    C    = V Λ V^T                          # eigendecomp; sort λ descending
    V_k  = top-k columns by cumulative energy `target`
    w'_l = V_k V_k^T w_l                    # project rows onto top-k output dirs

Then re-run DD eval with `w'`. Drop test: `w_l - w'_l` (necessity-side).

Win condition: `top_25pct_act_keep` retained > 0.5 (vs ~0.1 in own-SVD lens).

Caveats (recorded for the analysis caveats list):
- Σ_x is collected on the same DD prompts used for eval. A positive result is
  still informative ("dW low-rank in eval-activation basis") but doesn't yet
  generalize to held-out activations. Split if H1 holds.
- Σ_x is from the base model (coeff=0). Activations under coeff=1 will differ;
  for small-coeff regime the base distribution is the right reference.
- Cropping shrinks Frobenius norm -> nonlinear-in-alpha caveat applies.
  `random_norm_matched_top_25pct_act` is the sufficiency-side anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
import tyro
from loguru import logger
from tabulate import tabulate
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding

from ws._log import final_summary, get_argv, setup_logging
from ws.diff import DIFF_FILENAME, load_diff
from ws.eval.dilemmas import DilemmasCfg, _load_eval, evaluate as evaluate_dd


@dataclass
class ActivationBasisCfg:
    model: str = "Qwen/Qwen3-0.6B"
    behavior: str = "sycophancy"
    adapter: str = "pissa"
    coeffs: tuple[float, ...] = (0.0, 1.0)
    n_dilemmas: int = 219
    n_calib_prompts: int = 64
    batch_size: int = 8
    out: Path = Path("out")
    diff_root: Path = Path("out")
    energy_targets: tuple[float, ...] = (0.25, 0.50)
    seed: int = 0
    max_tokens: int = 512


def _module_for_param(model, param_key: str):
    return model.get_submodule(param_key.removesuffix(".weight"))


def _collect_input_cov(
    model, tok, w_keys: list[str], cfg: ActivationBasisCfg
) -> dict[str, Tensor]:
    """Run base model on DD prompts; accumulate Σ_x = Σ_t x_t x_t^T per module (CPU float32).

    DD prompts are left-padded; attention_mask is used to skip pad-token activations.
    """
    sigma: dict[str, Tensor] = {}
    handles = []
    mask_holder: dict[str, Tensor | None] = {"mask": None}

    def make_hook(key: str):
        def hook(_module, inputs):
            x = inputs[0]
            if x.dim() == 3:
                _, _, D = x.shape
                x_flat = x.reshape(-1, D)
                mask = mask_holder["mask"]
                if mask is not None:
                    x_flat = x_flat[mask.bool().reshape(-1)]
            else:
                x_flat = x
            cov = (x_flat.float().T @ x_flat.float()).cpu()
            sigma[key] = cov if key not in sigma else sigma[key] + cov
        return hook

    for k in w_keys:
        mod = _module_for_param(model, k)
        handles.append(mod.register_forward_pre_hook(make_hook(k)))

    _, ds_pt, _ = _load_eval(tok, cfg.n_dilemmas, cfg.max_tokens, system_prompt="")
    n = min(cfg.n_calib_prompts, len(ds_pt))
    ds_pt = ds_pt.select(range(n))
    tok.padding_side = "left"
    collator = DataCollatorWithPadding(tok, return_tensors="pt")
    dl = DataLoader(ds_pt, batch_size=cfg.batch_size, collate_fn=collator, shuffle=False)

    try:
        with torch.no_grad():
            for batch in dl:
                ids = batch["input_ids"].to(model.device)
                mask = batch["attention_mask"].to(model.device) if "attention_mask" in batch else None
                mask_holder["mask"] = mask
                _ = model(input_ids=ids, attention_mask=mask)
        logger.info(f"collected Σ_x on {n} DD prompts for {len(sigma)} tensors")
    finally:
        for h in handles:
            h.remove()
        mask_holder["mask"] = None
    return sigma


def _act_basis_keep_drop(
    w: dict[str, Tensor], sigma: dict[str, Tensor], target: float
) -> tuple[dict[str, Tensor], dict[str, Tensor], float]:
    """Per-tensor: eigh(w Σ_x w^T), keep top-k by cumulative energy `target`.

    Returns (keep, drop, mean_k_frac) where mean_k_frac is the average rank
    fraction kept across tensors (sanity check that top-k is actually small).
    """
    keep: dict[str, Tensor] = {}
    drop: dict[str, Tensor] = {}
    k_fracs = []
    for key, value in w.items():
        if key not in sigma:
            raise ValueError(f"Σ_x missing for {key}")
        W = value.float().cpu()
        C = W @ sigma[key] @ W.T
        eigvals, eigvecs = torch.linalg.eigh(C)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order].clamp(min=0)
        eigvecs = eigvecs[:, order]
        total = float(eigvals.sum())
        if total <= 0:
            keep[key] = torch.zeros_like(value)
            drop[key] = value.clone()
            continue
        csum = torch.cumsum(eigvals, dim=0)
        k = int((csum < target * total).sum().item()) + 1
        V_k = eigvecs[:, :k]
        W_keep = (V_k @ (V_k.T @ W)).to(dtype=value.dtype)
        keep[key] = W_keep
        drop[key] = (value.cpu() - W_keep)
        k_fracs.append(k / V_k.shape[0])
    return keep, drop, sum(k_fracs) / max(len(k_fracs), 1)


def _frob(d: dict[str, Tensor]) -> float:
    return float(sum(v.float().pow(2).sum() for v in d.values()) ** 0.5)


def _random_norm_matched(target: dict[str, Tensor], seed: int) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in sorted(target.items()):
        n = torch.randn(v.shape, generator=g, dtype=torch.float32)
        nrm = v.float().norm()
        if float(nrm) > 0:
            n = n * (nrm / n.norm())
        out[k] = n.to(dtype=v.dtype)
    return out


def main(cfg: ActivationBasisCfg) -> None:
    setup_logging("activation_basis_ablation")
    out_dir = cfg.out / cfg.behavior / "activation_basis_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    w_full = load_diff(cfg.diff_root / cfg.behavior / cfg.adapter / DIFF_FILENAME)
    bad = [(k, tuple(v.shape)) for k, v in w_full.items() if v.dim() != 2]
    if bad:
        raise ValueError(f"activation-basis lens needs 2D tensors; non-2D found: {bad[:5]}")
    keys = sorted(w_full.keys())
    logger.info(f"loaded {cfg.adapter} dW: {len(keys)} 2D tensors, ||w||_F={_frob(w_full):.4g}")

    sigma = _collect_input_cov(model, tok, keys, cfg)

    variants = [
        {"component": "full_dW", "keep_or_drop": "full", "energy_target": 1.0, "w": w_full},
        {"component": "zero", "keep_or_drop": "zero", "energy_target": 0.0,
         "w": {k: torch.zeros_like(v) for k, v in w_full.items()}},
    ]

    keep_top25 = None
    for target in cfg.energy_targets:
        keep, drop, kfrac = _act_basis_keep_drop(w_full, sigma, target)
        pct = int(round(target * 100))
        logger.info(f"target={target}: mean kept rank fraction = {kfrac:.3f}")
        variants.append({"component": f"top_{pct}pct_act_keep", "keep_or_drop": "keep",
                         "energy_target": target, "w": keep})
        variants.append({"component": f"residual_not_top_{pct}pct_act", "keep_or_drop": "drop",
                         "energy_target": target, "w": drop})
        if target == 0.25:
            keep_top25 = keep

    if keep_top25 is not None:
        rnd = _random_norm_matched(keep_top25, seed=cfg.seed + 17)
        variants.append({"component": "random_norm_matched_top_25pct_act",
                         "keep_or_drop": "random", "energy_target": 0.25, "w": rnd})

    parts = []
    full_norm = _frob(w_full)
    for variant in variants:
        w_v = variant.pop("w")
        meta = {"adapter": cfg.adapter, **variant,
                "frob_frac": _frob(w_v) / full_norm if full_norm > 0 else 0.0}
        logger.info(f"eval component={meta['component']} frob_frac={meta['frob_frac']:.3f}")
        df = evaluate_dd(
            DilemmasCfg(model_id=cfg.model, coeffs=cfg.coeffs,
                        n_dilemmas=cfg.n_dilemmas, batch_size=cfg.batch_size),
            w_v, model=model, tok=tok,
        )
        df = df.with_columns(*(pl.lit(v).alias(k) for k, v in meta.items()))
        parts.append(df)

    dd = pl.concat(parts)

    grp = ["adapter", "component", "keep_or_drop", "energy_target", "frob_frac", "coeff"]
    sum_ = dd.group_by(grp).agg(
        pl.col("logratio_honesty").mean().alias("dd_mean"),
        pl.col("pmass").mean().alias("dd_pmass"),
        pl.len().alias("n_dd"),
    )
    base = sum_.filter((pl.col("component") == "full_dW") & (pl.col("coeff") == 0.0)).select(
        "adapter", pl.col("dd_mean").alias("dd_base")
    )
    summary = (
        sum_.join(base, on="adapter")
        .with_columns((pl.col("dd_mean") - pl.col("dd_base")).alias("dd_delta"))
        .sort(["component", "coeff"])
    )
    full_d_rows = summary.filter((pl.col("component") == "full_dW") & (pl.col("coeff") == 1.0))["dd_delta"]
    if full_d_rows.len() == 0:
        raise ValueError("missing full_dW @ coeff=1 row; cannot normalize")
    full_d = float(full_d_rows[0])
    if full_d == 0:
        raise ValueError("full_dW dd_delta is zero -- can't compute retained ratio")
    summary = summary.with_columns((pl.col("dd_delta") / full_d).alias("retained"))
    summary.write_csv(out_dir / "summary.csv")
    dd.write_csv(out_dir / "dd_per_row.csv")

    view = summary.filter(pl.col("coeff") == 1.0).sort("retained", descending=True)
    print("\nactivation-basis ablation (PiSSA, top-k of w Σ_x w^T)")
    print("SHOULD: top_25pct_act_keep retained > 0.5 if H1 (activation-basis) explains the puzzle; "
          "random_norm_matched_top_25pct_act near 0. ELSE H1 false, try input-side or look elsewhere.")
    print(tabulate(
        view.select("component", "keep_or_drop", "energy_target", "frob_frac", "dd_delta", "retained").to_pandas(),
        headers="keys", tablefmt="pipe", floatfmt="+.3f", showindex=False,
    ))

    top25_row = view.filter(pl.col("component") == "top_25pct_act_keep")
    top25_retained = float(top25_row["retained"][0]) if top25_row.height else float("nan")
    final_summary(
        out=out_dir / "summary.csv",
        argv=get_argv(),
        main_metric=f"top_25pct_act_keep_retained={top25_retained:+.3f} (>0.5 = H1 confirmed)",
        cue="🟢" if top25_retained > 0.5 else "🔴",
        table_rows=view.select(
            "component", "keep_or_drop", "energy_target", "frob_frac", "dd_delta", "retained"
        ).rows(),
        headers=["component", "kod", "energy", "frob_frac", "dd_delta", "retained"],
        floatfmt="",
    )


if __name__ == "__main__":
    main(tyro.cli(ActivationBasisCfg))
