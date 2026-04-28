"""DeLoRA per-tensor norm allocation vs within-tensor direction ablation.

Question: is the trained dW useful because of (a) its within-tensor
elementwise direction or (b) the per-tensor norm allocation (which
layers / modules get larger Frobenius-norm updates)? Each variant
preserves only one scalar per tensor (its Frobenius norm) or the full
tensor; within-tensor structure is either kept (full/dir_only) or
replaced by a single Gaussian draw (mag_only/random_norm). So this
isolates *per-tensor norm* vs *within-tensor direction*, not a broader
"magnitude pattern" notion. Variants:

  full         original dW (control)
  dir_only     dW with all tensors rescaled to a common Frobenius norm
               (preserves within-tensor direction; flattens per-tensor
               norm allocation)
  mag_only     each tensor replaced by a single Gaussian draw rescaled
               to the original tensor's Frobenius norm (preserves only
               the per-tensor norm scalar; within-tensor direction is
               random and seed-sensitive)
  random_norm  Gaussian random tensors all rescaled to a common norm
               (control: neither within-tensor direction nor per-tensor
               norm allocation)

mag_only and random_norm are single-seed Monte Carlo controls; rerun
across seeds before leaning on these conclusions.

Eval all four on daily-dilemmas (full 219 split) at coeffs {-1, 0, +1}
and dump dilemmas_per_row.csv so SI can be recomputed offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
import tyro
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from ws._log import final_summary, get_argv, setup_logging
from ws.diff import DIFF_FILENAME, load_diff
from ws.eval.dilemmas import DilemmasCfg, compute_full_metrics, evaluate


@dataclass
class DWDecompCfg:
    model: str = "Qwen/Qwen3-0.6B"
    behavior: str = "honesty"
    adapter: str = "delora"
    coeffs: tuple[float, ...] = (-1.0, 0.0, 1.0)
    n_dilemmas: int = 219
    batch_size: int = 8
    out: Path = Path("out")
    seed: int = 0


def _frob_norms(w: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(t.float().norm().item()) for k, t in w.items()}


def make_variants(w: dict[str, torch.Tensor], seed: int = 0) -> dict[str, dict[str, torch.Tensor]]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    norms = _frob_norms(w)
    n_tensors = len(w)
    total_sq = sum(n ** 2 for n in norms.values())
    common_norm = (total_sq / n_tensors) ** 0.5  # equal-norm such that sum-of-squares matches

    full = {k: t.clone() for k, t in w.items()}

    # dir_only: same elementwise direction, all tensors rescaled to common_norm
    dir_only = {}
    for k, t in w.items():
        n = norms[k]
        if n == 0:
            dir_only[k] = t.clone()
        else:
            dir_only[k] = (t.float() * (common_norm / n)).to(t.dtype)

    # mag_only: random direction, original per-tensor norm
    mag_only = {}
    for k, t in w.items():
        r = torch.randn(t.shape, generator=g, dtype=torch.float32)
        rn = float(r.norm().item())
        mag_only[k] = (r * (norms[k] / rn)).to(t.dtype)

    # random_norm: random direction, common per-tensor norm
    random_norm = {}
    for k, t in w.items():
        r = torch.randn(t.shape, generator=g, dtype=torch.float32)
        rn = float(r.norm().item())
        random_norm[k] = (r * (common_norm / rn)).to(t.dtype)

    return {"full": full, "dir_only": dir_only, "mag_only": mag_only, "random_norm": random_norm}


def main(cfg: DWDecompCfg) -> None:
    setup_logging("dw_decomp_ablation")
    out_dir = cfg.out / cfg.behavior / "dw_decomp_ablation" / cfg.adapter
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    w_path = cfg.out / cfg.behavior / cfg.adapter / DIFF_FILENAME
    w = load_diff(w_path)
    logger.info(f"loaded {len(w)} tensors from {w_path}; total ||dW||_F={sum(t.float().norm().item()**2 for t in w.values())**0.5:.3f}")
    variants = make_variants(w, seed=cfg.seed)

    parts = []
    dcfg = DilemmasCfg(
        model_id=cfg.model,
        coeffs=cfg.coeffs,
        n_dilemmas=cfg.n_dilemmas,
        batch_size=cfg.batch_size,
    )
    for name, ww in variants.items():
        logger.info(f"variant={name}: total ||W||_F={sum(t.float().norm().item()**2 for t in ww.values())**0.5:.3f}")
        df = evaluate(dcfg, ww, model=model, tok=tok).with_columns(pl.lit(name).alias("variant"))
        parts.append(df)

    per_row = pl.concat(parts)
    per_row_path = out_dir / "dilemmas_per_row.csv"
    per_row.write_csv(per_row_path)

    rows = []
    for name in variants:
        sub = per_row.filter(pl.col("variant") == name)
        m = compute_full_metrics(sub)
        zero = float(sub.filter(pl.col("coeff") == 0.0)["logratio_honesty"].mean())
        pos_mean = float(sub.filter(pl.col("coeff") == 1.0)["logratio_honesty"].mean())
        neg_mean = float(sub.filter(pl.col("coeff") == -1.0)["logratio_honesty"].mean())
        rows.append({
            "variant": name,
            "SI": m.get("surgical_informedness", float("nan")),
            "si_fwd": m.get("si_fwd", float("nan")),
            "si_rev": m.get("si_rev", float("nan")),
            "fix_fwd": m.get("fix_fwd", -1),
            "broke_fwd": m.get("broke_fwd", -1),
            "flip_rev": m.get("flip_rev", -1),
            "counter_rev": m.get("counter_rev", -1),
            "lr_at_zero": zero,
            "delta_pos": pos_mean - zero,
            "delta_neg": neg_mean - zero,
        })
    summary = pl.DataFrame(rows).sort("SI", descending=True, nulls_last=True)
    summary_path = out_dir / "summary.csv"
    summary.write_csv(summary_path)

    print("\nDeLoRA dW magnitude/direction ablation (honesty axis, daily dilemmas)")
    print(tabulate(summary.to_pandas(), headers="keys", tablefmt="tsv", floatfmt="+.3f", showindex=False))
    final_summary(
        out=summary_path,
        argv=get_argv(),
        main_metric=f"best_variant={summary['variant'][0]} SI={float(summary['SI'][0] or 0):+.3f}",
        cue="🟢",
        table_rows=summary.select("variant", "SI", "si_fwd", "si_rev", "delta_pos", "delta_neg").rows(),
        headers=["variant", "SI", "si_fwd", "si_rev", "delta_pos", "delta_neg"],
        floatfmt="",
    )


if __name__ == "__main__":
    main(tyro.cli(DWDecompCfg))
