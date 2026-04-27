"""From-scratch weight steering candidates built without adapter deltas.

The goal is stricter than decomposing a trained `dW`: construct a weight-space
intervention from base-model weights and persona-contrast activations alone,
then compare it to the trained adapter `dW` on identical sycophancy and DD rows.

Current candidate: for every residual-write matrix (`o_proj`, `down_proj`), write
along the RepE persona direction at that layer and gate by a base-weight SVD input
axis. This is a rank-1 update:

    dW'_l = u_persona_l[:, None] @ v_base_l[None, :]

where `u_persona_l` is fit from positive-vs-negative persona residual activations
and `v_base_l` is a right singular vector of the unmodified base weight. A random
input axis is included as the null with identical output direction and norm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
import tyro
from loguru import logger
from tabulate import tabulate
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from ws._log import final_summary, get_argv, setup_logging
from ws.diff import DIFF_FILENAME, load_diff
from ws.eval.activation_baseline import _fit_repe_directions
from ws.eval.dilemmas import DilemmasCfg, evaluate as evaluate_dilemmas
from ws.eval.sycophancy import EvalCfg, evaluate as evaluate_sycophancy

_RESID_WRITE_RE = re.compile(r"model\.layers\.(\d+)\.(self_attn\.o_proj|mlp\.down_proj)\.weight$")


@dataclass
class FromScratchSteeringCfg:
    model: str = "Qwen/Qwen3-0.6B"
    behavior: str = "sycophancy"
    trained_adapter: str = "delora"
    out: Path = Path("out")
    diff_root: Path = Path("out")
    coeffs: tuple[float, ...] = (-1.0, 0.0, 1.0)
    n_dilemmas: int = 219
    batch_size: int = 8
    n_train_topics: int = 20
    n_eval_topics: int = 12
    tensor_norm_frac: float = 1e-3
    random_seed: int = 0


def _right_singular_axis(W: Tensor, mode: str) -> Tensor:
    _U, _S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    if mode == "top":
        return Vh[0]
    if mode == "tail":
        return Vh[-1]
    raise ValueError(f"unknown singular axis mode: {mode}")


def _random_axis(n: int, *, seed: int) -> Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    v = torch.randn(n, generator=gen)
    return v / v.norm()


def _rank1_write(u_out: Tensor, v_in: Tensor, target_norm: Tensor, dtype: torch.dtype) -> Tensor:
    u = u_out.float() / u_out.float().norm()
    v = v_in.float() / v_in.float().norm()
    dw = torch.outer(u, v)
    dw = dw * target_norm.float()
    return dw.to(dtype=dtype, device="cpu")


def _construct_candidates(model, directions: Tensor, cfg: FromScratchSteeringCfg) -> dict[str, dict[str, Tensor]]:
    candidates: dict[str, dict[str, Tensor]] = {
        "persona_write_top_svd": {},
        "persona_write_tail_svd": {},
        "persona_write_random": {},
    }
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    for name, W in state.items():
        match = _RESID_WRITE_RE.search(name)
        if match is None or W.dim() != 2:
            continue
        layer = int(match.group(1))
        if layer >= directions.shape[0] or W.shape[0] != directions.shape[1]:
            raise ValueError(f"residual-write shape mismatch for {name}: W={tuple(W.shape)} dir={tuple(directions.shape)}")

        target_norm = W.float().norm() * cfg.tensor_norm_frac
        u_out = directions[layer]
        candidates["persona_write_top_svd"][name] = _rank1_write(
            u_out, _right_singular_axis(W, "top"), target_norm, W.dtype
        )
        candidates["persona_write_tail_svd"][name] = _rank1_write(
            u_out, _right_singular_axis(W, "tail"), target_norm, W.dtype
        )
        candidates["persona_write_random"][name] = _rank1_write(
            u_out, _random_axis(W.shape[1], seed=cfg.random_seed + layer), target_norm, W.dtype
        )

    for method, w in candidates.items():
        if not w:
            raise ValueError(f"candidate {method} has zero tensors; residual-write regex missed the model")
        norm = sum((dw.float() ** 2).sum() for dw in w.values()).sqrt().item()
        logger.info(f"constructed {method}: {len(w)} tensors, ||dW'||={norm:.4g}")
    return candidates


def _norm_table(candidates: dict[str, dict[str, Tensor]]) -> pl.DataFrame:
    rows = []
    for method, w in candidates.items():
        rows.append({
            "method": method,
            "n_tensors": len(w),
            "n_params": sum(dw.numel() for dw in w.values()),
            "norm": float(sum((dw.float() ** 2).sum() for dw in w.values()).sqrt().item()),
        })
    return pl.DataFrame(rows).sort("method")


def _eval_method(method: str, w: dict[str, Tensor], cfg: FromScratchSteeringCfg) -> tuple[pl.DataFrame, pl.DataFrame]:
    syc = evaluate_sycophancy(
        EvalCfg(model_id=cfg.model, coeffs=cfg.coeffs, n_held_out=cfg.n_eval_topics), w
    ).with_columns(pl.lit(method).alias("method"))
    dd = evaluate_dilemmas(
        DilemmasCfg(
            model_id=cfg.model,
            coeffs=cfg.coeffs,
            n_dilemmas=cfg.n_dilemmas,
            batch_size=cfg.batch_size,
        ),
        w,
    ).with_columns(pl.lit(method).alias("method"))
    return syc, dd


def _summary(syc: pl.DataFrame, dd: pl.DataFrame) -> pl.DataFrame:
    syc_summary = syc.group_by(["method", "coeff"]).agg(
        pl.col("logratio").mean().alias("syc_mean"),
        pl.col("pmass").mean().alias("syc_pmass"),
        pl.len().alias("n_syc"),
    )
    syc_zero = syc_summary.filter(pl.col("coeff") == 0.0).select(
        "method", pl.col("syc_mean").alias("syc_zero")
    )
    syc_summary = syc_summary.join(syc_zero, on="method", how="left").with_columns(
        (pl.col("syc_mean") - pl.col("syc_zero")).alias("syc_delta")
    )

    dd_summary = dd.group_by(["method", "coeff"]).agg(
        pl.col("logratio_honesty").mean().alias("dd_mean"),
        pl.col("pmass").mean().alias("dd_pmass"),
        pl.col("low_pmass").mean().alias("dd_frac_low_pmass"),
        pl.len().alias("n_dd"),
    )
    dd_zero = dd_summary.filter(pl.col("coeff") == 0.0).select(
        "method", pl.col("dd_mean").alias("dd_zero")
    )
    dd_summary = dd_summary.join(dd_zero, on="method", how="left").with_columns(
        (pl.col("dd_mean") - pl.col("dd_zero")).alias("dd_delta"),
        pl.col("n_dd").alias("n_base_rows_per_coeff"),
    )
    return syc_summary.join(dd_summary, on=["method", "coeff"], how="inner").sort(["method", "coeff"])


def _idx_symmetric_diff(dd: pl.DataFrame) -> int:
    trained_idx = set(
        dd.filter(pl.col("method") == "trained_dW")
        .select("idx", "dilemma_idx", "action_type")
        .iter_rows()
    )
    max_diff = 0
    for row in dd.select("method", "coeff").unique().iter_rows(named=True):
        idx = set(
            dd.filter((pl.col("method") == row["method"]) & (pl.col("coeff") == row["coeff"]))
            .select("idx", "dilemma_idx", "action_type")
            .iter_rows()
        )
        max_diff = max(max_diff, len(trained_idx.symmetric_difference(idx)))
    return max_diff


def _claim_idx_symmetric_diff(syc: pl.DataFrame) -> int:
    trained_idx = set(syc.filter(pl.col("method") == "trained_dW")["claim_idx"].to_list())
    max_diff = 0
    for row in syc.select("method", "coeff").unique().iter_rows(named=True):
        idx = set(
            syc.filter((pl.col("method") == row["method"]) & (pl.col("coeff") == row["coeff"]))[
                "claim_idx"
            ].to_list()
        )
        max_diff = max(max_diff, len(trained_idx.symmetric_difference(idx)))
    return max_diff


def main(cfg: FromScratchSteeringCfg) -> None:
    setup_logging("from_scratch_steering")
    out_dir = cfg.out / cfg.behavior / "from_scratch_steering"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    directions = _fit_repe_directions(model, tok, cfg.n_train_topics)
    candidates = _construct_candidates(model, directions, cfg)
    norm_df = _norm_table(candidates).with_columns(pl.lit(True).alias("constructed_before_trained_diff_load"))
    if norm_df.filter(pl.col("norm") <= 0.0).height:
        raise ValueError("constructed candidate has non-positive norm")
    norm_path = out_dir / "candidate_norms.csv"
    norm_df.write_csv(norm_path)
    del model

    syc_parts = []
    dd_parts = []
    for method, w in candidates.items():
        syc, dd = _eval_method(method, w, cfg)
        syc_parts.append(syc)
        dd_parts.append(dd)

    trained_w = load_diff(cfg.diff_root / cfg.behavior / cfg.trained_adapter / DIFF_FILENAME)
    syc, dd = _eval_method("trained_dW", trained_w, cfg)
    syc_parts.append(syc)
    dd_parts.append(dd)

    syc_all = pl.concat(syc_parts)
    dd_all = pl.concat(dd_parts)
    syc_path = out_dir / "sycophancy_per_row.csv"
    dd_path = out_dir / "dilemmas_per_row.csv"
    syc_all.write_csv(syc_path)
    dd_all.write_csv(dd_path)

    idx_diff = _idx_symmetric_diff(dd_all)
    syc_idx_diff = _claim_idx_symmetric_diff(syc_all)
    expected_rows = 2 * cfg.n_dilemmas
    summary = _summary(syc_all, dd_all).with_columns(
        pl.lit(idx_diff).alias("idx_symmetric_diff"),
        pl.lit(syc_idx_diff).alias("syc_claim_idx_symmetric_diff"),
        (pl.col("n_base_rows_per_coeff") == expected_rows).alias("row_count_ok"),
    )
    summary_path = out_dir / "summary.csv"
    summary.write_csv(summary_path)

    best = summary.sort("dd_delta", descending=True).head(12)
    print("\nfrom-scratch steering summary")
    print("SHOULD: constructed_before_trained_diff_load=True; idx_symmetric_diff=0; full run rows=438. ELSE candidate used trained dW or row mismatch.")
    print(tabulate(best.to_pandas(), tablefmt="tsv", headers="keys", floatfmt="+.3f", showindex=False))
    bad_rows = summary.filter(~pl.col("row_count_ok")).height
    cue = "🟢" if idx_diff == 0 and syc_idx_diff == 0 and bad_rows == 0 else "🔴"
    final_summary(
        out=summary_path,
        argv=get_argv(),
        main_metric=f"idx_symmetric_diff={idx_diff}; syc_claim_idx_symmetric_diff={syc_idx_diff}; bad_row_count_groups={bad_rows}; best_dd_delta={float(best['dd_delta'][0]):+.3f}",
        cue=cue,
        table_rows=best.select("method", "coeff", "syc_delta", "dd_delta", "n_base_rows_per_coeff", "idx_symmetric_diff", "syc_claim_idx_symmetric_diff", "row_count_ok").rows(),
        headers=["method", "coeff", "syc_delta", "dd_delta", "rows_per_coeff", "idx_diff", "syc_idx_diff", "rows_ok"],
        floatfmt="",
    )


if __name__ == "__main__":
    main(tyro.cli(FromScratchSteeringCfg))
