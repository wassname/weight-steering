"""Multi-seed Qwen adapter benchmark.

Runs the `fork_plan.md` stability check: seeds 0/1/2 for LoRA, PiSSA, and
DeLoRA, then reports sycophancy and daily-dilemmas deltas with seed-level signs.
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
from ws.diff import DIFF_FILENAME, compute_diff, load_base_state, load_delta, save_diff
from ws.eval.dilemmas import DilemmasCfg, evaluate as evaluate_dd
from ws.eval.sycophancy import EvalCfg, evaluate as evaluate_syc, summarize as summarize_syc
from ws.replicate import Cfg as ReplicateCfg
from ws.replicate import _maybe_data
from ws.train import TrainCfg, train_adapter


@dataclass
class MultiSeedBenchmarkCfg:
    model: str = "Qwen/Qwen3-0.6B"
    behavior: str = "sycophancy"
    adapters: tuple[str, ...] = ("lora", "pissa", "delora")
    seeds: tuple[int, ...] = (0, 1, 2)
    n_topics: int = 20
    n_personas: int = 5
    n_samples: int = 10
    rank: int = 32
    lr: float = 2e-4
    warmup_steps: int = 5
    epochs: float = 1.0
    max_steps: int = -1
    coeffs: tuple[float, ...] = (-1.0, 0.0, 1.0)
    n_dilemmas: int = 219
    batch_size: int = 8
    out: Path = Path("out")
    data_root: Path = Path("out/data")


def _model_slug(model: str) -> str:
    return model.replace("/", "__")


def _delta_at_one(summary: pl.DataFrame, value_col: str) -> float:
    zero = float(summary.filter(pl.col("coeff") == 0.0)[value_col][0])
    one = float(summary.filter(pl.col("coeff") == 1.0)[value_col][0])
    return one - zero


def _summarize_dd(df: pl.DataFrame) -> pl.DataFrame:
    summary = df.group_by("coeff").agg(
        pl.col("logratio_honesty").mean().alias("mean_logratio_honesty"),
        pl.col("logratio_honesty").std().alias("std_logratio_honesty"),
        pl.col("pmass").mean().alias("mean_pmass"),
        pl.col("low_pmass").mean().alias("frac_low_pmass"),
        pl.len().alias("n_rows"),
    ).sort("coeff")
    zero = float(summary.filter(pl.col("coeff") == 0.0)["mean_logratio_honesty"][0])
    return summary.with_columns(
        (pl.col("mean_logratio_honesty") - zero).alias("dd_delta_vs_0")
    )


def _run_one(cfg: MultiSeedBenchmarkCfg, adapter: str, seed: int, ds) -> dict:
    if cfg.max_steps > 0 and cfg.warmup_steps >= cfg.max_steps:
        raise ValueError(f"warmup_steps={cfg.warmup_steps} prevents learning with max_steps={cfg.max_steps}")
    seed_root = cfg.out / cfg.behavior / "multiseed" / _model_slug(cfg.model) / f"seed_{seed}"
    run_dir = seed_root / cfg.behavior / adapter
    paths = {}
    for sign in ("pos", "neg"):
        tcfg = TrainCfg(
            model_id=cfg.model,
            behavior=cfg.behavior,
            sign=sign,
            adapter=adapter,
            rank=cfg.rank,
            lr=cfg.lr,
            warmup_steps=cfg.warmup_steps,
            epochs=cfg.epochs,
            max_steps=cfg.max_steps,
            out=seed_root,
            seed=seed,
        )
        paths[sign] = train_adapter(tcfg, ds)
        torch.cuda.empty_cache()

    base = load_base_state(cfg.model)
    d_pos = load_delta(cfg.model, paths["pos"], base)
    d_neg = load_delta(cfg.model, paths["neg"], base)
    w = compute_diff(d_pos, d_neg)
    w_path = run_dir / DIFF_FILENAME
    save_diff(w, w_path)
    w_norm = float(sum((value.float().pow(2).sum() for value in w.values()), torch.tensor(0.0)).sqrt().item())
    if w_norm <= 0.0:
        raise ValueError(f"non-positive diff norm for adapter={adapter} seed={seed}")
    del base, d_pos, d_neg
    torch.cuda.empty_cache()

    syc_df = evaluate_syc(EvalCfg(model_id=cfg.model, coeffs=cfg.coeffs), w)
    syc_path = run_dir / "sycophancy_per_row.csv"
    syc_df.write_csv(syc_path)
    syc_summary = summarize_syc(syc_df)
    syc_summary_path = run_dir / "eval_summary.csv"
    syc_summary.write_csv(syc_summary_path)
    syc_delta = _delta_at_one(syc_summary, "mean_logratio")
    syc_pmass_min = float(syc_summary["mean_pmass"].min())

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    dd_df = evaluate_dd(
        DilemmasCfg(
            model_id=cfg.model,
            coeffs=cfg.coeffs,
            n_dilemmas=cfg.n_dilemmas,
            batch_size=cfg.batch_size,
        ),
        w,
        model=model,
        tok=tok,
    )
    dd_path = run_dir / "dd_per_row.csv"
    dd_df.write_csv(dd_path)
    dd_summary = _summarize_dd(dd_df)
    dd_summary_path = run_dir / "dd_summary.csv"
    dd_summary.write_csv(dd_summary_path)
    dd_delta = _delta_at_one(dd_summary, "mean_logratio_honesty")
    dd_pmass_min = float(dd_summary["mean_pmass"].min())
    del model
    torch.cuda.empty_cache()

    return {
        "adapter": adapter,
        "model": cfg.model,
        "seed": seed,
        "w_path": str(w_path),
        "w_exists": w_path.exists(),
        "w_norm": w_norm,
        "syc_summary_path": str(syc_summary_path),
        "syc_summary_exists": syc_summary_path.exists(),
        "dd_summary_path": str(dd_summary_path),
        "dd_summary_exists": dd_summary_path.exists(),
        "syc_delta": syc_delta,
        "dd_delta": dd_delta,
        "syc_pmass_min": syc_pmass_min,
        "dd_pmass_min": dd_pmass_min,
        "dd_rows_per_coeff": int(dd_summary["n_rows"].min()),
        "syc_sign": int(syc_delta > 0) - int(syc_delta < 0),
        "dd_sign": int(dd_delta > 0) - int(dd_delta < 0),
    }


def _ranking(per_seed: pl.DataFrame) -> pl.DataFrame:
    return per_seed.group_by(["model", "adapter"]).agg(
        pl.len().alias("n_seeds"),
        pl.col("w_path").n_unique().alias("n_w_files"),
        pl.col("w_exists").sum().alias("n_w_existing"),
        pl.col("w_norm").min().alias("min_w_norm"),
        pl.col("syc_summary_path").n_unique().alias("n_syc_summaries"),
        pl.col("syc_summary_exists").sum().alias("n_syc_existing"),
        pl.col("dd_summary_path").n_unique().alias("n_dd_summaries"),
        pl.col("dd_summary_exists").sum().alias("n_dd_existing"),
        pl.col("syc_delta").mean().alias("mean_syc_delta"),
        pl.col("syc_delta").std().alias("std_syc_delta"),
        pl.col("dd_delta").mean().alias("mean_dd_delta"),
        pl.col("dd_delta").std().alias("std_dd_delta"),
        (pl.col("dd_sign") == pl.col("dd_sign").mode().first()).mean().alias("sign_agreement"),
        pl.col("dd_rows_per_coeff").min().alias("min_dd_rows_per_coeff"),
        pl.col("dd_rows_per_coeff").max().alias("max_dd_rows_per_coeff"),
    ).sort(["model", "mean_dd_delta"], descending=[False, True])


def main(cfg: MultiSeedBenchmarkCfg) -> None:
    setup_logging("multiseed_benchmark")
    out_dir = cfg.out / cfg.behavior / "multiseed" / _model_slug(cfg.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    rcfg = ReplicateCfg(
        model=cfg.model,
        behavior=cfg.behavior,
        n_topics=cfg.n_topics,
        n_personas=cfg.n_personas,
        n_samples=cfg.n_samples,
        out=cfg.out,
        data_root=cfg.data_root,
    )
    ds = _maybe_data(rcfg)

    rows = []
    for adapter in cfg.adapters:
        for seed in cfg.seeds:
            logger.info(f"=== multiseed adapter={adapter} seed={seed} ===")
            rows.append(_run_one(cfg, adapter, seed, ds))

    per_seed = pl.DataFrame(rows)
    per_seed_path = out_dir / "per_seed.csv"
    per_seed.write_csv(per_seed_path)
    ranking = _ranking(per_seed)
    ranking_path = out_dir / "ranking.csv"
    ranking.write_csv(ranking_path)

    expected_rows = 2 * cfg.n_dilemmas
    bad = ranking.filter(
        (pl.col("n_seeds") != len(cfg.seeds))
        | (pl.col("n_w_files") != len(cfg.seeds))
        | (pl.col("n_w_existing") != len(cfg.seeds))
        | (pl.col("min_w_norm") <= 0.0)
        | (pl.col("n_syc_summaries") != len(cfg.seeds))
        | (pl.col("n_syc_existing") != len(cfg.seeds))
        | (pl.col("n_dd_summaries") != len(cfg.seeds))
        | (pl.col("n_dd_existing") != len(cfg.seeds))
        | (pl.col("min_dd_rows_per_coeff") != expected_rows)
        | (pl.col("max_dd_rows_per_coeff") != expected_rows)
    ).height

    print("\nmultiseed adapter ranking")
    print(
        f"SHOULD: each adapter has n_seeds=n_w_files=n_syc_summaries=n_dd_summaries={len(cfg.seeds)} "
        f"and DD rows per coeff={expected_rows}. ELSE run is incomplete or row subset changed."
    )
    print(tabulate(ranking.to_pandas(), headers="keys", tablefmt="tsv", floatfmt="+.3f", showindex=False))
    best = ranking.row(0, named=True)
    cue = "🟢" if bad == 0 else "🔴"
    final_summary(
        out=ranking_path,
        argv=get_argv(),
        main_metric=f"bad_adapters={bad}; best={best['adapter']} mean_dd_delta={best['mean_dd_delta']:+.3f}",
        cue=cue,
        table_rows=ranking.select(
            "model", "adapter", "n_seeds", "n_w_existing", "min_w_norm", "mean_syc_delta", "std_syc_delta", "mean_dd_delta", "std_dd_delta", "sign_agreement"
        ).rows(),
        headers=["model", "adapter", "n_seeds", "n_w_existing", "min_w_norm", "mean_syc_delta", "std_syc_delta", "mean_dd_delta", "std_dd_delta", "sign_agreement"],
        floatfmt="",
    )


if __name__ == "__main__":
    main(tyro.cli(MultiSeedBenchmarkCfg))