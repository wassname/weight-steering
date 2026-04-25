"""Phase 1 entrypoint: data -> train pos -> train neg -> diff -> eval.

Usage:
    uv run python -m scripts.replicate --model Qwen/Qwen3-0.6B --behavior sycophancy --adapter lora
    uv run python -m scripts.replicate --smoke   # 32 pairs, 20 steps, ~5 min
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro
from datasets import Dataset
from loguru import logger
from tabulate import tabulate

from ws.data import DataCfg, generate_pairs, load_pairs
from ws.diff import compute_diff, load_base_state, load_delta, save_diff
from ws.eval.sycophancy import EvalCfg, evaluate, summarize
from ws.subspace import alignment_table, summarize_by_kind
from ws.train import TrainCfg, train_adapter


@dataclass
class Cfg:
    model: str = "Qwen/Qwen3-0.6B"
    behavior: str = "sycophancy"
    adapter: str = "lora"
    n_pairs: int = 1000
    rank: int = 16
    lr: float = 5e-5
    max_steps: int = -1
    out: Path = Path("out")
    smoke: bool = False
    coeffs: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _maybe_data(cfg: Cfg) -> Dataset:
    data_root = cfg.out / "data"
    try:
        ds = load_pairs(cfg.behavior, root=data_root)
        logger.info(f"reusing {len(ds)} pairs at {data_root / cfg.behavior}")
        return ds
    except (FileNotFoundError, Exception):
        pass
    dcfg = DataCfg(model_id=cfg.model, behavior=cfg.behavior, n_pairs=cfg.n_pairs, out=data_root)
    generate_pairs(dcfg)
    return load_pairs(cfg.behavior, root=data_root)


def main(cfg: Cfg) -> None:
    if cfg.smoke:
        cfg.n_pairs = 32
        cfg.max_steps = 20
        cfg.coeffs = (-1.0, 0.0, 1.0)

    ds = _maybe_data(cfg)

    # Train pos and neg.
    paths: dict[str, Path] = {}
    for sign in ("pos", "neg"):
        tcfg = TrainCfg(
            model_id=cfg.model, behavior=cfg.behavior, sign=sign,
            adapter=cfg.adapter, rank=cfg.rank, lr=cfg.lr,
            max_steps=cfg.max_steps, out=cfg.out,
        )
        paths[sign] = train_adapter(tcfg, ds)
        torch.cuda.empty_cache()

    # Diff in delta-W space.
    base = load_base_state(cfg.model)
    d_pos = load_delta(cfg.model, paths["pos"], base)
    d_neg = load_delta(cfg.model, paths["neg"], base)
    w = compute_diff(d_pos, d_neg)
    out_dir = cfg.out / cfg.behavior / cfg.adapter
    save_diff(w, out_dir / "w.pt")
    del d_pos, d_neg
    torch.cuda.empty_cache()

    # Phase 2: subspace alignment (uses base, then frees it).
    align_df = alignment_table(w, base)
    align_summary = summarize_by_kind(align_df)
    align_df.write_csv(out_dir / "subspace_per_layer.csv")
    align_summary.write_csv(out_dir / "subspace_summary.csv")
    del base
    torch.cuda.empty_cache()

    # Eval: sweep alpha.
    ecfg = EvalCfg(model_id=cfg.model, coeffs=cfg.coeffs)
    df = evaluate(ecfg, w)
    summary = summarize(df)

    print()
    print(f"# eval: {cfg.behavior} / {cfg.adapter} / {cfg.model}")
    print("# SHOULD: mean_logratio increases monotonically with coeff. ELSE diff is not steering.")
    print(tabulate(summary.to_pandas(), tablefmt="pipe", headers="keys", floatfmt="+.3f", showindex=False))
    summary.write_csv(out_dir / "eval_summary.csv")

    print()
    print(f"# subspace alignment: {cfg.behavior} / {cfg.adapter} / {cfg.model}")
    print("# SHOULD: ratio_top > 1 (top-SVD aligned) or ratio_weak > 1 (weak-readout writes).")
    print("# ELSE: w sits in random directions, no structural alignment.")
    print(tabulate(align_summary.to_pandas(), tablefmt="pipe", headers="keys", floatfmt="+.3f", showindex=False))


if __name__ == "__main__":
    main(tyro.cli(Cfg))
