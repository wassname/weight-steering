"""Phase 1 entrypoint: data -> train pos -> train neg -> diff.

Usage:
    uv run python -m ws.replicate --model Qwen/Qwen3-0.6B --behavior authority --adapter lora
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from datasets import Dataset
from loguru import logger

from ws._log import final_summary, get_argv, setup_logging
from ws.data import DataCfg, generate_pairs, load_pairs
from ws.diff import compute_diff, load_base_state, load_delta, save_diff
from ws.train import TrainCfg, train_adapter


@dataclass
class Cfg:
    model: str = "Qwen/Qwen3-0.6B"
    behavior: str = "authority"
    adapter: str = "lora"
    # Data grid (paper recipe: 20 × 5 × 10 = 1000). Smoke shrinks via CLI.
    n_topics: int = 20
    n_personas: int = 5
    n_samples: int = 10
    rank: int = 32
    lr: float = 2e-4
    epochs: float = 1.0
    max_steps: int = -1
    out: Path = Path("out")
    # Shared data dir — kept separate from out so multiple runs reuse the same pairs.
    data_root: Path = Path("out/data")
    data_batch_size: int = 8
    data_min_new_tokens: int = 1024
    data_max_new_tokens: int = 1280
    data_temperature: float | None = None
    data_top_p: float | None = None
    data_top_k: int | None = None
    data_min_p: float | None = None
    data_presence_penalty: float = 0.0
    coeffs: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _maybe_data(cfg: Cfg) -> Dataset:
    from ws.data import _personas
    data_root = cfg.data_root
    behavior_dir = data_root / cfg.behavior
    sys_pos_all, _ = _personas(cfg.behavior)
    n_personas = min(cfg.n_personas, len(sys_pos_all))
    expected = cfg.n_topics * n_personas * cfg.n_samples
    if behavior_dir.exists():
        ds = load_pairs(cfg.behavior, root=data_root)
        if len(ds) != expected:
            raise ValueError(
                f"on-disk data at {behavior_dir} has {len(ds)} pairs but "
                f"grid {cfg.n_topics}×{n_personas}×{cfg.n_samples}={expected}. "
                f"Delete the dir to regenerate."
            )
        logger.info(f"reusing {len(ds)} pairs at {behavior_dir}")
        return ds
    dcfg = DataCfg(
        model_id=cfg.model, behavior=cfg.behavior, out=data_root,
        n_topics=cfg.n_topics, n_personas=cfg.n_personas, n_samples=cfg.n_samples,
        batch_size=cfg.data_batch_size,
        min_new_tokens=cfg.data_min_new_tokens,
        max_new_tokens=cfg.data_max_new_tokens,
        temperature=cfg.data_temperature,
        top_p=cfg.data_top_p,
        top_k=cfg.data_top_k,
        min_p=cfg.data_min_p,
        presence_penalty=cfg.data_presence_penalty,
    )
    generate_pairs(dcfg)
    return load_pairs(cfg.behavior, root=data_root)


def main(cfg: Cfg) -> None:
    setup_logging("replicate")
    ds = _maybe_data(cfg)

    # Train pos and neg.
    paths: dict[str, Path] = {}
    for sign in ("pos", "neg"):
        tcfg = TrainCfg(
            model_id=cfg.model, behavior=cfg.behavior, sign=sign,
            adapter=cfg.adapter, rank=cfg.rank, lr=cfg.lr,
            epochs=cfg.epochs, max_steps=cfg.max_steps, out=cfg.out,
        )
        paths[sign] = train_adapter(tcfg, ds)
        torch.cuda.empty_cache()

    # Diff in delta-W space.
    base = load_base_state(cfg.model)
    d_pos = load_delta(cfg.model, paths["pos"], base)
    d_neg = load_delta(cfg.model, paths["neg"], base)
    w = compute_diff(d_pos, d_neg)
    out_dir = cfg.out / cfg.behavior / cfg.adapter
    out_dir.mkdir(parents=True, exist_ok=True)
    save_diff(w, out_dir / "w.pt")

    final_summary(
        out=out_dir / "w.pt",
        argv=get_argv(),
        main_metric=f"diff saved behavior={cfg.behavior} adapter={cfg.adapter}",
        cue="🟢",
        table_rows=[[cfg.behavior, cfg.adapter, cfg.model, str(out_dir / "w.pt")]],
        headers=["behavior", "adapter", "model", "out"],
        floatfmt="",
    )


if __name__ == "__main__":
    main(tyro.cli(Cfg))
