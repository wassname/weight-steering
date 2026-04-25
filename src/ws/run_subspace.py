"""Phase 2 entrypoint: project w onto SVD + weak-readout subspaces, print alignment table.

Reads a precomputed diff (out/<behavior>/<adapter>/w.pt) and the base model state
dict, computes per-layer alignment ratios, and prints a tabulated summary by
param-kind (q_proj / o_proj / down_proj / ...).

A ratio_top > 1 ⇒ steering signal concentrates in W's principal SVD components.
A ratio_weak > 1 ⇒ steering writes into directions the unembedding reads weakly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro
from loguru import logger
from tabulate import tabulate

from ws.diff import load_base_state, load_diff
from ws.subspace import alignment_table, summarize_by_kind


@dataclass
class Cfg:
    model: str = "Qwen/Qwen3-0.6B"
    behavior: str = "sycophancy"
    adapter: str = "lora"
    out: Path = Path("out")
    k_frac: float = 0.1
    weak_frac: float = 0.01


def main(cfg: Cfg) -> None:
    diff_path = cfg.out / cfg.behavior / cfg.adapter / "w.pt"
    if not diff_path.exists():
        raise FileNotFoundError(f"no diff at {diff_path}; run replicate first")

    w = load_diff(diff_path)
    base = load_base_state(cfg.model)
    logger.info(f"loaded {len(w)} touched params; base has {len(base)} keys")

    df = alignment_table(w, base, k_frac=cfg.k_frac, weak_frac=cfg.weak_frac)
    summary = summarize_by_kind(df)

    out_dir = cfg.out / cfg.behavior / cfg.adapter
    df.write_csv(out_dir / "subspace_per_layer.csv")
    summary.write_csv(out_dir / "subspace_summary.csv")

    print()
    print(f"# subspace alignment: {cfg.behavior} / {cfg.adapter} / {cfg.model}")
    print(
        f"# k_frac={cfg.k_frac} (top SVD), weak_frac={cfg.weak_frac} (bottom of lm_head)"
    )
    print(
        "# SHOULD: ratio_top > 1 (PiSSA-aligned) or ratio_weak > 1 (writes into weak-readout)."
    )
    print("# ELSE: w is no more aligned than a random direction in the same space.")
    print(
        tabulate(
            summary.to_pandas(),
            tablefmt="pipe",
            headers="keys",
            floatfmt="+.3f",
            showindex=False,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Cfg))
