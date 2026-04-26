"""Smoke: full pipeline on a tiny random model. CPU-feasible. ~1 min.

Set BEARTYPE=1 to enable jaxtyping runtime shape/dtype checks via the
jaxtyping import hook (autochars2 pattern). Catches dim errors early.

Pipeline exercised: data gen -> train pos -> train neg -> diff -> alpha sweep eval.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Install jaxtyping+beartype import hook BEFORE importing ws.* — this makes
# every Float[Tensor, "..."] annotation in ws/* a runtime check.
if os.environ.get("BEARTYPE"):
    from jaxtyping import install_import_hook
    # Returned manager auto-installs; keep ref alive for the process lifetime.
    _hook = install_import_hook("ws", "beartype.beartype")
    print("[smoke] BEARTYPE=1: jaxtyping runtime checks ENABLED for ws.*", flush=True)

import tyro
from dataclasses import dataclass

from ws.replicate import Cfg, main as replicate_main


@dataclass
class SmokeCfg:
    model: str = "katuni4ka/tiny-random-qwen3"  # or any tiny-random LM
    n_pairs: int = 4
    max_steps: int = 2
    out: Path = Path("out/smoke")
    adapter: str = "lora"


def main(cfg: SmokeCfg) -> None:
    print(f"[smoke] model={cfg.model} adapter={cfg.adapter} n_pairs={cfg.n_pairs} max_steps={cfg.max_steps}")
    rcfg = Cfg(
        model=cfg.model,
        behavior="sycophancy",
        adapter=cfg.adapter,
        n_pairs=cfg.n_pairs,
        max_steps=cfg.max_steps,
        out=cfg.out,
        smoke=False,  # we set knobs explicitly above
        coeffs=(-1.0, 0.0, 1.0),
        rank=4,  # tiny model, tiny rank
        n_topics=2,  # smoke: shrink data grid (paper recipe is 20×5)
        n_personas=1,
    )
    replicate_main(rcfg)
    print("[smoke] OK", flush=True)


if __name__ == "__main__":
    main(tyro.cli(SmokeCfg))
