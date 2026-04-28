"""Phase 1 entrypoint: data -> train pos -> train neg -> diff -> eval.

Usage:
    uv run python -m ws.replicate --model Qwen/Qwen3-0.6B --behavior sycophancy --adapter lora
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from datasets import Dataset
from loguru import logger
from tabulate import tabulate

from transformers import AutoTokenizer

from ws._log import final_summary, get_argv, setup_logging
from ws._tok_extras import has_thinking_mode
from ws.data import DataCfg, generate_pairs, load_pairs
from ws.diff import compute_diff, load_base_state, load_delta, save_diff
from ws.eval.sycophancy import EvalCfg, evaluate, summarize
from ws.run_demo import Cfg as DemoCfg, _demo_claims, phase_a1, phase_a2
from ws.subspace import alignment_table, summarize_by_kind
from ws.train import TrainCfg, train_adapter


@dataclass
class Cfg:
    model: str = "Qwen/Qwen3-0.6B"
    behavior: str = "sycophancy"
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

    print(f"\neval_summary {cfg.behavior}/{cfg.adapter}/{cfg.model}")
    print("SHOULD: mean_logratio monotone-increasing in coeff (more positive alpha => more Yes-mass on sycophantic claims), "
          "pmass~=1.0 across the sweep (Yes/No soak up next-token probability). "
          "Flat curve = diff not steering, retrain longer or check sign convention. "
          "pmass < 0.95 at alpha=0 = format broken, choice-id extraction wrong. "
          "Caveat: this is single-token off-policy. Compare to phase_a2 margin to detect teacher-forcing gap.")
    print(tabulate(summary.to_pandas(), tablefmt="tsv", headers="keys", floatfmt="+.3f", showindex=False))
    summary.write_csv(out_dir / "eval_summary.csv")

    print(f"\nsubspace_alignment {cfg.behavior}/{cfg.adapter}/{cfg.model}")
    print("SHOULD (priors from AntiPaSTO steering_methods.qmd:340): SVD-of-W test is known to be ~uninformative "
          "for task differences (~0.08 cosine). Expect mean_ratio_top ~= 1.0 across kinds; this is *not* a falsification. "
          "ratio_weak > 1 (weak-readout writes) is the more meaningful signal here — it's the Logits_Null primitive. "
          "ratio_weak >> 1 = w writes into directions the unembed ignores (stenographic-shaped). "
          "Real task-aware tests (TaskDiff/Suppressed/Stenographic) are phase 2.5.")
    print(tabulate(align_summary.to_pandas(), tablefmt="tsv", headers="keys", floatfmt="+.3f", showindex=False))

    # Phase A demo: on-policy coherence + guided CoT under w. Catches incoherent
    # adapters and the teacher-forcing gap (off-policy logratio inflated vs rollout).
    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dcfg = DemoCfg(model=cfg.model, behavior=cfg.behavior, adapter=cfg.adapter, out=cfg.out)
    claims = _demo_claims(dcfg.ood_claim)
    phase_a1(dcfg, claims, tok)
    if has_thinking_mode(tok):
        demo_df = phase_a2(dcfg, claims, tok)
        demo_df.write_csv(out_dir / "demo_guided_cot.csv")
    else:
        logger.info("skipping guided-CoT demo: model has no </think> special token")

    # BLUF: headline = max margin across alpha sweep on in_dist claim
    sp = summary.to_pandas()
    # mean_logratio at largest positive coeff
    top = sp.sort_values("coeff").iloc[-1]
    bot = sp.sort_values("coeff").iloc[0]
    spread = float(top["mean_logratio"]) - float(bot["mean_logratio"])
    pmin = float(sp["mean_pmass"].min()) if "mean_pmass" in sp.columns else float("nan")
    cue = "🟢" if (spread > 1.0 and pmin > 0.95) else ("🟡" if spread > 0.3 else "🔴")
    final_summary(
        out=out_dir / "eval_summary.csv",
        argv=get_argv(),
        main_metric=f"logratio_spread={spread:+.3f} pmass_min={pmin:.3f}",
        cue=cue,
        table_rows=[[
            f"{spread:+.3f}", f"{pmin:.3f}",
            f"{float(top['coeff']):+.1f}", f"{float(top['mean_logratio']):+.3f}",
            cfg.behavior, cfg.adapter, cfg.model,
            f"r{cfg.rank},lr{cfg.lr},ep{cfg.epochs}",
            str(out_dir / "eval_summary.csv"),
        ]],
        headers=["logratio_spread", "pmass_min", "coeff_top", "logratio_top",
                 "behavior", "adapter", "model", "flags", "out"],
        floatfmt="",
    )


if __name__ == "__main__":
    main(tyro.cli(Cfg))
