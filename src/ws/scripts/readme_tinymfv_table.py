"""README-ready tiny-mfv table: ws adapters + steering-lite baselines side-by-side.

Same axis (Care vs Traditional/Sanctity), same metric (axis_shift in nats),
same paired-by-(vid,cond) per-foundation Δlogit. ws rows are read from
`out/trad_care/{adapter|base}/*__foundations_dlogit.csv` (the eval already
computes them); steering-lite rows are read from
`/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite/outputs/tinymfv_sweep/*.json`.

NB: ws weight-steering uses iso-KL calibrated alpha (target_kl=1.0 nat); the
steering-lite calibration is the same target. Both repos' rows are therefore
at matched KL footprint, so axis_shift is directly comparable. The ws
prompt_only row (alpha=+1, no calibration) and steering-lite's prompt_only
row are the only un-calibrated points -- they're included for context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import tyro
from tabulate import tabulate

from ws._artifacts import latest_matching


FOUNDATION_ORDER = ["Care", "Sanctity", "Authority", "Loyalty", "Fairness", "Liberty", "Social Norms"]
FOUNDATION_SHORT = {
    "Care": "Care", "Sanctity": "Sanc", "Authority": "Auth",
    "Loyalty": "Loy", "Fairness": "Fair", "Liberty": "Lib", "Social Norms": "SocN",
}


@dataclass
class ReadmeTinymfvCfg:
    behavior: str = "trad_care"
    out: Path = Path("out")
    adapters: tuple[str, ...] = ("lora", "dora", "pissa", "delora", "oft", "ia3")
    include_base: bool = True
    include_prompt_baseline: bool = True
    steering_lite_root: Path = Path("/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite")
    steering_lite_methods: tuple[str, ...] = (
        "bare", "prompt_only", "mean_diff", "mean_centred",
        "pca", "sspace", "cosine_gated", "topk_clusters",
    )


def _cue(axis: float) -> str:
    if axis != axis:  # NaN
        return "⚪"
    a = abs(axis)
    if a > 0.5:
        return "🟢"
    if a > 0.15:
        return "🟡"
    return "🔴"


def _load_ws_row(cfg: ReadmeTinymfvCfg, adapter_dir: Path, label: str, alpha: float = 1.0) -> dict | None:
    """Read latest eval artefacts in `adapter_dir`; return one row dict or None."""
    try:
        summary_path = latest_matching(adapter_dir, "*__summary.csv")
        dlogit_path = latest_matching(adapter_dir, "*__foundations_dlogit.csv")
    except FileNotFoundError:
        return None
    summary = pl.read_csv(summary_path)
    dlogit = pl.read_csv(dlogit_path)
    sub = summary.filter(pl.col("alpha") == alpha)
    if sub.is_empty():
        return None
    axis = float(sub["axis_shift"][0])
    sub_d = dlogit.filter(pl.col("alpha") == alpha)
    by_f = {row["foundation_coarse"]: row["dlogit_mean"] for row in sub_d.to_dicts()}
    row = {"row": label, "axis_shift": axis, "cue": _cue(axis), "n_vig": int(sub["n_vignettes"][0])}
    for f in FOUNDATION_ORDER:
        row[FOUNDATION_SHORT[f]] = by_f.get(f, float("nan"))
    return row


def _load_steering_lite_row(json_path: Path) -> dict | None:
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    method = data.get("method", json_path.stem)
    label = f"sl:{method}"
    if "axis_shift" in data and "dlogit_per_foundation" in data:
        axis = float(data["axis_shift"])
        dlf = data["dlogit_per_foundation"]
        row = {"row": label, "axis_shift": axis, "cue": _cue(axis),
               "n_vig": sum(int(d.get("n", 0)) for d in dlf.values()) // max(1, len(dlf))}
        for f in FOUNDATION_ORDER:
            row[FOUNDATION_SHORT[f]] = dlf.get(f, {}).get("mean", float("nan"))
        return row
    # bare.json has absolute_logit_per_foundation, no Δ
    if "absolute_logit_per_foundation" in data:
        alf = data["absolute_logit_per_foundation"]
        row = {"row": f"sl:{method} (abs logit)", "axis_shift": float("nan"), "cue": "⚪",
               "n_vig": sum(int(d.get("n", 0)) for d in alf.values()) // max(1, len(alf))}
        for f in FOUNDATION_ORDER:
            row[FOUNDATION_SHORT[f]] = alf.get(f, {}).get("mean", float("nan"))
        return row
    return None


def main(cfg: ReadmeTinymfvCfg) -> None:
    rows: list[dict] = []

    # ws bare row (alpha=0 absolute, no steering) -- read from any adapter's
    # alpha=0 row in the foundation CSV. axis_shift is NaN at alpha=0 by
    # construction (Δ vs itself = 0); we just want the model's prior.
    if cfg.include_base:
        for adapter in cfg.adapters:
            d = cfg.out / cfg.behavior / adapter
            if not d.exists():
                continue
            try:
                fpath = latest_matching(d, "*__foundations.csv")
            except FileNotFoundError:
                continue
            fdf = pl.read_csv(fpath).filter(pl.col("alpha") == 0.0)
            if fdf.is_empty():
                continue
            by_f = {r["foundation_coarse"]: r["wrongness_logit"]
                    for r in fdf.to_dicts() if "wrongness_logit" in r}
            if not by_f:
                # fallback: use mean wrongness column
                by_f = {r["foundation_coarse"]: r.get("wrongness", float("nan"))
                        for r in fdf.to_dicts()}
            row = {"row": "ws:bare (abs logit)", "axis_shift": float("nan"),
                   "cue": "⚪", "n_vig": int(fdf["n_vignettes"].sum()) if "n_vignettes" in fdf.columns else 0}
            for f in FOUNDATION_ORDER:
                row[FOUNDATION_SHORT[f]] = by_f.get(f, float("nan"))
            rows.append(row)
            break

    # ws prompt-only baseline (out/<behavior>/base/...)
    if cfg.include_prompt_baseline:
        base_dir = cfg.out / cfg.behavior / "base"
        if base_dir.exists():
            row = _load_ws_row(cfg, base_dir, "ws:prompt_only", alpha=1.0)
            if row is not None:
                rows.append(row)

    # ws adapters
    for adapter in cfg.adapters:
        d = cfg.out / cfg.behavior / adapter
        if not d.exists():
            continue
        row = _load_ws_row(cfg, d, f"ws:{adapter}", alpha=1.0)
        if row is not None:
            rows.append(row)

    # steering-lite rows (frozen baselines)
    for method in cfg.steering_lite_methods:
        json_path = cfg.steering_lite_root / "outputs" / "tinymfv_sweep" / f"{method}.json"
        row = _load_steering_lite_row(json_path)
        if row is not None:
            rows.append(row)

    if not rows:
        print("no rows to emit -- have any tiny-mfv evals run?")
        return

    cols = ["cue", "row", "axis_shift", "n_vig"] + [FOUNDATION_SHORT[f] for f in FOUNDATION_ORDER]
    df = pl.DataFrame(rows).select(cols)

    print("\n## OOD: tiny-mfv Care-vs-Traditional axis (directly comparable to steering-lite)\n")
    print("axis_shift = ΔlogitSanctity − ΔlogitCare (nats). +ve = moved toward "
          "traditional/binding; -ve = toward care. Per-foundation Δlogit is "
          "paired by (vid, cond) vs the unsteered (alpha=0) baseline. "
          "🟢 |axis|>0.5  🟡 >0.15  🔴 below noise.\n")
    print(tabulate(df.to_pandas(), headers="keys", tablefmt="pipe",
                   floatfmt="+.2f", showindex=False))


if __name__ == "__main__":
    main(tyro.cli(ReadmeTinymfvCfg))
