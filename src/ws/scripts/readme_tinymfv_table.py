"""README-ready tiny-mfv tables: ws adapters + steering-lite baselines.

Layout mirrors steering-lite's README:
  - Table 1 (bare): per-foundation absolute logit(is_wrong), one row, no Δ.
    Every Δ row below is measured against this prior.
  - Table 2 (Δ rows): cue | axis | method | C | kl | per-foundation
    `mean±std`. Header arrows mark target direction (Care ↓, Sanc ↑).

Same axis (Care vs Traditional/Sanctity), same metric (axis_shift in nats),
same paired-by-(vid,cond) per-foundation Δlogit. ws rows are read from
`out/trad_care/{adapter|base}/*__foundations_dlogit.csv` (eval already
computes them) plus `out/trad_care/kl_calibration/summary.csv` (calibrated
α and achieved p95). Steering-lite rows are read from
`<lite>/outputs/tinymfv_sweep/*.json`.

NB: ws weight-steering uses iso-KL calibrated alpha (target_kl=1.0 nat); the
steering-lite calibration is the same target. Both repos' rows are at the
same KL footprint, so axis_shift is directly comparable. ws prompt_only
(alpha=+1, no calibration) and steering-lite prompt_only are the only
un-calibrated points -- shown for context, C=n/a, kl=n/a.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import tyro
from tabulate import tabulate

from ws._artifacts import latest_matching


FOUNDATION_ORDER = ["Care", "Sanctity", "Authority", "Loyalty", "Fairness", "Liberty", "Social Norms"]
FOUNDATION_BARE = {
    "Care": "Care", "Sanctity": "Sanc", "Authority": "Auth",
    "Loyalty": "Loy", "Fairness": "Fair", "Liberty": "Lib", "Social Norms": "SocN",
}

# Per-behavior axis labels: arrows mark the target direction at +alpha.
# auth_care: POS persona = anti-authority + caring  ->  Care ↑, Auth ↓.
# trad_care: POS persona = traditional/sanctity     ->  Sanc ↑, Care ↓.
# auth_socn: POS persona = anti-authority + socnorm ->  SocN ↑, Auth ↓.
BEHAVIOR_AXIS: dict[str, dict] = {
    "auth_care": {
        "title": "OOD: tiny-mfv Authority↓+Care↑ axis (directly comparable to steering-lite)",
        "blurb": (
            "Task: shift the model away from authority-deference toward care for affected "
            "stakeholders. Headline metric `axis = ΔlogitCare − ΔlogitAuthority` (nats); Δ values "
            "are paired by (vignette, condition) so vignette difficulty cancels. Setup: "
            "target_kl=1.0 nat (iso-KL across methods), max_think=64, vignettes=airisk."
        ),
        "arrow_pos": "Care", "arrow_neg": "Authority",
    },
    "trad_care": {
        "title": "OOD: tiny-mfv Care-vs-Traditional axis (directly comparable to steering-lite)",
        "blurb": (
            "Task: shift the model from Care/harm morality toward Sanctity/traditionalist. "
            "Headline metric `axis = ΔlogitSanc − ΔlogitCare` (nats); Δ values are paired by "
            "(vignette, condition) so vignette difficulty cancels. Setup: target_kl=1.0 nat "
            "(iso-KL across methods), max_think=64, vignettes=airisk."
        ),
        "arrow_pos": "Sanctity", "arrow_neg": "Care",
    },
    "auth_socn": {
        "title": "OOD: tiny-mfv Authority↓+SocialNorms↑ axis (directly comparable to steering-lite)",
        "blurb": (
            "Task: shift the model away from formal authority toward peer/community consensus. "
            "Headline metric `axis = ΔlogitSocN − ΔlogitAuthority` (nats); Δ values are paired by "
            "(vignette, condition) so vignette difficulty cancels. Setup: target_kl=1.0 nat "
            "(iso-KL across methods), max_think=64, vignettes=airisk."
        ),
        "arrow_pos": "Social Norms", "arrow_neg": "Authority",
    },
    "authority": {
        "title": "ws Authority↓ (MFT framing) — directly comparable to steering-lite",
        "blurb": (
            "Task: shift the model away from authority-deference on the single Authority "
            "foundation (MFT-paper framing). Headline metric `axis = −ΔlogitAuthority` (nats); "
            "Δ values are paired by (vignette, condition) so vignette difficulty cancels. "
            "Setup: target_kl=1.0 nat (iso-KL across methods), max_think=64, vignettes=airisk. "
            "Persona prompts only (no engineered prompt)."
        ),
        "arrow_pos": None, "arrow_neg": "Authority",
    },
}


def _foundation_short(behavior: str) -> dict[str, str]:
    """Annotate FOUNDATION_BARE labels with ↑/↓ arrows for the active axis."""
    axis = BEHAVIOR_AXIS[behavior]
    out = dict(FOUNDATION_BARE)
    if axis["arrow_pos"] is not None:
        out[axis["arrow_pos"]] = f"{FOUNDATION_BARE[axis['arrow_pos']]} ↑"
    if axis["arrow_neg"] is not None:
        out[axis["arrow_neg"]] = f"{FOUNDATION_BARE[axis['arrow_neg']]} ↓"
    return out


@dataclass
class ReadmeTinymfvCfg:
    behavior: str = "auth_care"
    model_label: str = "Qwen3.5-4B"
    out: Path = Path("out")
    adapters: tuple[str, ...] = ("lora", "dora", "pissa", "delora", "oft", "ia3")
    include_prompt_baseline: bool = True
    include_steering_lite: bool = True
    steering_lite_root: Path = Path("/media/wassname/SGIronWolf/projects5/2026/lite/steering-lite")
    steering_lite_methods: tuple[str, ...] = (
        "prompt_only", "mean_diff", "mean_centred",
        "pca", "sspace", "cosine_gated", "topk_clusters",
    )
    target_alpha_sign: float = 1.0  # +1 = POS arm (engineered/POS persona); flip to read NEG side


def _cue(axis: float) -> str:
    if axis != axis:
        return "⚪"
    a = abs(axis)
    if a > 0.5:
        return "🟢"
    if a > 0.15:
        return "🟡"
    return "🔴"


def _fmt_pm(mean: float, std: float) -> str:
    if mean != mean:
        return "—"
    if std != std:
        return f"{mean:+.2f}"
    return f"{mean:+.2f}±{std:.2f}"


def _fmt_axis(axis: float) -> str:
    if axis != axis:
        return "—"
    return f"{axis:+.2f}"


def _fmt_C(c: float | None) -> str:
    if c is None or c != c:
        return "n/a"
    return f"{c:+.2f}"


def _fmt_kl(kl: float | None) -> str:
    if kl is None or kl != kl:
        return "n/a"
    return f"{kl:.2f}"


def _logit(w: float, eps: float = 0.01) -> float:
    w = max(eps, min(1.0 - eps, w))
    return math.log(w / (1.0 - w))


def _load_ws_calib(cfg: ReadmeTinymfvCfg) -> dict[str, dict]:
    """Read out/<behavior>/kl_calibration/summary.csv -> by adapter."""
    p = cfg.out / cfg.behavior / "kl_calibration" / "summary.csv"
    if not p.exists():
        return {}
    df = pl.read_csv(p)
    out: dict[str, dict] = {}
    for row in df.to_dicts():
        method = row.get("method", "")
        if not method.startswith("dW:"):
            continue
        adapter = method.split(":", 1)[1]
        out[adapter] = row
    return out


def _ws_bare_row(cfg: ReadmeTinymfvCfg) -> dict | None:
    """Compute absolute logit per foundation at α=0 from any adapter's per-vignette CSV.

    Mirrors steering-lite's bare table: mean over (vid, cond) of logit(wrongness).
    """
    for adapter in cfg.adapters:
        d = cfg.out / cfg.behavior / adapter
        if not d.exists():
            continue
        try:
            pv_path = latest_matching(d, "*__per_vignette.csv")
        except FileNotFoundError:
            continue
        pv = pl.read_csv(pv_path).filter(pl.col("alpha") == 0.0)
        if pv.is_empty():
            continue
        # per_vignette has wrongness_other_violate / wrongness_self_violate.
        # Unpivot to (vid, cond) -> wrongness, then logit-mean per foundation.
        long_rows = []
        for r in pv.to_dicts():
            for cond in ("other_violate", "self_violate"):
                w = r.get(f"wrongness_{cond}")
                if w is None:
                    continue
                long_rows.append({"foundation_coarse": r["foundation_coarse"],
                                  "logit": _logit(float(w))})
        if not long_rows:
            continue
        long_df = pl.DataFrame(long_rows)
        agg = long_df.group_by("foundation_coarse").agg(
            pl.col("logit").mean().alias("mean"),
            pl.col("logit").std().alias("std"),
            pl.len().alias("n"),
        )
        by_f = {r["foundation_coarse"]: r for r in agg.to_dicts()}
        return {"source": "ws", "by_f": by_f}
    return None


def _sl_bare_row(cfg: ReadmeTinymfvCfg) -> dict | None:
    p = cfg.steering_lite_root / "outputs" / "tinymfv_sweep" / "bare.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    alf = data.get("absolute_logit_per_foundation", {})
    if not alf:
        return None
    return {"source": "sl", "by_f": {f: {"mean": d.get("mean", float("nan")),
                                          "std": d.get("std", float("nan")),
                                          "n": d.get("n", 0)} for f, d in alf.items()}}


def _ws_delta_row(cfg: ReadmeTinymfvCfg, adapter: str, calib: dict[str, dict]) -> dict | None:
    d = cfg.out / cfg.behavior / adapter
    if not d.exists():
        return None
    try:
        summary_path = latest_matching(d, "*__summary.csv")
        dlogit_path = latest_matching(d, "*__foundations_dlogit.csv")
    except FileNotFoundError:
        return None
    summary = pl.read_csv(summary_path)
    dlogit = pl.read_csv(dlogit_path)
    # Pick the alpha row whose sign matches target_alpha_sign and is non-zero.
    alphas = [a for a in summary["alpha"].to_list()
              if a != 0.0 and (a > 0) == (cfg.target_alpha_sign > 0)]
    if not alphas:
        return None
    alpha = max(alphas, key=lambda x: abs(x))  # the largest-magnitude calibrated one
    sub = summary.filter(pl.col("alpha") == alpha)
    sub_d = dlogit.filter(pl.col("alpha") == alpha)
    if sub.is_empty() or sub_d.is_empty():
        return None
    by_f = {r["foundation_coarse"]: r for r in sub_d.to_dicts()}
    cal = calib.get(adapter, {})
    p95_key = "p95_at_pos" if cfg.target_alpha_sign > 0 else "p95_at_neg"
    row_dict = {
        "method": f"ws:{adapter}",
        "axis": float(sub["axis_shift"][0]),
        "C": float(alpha),
        "kl": float(cal.get(p95_key, float("nan"))) if cal else float("nan"),
        "by_f": by_f,
    }
    # Read SI if available (authority behavior)
    if "SI_Authority" in sub.columns:
        row_dict["si_authority"] = float(sub["SI_Authority"][0])
    return row_dict


def _ws_prompt_row(cfg: ReadmeTinymfvCfg) -> dict | None:
    base_dir = cfg.out / cfg.behavior / "base"
    if not base_dir.exists():
        return None
    try:
        summary_path = latest_matching(base_dir, "*__summary.csv")
        dlogit_path = latest_matching(base_dir, "*__foundations_dlogit.csv")
    except FileNotFoundError:
        return None
    summary = pl.read_csv(summary_path)
    dlogit = pl.read_csv(dlogit_path)
    alphas = [a for a in summary["alpha"].to_list()
              if a != 0.0 and (a > 0) == (cfg.target_alpha_sign > 0)]
    if not alphas:
        return None
    alpha = max(alphas, key=lambda x: abs(x))
    sub = summary.filter(pl.col("alpha") == alpha)
    sub_d = dlogit.filter(pl.col("alpha") == alpha)
    if sub.is_empty() or sub_d.is_empty():
        return None
    row_dict = {
        "method": "ws:prompt_only",
        "axis": float(sub["axis_shift"][0]),
        "C": float("nan"),
        "kl": float("nan"),
        "by_f": {r["foundation_coarse"]: r for r in sub_d.to_dicts()},
    }
    if "SI_Authority" in sub.columns:
        row_dict["si_authority"] = float(sub["SI_Authority"][0])
    return row_dict


def _sl_delta_row(cfg: ReadmeTinymfvCfg, method: str) -> dict | None:
    p = cfg.steering_lite_root / "outputs" / "tinymfv_sweep" / f"{method}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if "axis_shift" not in data or "dlogit_per_foundation" not in data:
        return None
    return {
        "method": f"sl:{method}",
        "axis": float(data["axis_shift"]),
        "C": float(data.get("coeff_calibrated", float("nan"))),
        "kl": float(data.get("kl_p95_at_calib", float("nan"))),
        "by_f": {f: {"dlogit_mean": d.get("mean", float("nan")),
                     "dlogit_std": d.get("std", float("nan")),
                     "n": d.get("n", 0)} for f, d in data["dlogit_per_foundation"].items()},
    }


def _print_bare_table(rows: list[dict], model_label: str) -> None:
    print("\n#### Bare model (no steering)\n")
    print("Absolute logit(is_wrong) per moral foundation, mean over vignettes × frames × conditions. "
          "Δ-rows below are measured against this prior.\n")
    headers = ["source"] + [FOUNDATION_BARE[f] for f in FOUNDATION_ORDER]
    out_rows = []
    for r in rows:
        if r is None:
            continue
        line = [f"ws ({model_label})" if r["source"] == "ws" else f"steering-lite ({model_label})"]
        for f in FOUNDATION_ORDER:
            d = r["by_f"].get(f, {})
            mean = d.get("mean", float("nan")) if isinstance(d, dict) else float("nan")
            std = d.get("std", float("nan")) if isinstance(d, dict) else float("nan")
            line.append(_fmt_pm(mean, std))
        out_rows.append(line)
    if not out_rows:
        print("(no bare data — alpha=0 eval not run yet)")
        return
    print(tabulate(out_rows, headers=headers, tablefmt="pipe", stralign="right",
                   disable_numparse=True))


def _print_delta_table(rows: list[dict], behavior: str) -> None:
    print("\n#### Steering methods (Δlogit vs bare, paired by (vid, cond))\n")
    print("`C` = calibrated coefficient at iso-KL target_kl=1.0 nat; `kl` = achieved kl_p95. "
          "Cells: `mean±std`. Cue: 🟢 |axis|>0.5  🟡 >0.15  🔴 below noise.\n")
    short = _foundation_short(behavior)
    headers = ["cue", "axis", "method", "C", "kl"] + [short[f] for f in FOUNDATION_ORDER]
    # Add SI column for authority behavior (single-foundation SI metric)
    has_si = behavior == "authority"
    if has_si:
        headers.append("SI_Auth")
    rows_sorted = sorted(rows, key=lambda r: -abs(r["axis"]) if r["axis"] == r["axis"] else 0)
    out_rows = []
    for r in rows_sorted:
        line = [_cue(r["axis"]), _fmt_axis(r["axis"]), r["method"], _fmt_C(r["C"]), _fmt_kl(r["kl"])]
        for f in FOUNDATION_ORDER:
            d = r["by_f"].get(f, {})
            mean = d.get("dlogit_mean", float("nan")) if isinstance(d, dict) else float("nan")
            std = d.get("dlogit_std", float("nan")) if isinstance(d, dict) else float("nan")
            line.append(_fmt_pm(mean, std))
        if has_si:
            si_val = r.get("si_authority", float("nan"))
            line.append(f"{si_val:+.2f}" if si_val == si_val else "—")
        out_rows.append(line)
    if not out_rows:
        print("(no Δ-rows -- run the calibrated tinymfv eval first)")
        return
    print(tabulate(out_rows, headers=headers, tablefmt="pipe", stralign="right",
                   disable_numparse=True))


def main(cfg: ReadmeTinymfvCfg) -> None:
    axis = BEHAVIOR_AXIS[cfg.behavior]
    print(f"\n## {axis['title']}\n")
    print(axis["blurb"] + "\n")
    print("Caveat: ws and steering-lite share the same persona pairs, dataset, and 1-nat KL "
          "budget, so calibrated rows are directly comparable. Uncalibrated rows "
          "(prompt_only, engineered_prompt) have no coefficient dial -- C=n/a, kl=n/a.\n")

    bare_rows = []
    ws_bare = _ws_bare_row(cfg)
    if ws_bare is not None:
        bare_rows.append(ws_bare)
    if cfg.include_steering_lite:
        sl_bare = _sl_bare_row(cfg)
        if sl_bare is not None:
            bare_rows.append(sl_bare)
    _print_bare_table(bare_rows, cfg.model_label)

    delta_rows = []
    if cfg.include_prompt_baseline:
        r = _ws_prompt_row(cfg)
        if r is not None:
            delta_rows.append(r)
    calib = _load_ws_calib(cfg)
    for adapter in cfg.adapters:
        r = _ws_delta_row(cfg, adapter, calib)
        if r is not None:
            delta_rows.append(r)
    if cfg.include_steering_lite:
        for method in cfg.steering_lite_methods:
            r = _sl_delta_row(cfg, method)
            if r is not None:
                delta_rows.append(r)
    _print_delta_table(delta_rows, cfg.behavior)


if __name__ == "__main__":
    main(tyro.cli(ReadmeTinymfvCfg))
