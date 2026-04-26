# %% [markdown]
# # v10 functional projection falsifier
#
# v9 measured geometric span overlap. This script asks the load-bearing question:
# if we keep only the part of `dW` that writes inside the block-local
# activation oracle, does daily-dilemmas steering survive?
#
# Interpretation:
# - high retention at small K: v9 overlap metric was the wrong norm.
# - low retention even at K=32: act_oracle PCA is not the functional steering subspace.

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
from datasets import load_dataset
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding

from ws.data import SYCOPHANCY_TOPICS
from ws.diff import load_diff
from ws.eval.dilemmas import DilemmasCfg, _eval_at_coeff, _format_row, summarize
from ws.eval.sycophancy import get_choice_ids
from ws.steer import weight_steer


MODEL_ID = "Qwen/Qwen3-0.6B"
RESIDUAL_WRITE_RE = re.compile(r"model\.layers\.(\d+)\.(self_attn\.o_proj|mlp\.down_proj)\.weight")


@dataclass
class Cli:
    out: Path = Path("out/sycophancy/v10_functional_projection")
    adapters: tuple[str, ...] = ("lora", "dora", "pissa", "delora", "oft", "ia3")
    ks: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    alphas: tuple[float, ...] = (1.0,)
    n_dilemmas: int = 40
    batch_size: int = 8
    max_tokens: int = 512
    model_id: str = MODEL_ID


def setup_logger() -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=False, format="{message}")


def sycophancy_probe_prompts() -> list[str]:
    return [f"I think {claim}. Do you agree? Answer Yes or No." for claim, _q in SYCOPHANCY_TOPICS]


def encode_last_token(tok, prompts: list[str], device: torch.device):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    seq_idx = enc.attention_mask.sum(-1) - 1
    return enc, seq_idx


@torch.no_grad()
def capture_pre_post(model, tok, w: dict[str, Tensor], prompts: list[str], alpha: float) -> tuple[Tensor, Tensor]:
    enc, seq_idx = encode_last_token(tok, prompts, model.device)
    with weight_steer(model, w, alpha):
        out = model(**enc, output_hidden_states=True)
    if out.hidden_states is None:
        raise RuntimeError("output_hidden_states is None")

    b = enc.input_ids.shape[0]
    d_model = out.hidden_states[0].shape[-1]
    idx = seq_idx.cpu().view(b, 1, 1).expand(b, 1, d_model)
    pre, post = [], []
    for layer in range(model.config.num_hidden_layers):
        hs_pre = out.hidden_states[layer].float().cpu()
        hs_post = out.hidden_states[layer + 1].float().cpu()
        pre.append(hs_pre.gather(1, idx).squeeze(1))
        post.append(hs_post.gather(1, idx).squeeze(1))
    return torch.stack(pre), torch.stack(post)


def right_svd_basis(samples: Tensor, k: int) -> tuple[Tensor, Tensor]:
    norms = samples.norm(dim=1, keepdim=True).clamp(min=1e-12)
    samples_unit = samples.float().cpu() / norms
    _u, s, vh = torch.linalg.svd(samples_unit, full_matrices=False)
    return vh[: min(k, vh.shape[0])].T.contiguous(), s


def block_act_oracle_bases(model, tok, w: dict[str, Tensor], max_k: int) -> tuple[list[Tensor], list[Tensor]]:
    prompts = sycophancy_probe_prompts()
    pre_pos, post_pos = capture_pre_post(model, tok, w, prompts, alpha=+1.0)
    pre_neg, post_neg = capture_pre_post(model, tok, w, prompts, alpha=-1.0)
    block_diff = (post_pos - pre_pos) - (post_neg - pre_neg)
    bases, spectra = [], []
    for layer in range(model.config.num_hidden_layers):
        B, s = right_svd_basis(block_diff[layer], max_k)
        bases.append(B)
        spectra.append(s)
    return bases, spectra


def residual_write_layer(key: str) -> int | None:
    match = RESIDUAL_WRITE_RE.fullmatch(key)
    return None if match is None else int(match.group(1))


def project_w_to_layer_bases(w: dict[str, Tensor], bases: list[Tensor], k: int) -> dict[str, Tensor]:
    projected = {}
    for key, value in w.items():
        layer = residual_write_layer(key)
        if layer is None:
            continue
        B = bases[layer][:, : min(k, bases[layer].shape[1])]
        projected[key] = (B @ (B.T @ value.float().cpu())).to(value.dtype)
    if not projected:
        raise ValueError("projected diff is empty; no residual-output weight keys matched")
    return projected


def residual_write_only_w(w: dict[str, Tensor]) -> dict[str, Tensor]:
    residual = {key: value for key, value in w.items() if residual_write_layer(key) is not None}
    if not residual:
        raise ValueError("residual-write diff is empty; no o_proj/down_proj weight keys matched")
    return residual


def complement_w_to_layer_bases(w: dict[str, Tensor], bases: list[Tensor], k: int) -> dict[str, Tensor]:
    complement = {}
    for key, value in w.items():
        layer = residual_write_layer(key)
        if layer is None:
            continue
        B = bases[layer][:, : min(k, bases[layer].shape[1])]
        W = value.float().cpu()
        complement[key] = (W - B @ (B.T @ W)).to(value.dtype)
    if not complement:
        raise ValueError("complement diff is empty; no residual-output weight keys matched")
    return complement


def diff_norm(w: dict[str, Tensor]) -> float:
    return sum(tensor_energy(v) for v in w.values()) ** 0.5


def scale_diff(w: dict[str, Tensor], scale: float) -> dict[str, Tensor]:
    return {key: (value.float().cpu() * scale).to(value.dtype) for key, value in w.items()}


def tensor_energy(value: Tensor) -> float:
    return float(value.float().pow(2).sum().item())


def spectra_rows(adapter: str, w: dict[str, Tensor], bases: list[Tensor], act_spectra: list[Tensor], ks: tuple[int, ...]) -> list[dict]:
    rows = []
    for key, value in w.items():
        layer = residual_write_layer(key)
        if layer is None:
            continue
        W = value.float().cpu()
        dW_s = torch.linalg.svdvals(W)
        dW_s2 = dW_s.pow(2)
        act_s = act_spectra[layer]
        act_s2 = act_s.pow(2)
        dW_total = dW_s2.sum().clamp(min=1e-12)
        act_total = act_s2.sum().clamp(min=1e-12)
        dW_participation_rank = float(dW_s2.sum().pow(2) / dW_s2.pow(2).sum().clamp(min=1e-12))
        act_participation_rank = float(act_s2.sum().pow(2) / act_s2.pow(2).sum().clamp(min=1e-12))
        for k in ks:
            B = bases[layer][:, : min(k, bases[layer].shape[1])]
            dW_in_act = (B.T @ W).pow(2).sum() / W.pow(2).sum().clamp(min=1e-12)
            rows.append({
                "adapter": adapter,
                "layer": layer,
                "tensor": key,
                "k": k,
                "act_rank_available": bases[layer].shape[1],
                "act_energy_topk_frac": float(act_s2[: min(k, act_s2.numel())].sum() / act_total),
                "act_participation_rank": act_participation_rank,
                "dW_energy_topk_frac": float(dW_s2[: min(k, dW_s2.numel())].sum() / dW_total),
                "dW_participation_rank": dW_participation_rank,
                "dW_energy_in_actK_frac": float(dW_in_act),
                "dW_norm": float(W.pow(2).sum().sqrt()),
                "dW_projected_norm": float((B.T @ W).pow(2).sum().sqrt()),
            })
    return rows


def load_dilemmas_eval(tok, cfg: DilemmasCfg):
    ds = load_dataset("wassname/daily_dilemmas-self-honesty", "honesty_eval", split="test")
    honesty_labels = {(r["dilemma_idx"], r["action_type"]): r["honesty_label"] for r in ds}
    keep = set(sorted(set(ds["dilemma_idx"]))[: cfg.n_dilemmas])
    ds_eval = ds.filter(lambda x: x["dilemma_idx"] in keep)
    ds_pt = ds_eval.map(
        lambda x: _format_row(x, tok, cfg.max_tokens, cfg.system_prompt),
        remove_columns=ds_eval.column_names,
        load_from_cache_file=False,
    )
    ds_pt = ds_pt.with_format("torch", columns=["input_ids", "dilemma_idx", "idx"])
    dl = DataLoader(ds_pt, batch_size=cfg.batch_size, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer=tok, padding="longest"))
    meta = pl.DataFrame([
        {"idx": r["idx"], "action_type": r["action_type"], "honesty_label": float(honesty_labels[(r["dilemma_idx"], r["action_type"])])}
        for r in ds_eval
    ])
    return dl, meta


def rows_with_honesty(rows: list[dict], meta: pl.DataFrame, *, adapter: str, variant: str, k: int | None) -> pl.DataFrame:
    return pl.DataFrame(rows).join(meta, on="idx", how="left").with_columns(
        (pl.col("logratio") * pl.col("honesty_label")).alias("logratio_honesty"),
        pl.lit(adapter).alias("adapter"),
        pl.lit(variant).alias("variant"),
        pl.lit(k).cast(pl.Int64).alias("k"),
    )


def behavior_summary(df: pl.DataFrame) -> pl.DataFrame:
    by_coeff = behavior_by_coeff(df)
    base = by_coeff.select("adapter", "variant", "k", "coeff", "logratio_at_0")
    pos = (
        by_coeff.filter(pl.col("coeff") == 1.0)
        .select("adapter", "variant", "k", "logratio_at_pos", "logratio_at_0", "delta_pos_minus_zero", "mean_pmass", "frac_low_pmass", "n")
    )
    full_delta = pos.filter(pl.col("variant") == "full_all_tensors").select(
        "adapter", pl.col("delta_pos_minus_zero").alias("full_delta")
    )
    resid_delta = pos.filter(pl.col("variant") == "residual_write_full").select(
        "adapter", pl.col("delta_pos_minus_zero").alias("residual_write_delta")
    )
    return (
        pos.join(full_delta, on="adapter", how="left")
        .join(resid_delta, on="adapter", how="left")
        .with_columns(
            (pl.col("delta_pos_minus_zero") / pl.col("full_delta")).alias("retention_vs_full"),
            (pl.col("delta_pos_minus_zero") / pl.col("residual_write_delta")).alias("retention_vs_residual_write"),
        )
        .rename({"logratio_at_pos": "logratio_at_pos1"})
        .sort("adapter", "variant", "k")
    )


def behavior_by_coeff(df: pl.DataFrame) -> pl.DataFrame:
    by_coeff = (
        df.group_by("adapter", "variant", "k", "coeff")
        .agg(
            pl.col("logratio_honesty").mean().alias("mean_logratio_honesty"),
            pl.col("pmass").mean().alias("mean_pmass"),
            pl.col("low_pmass").mean().alias("frac_low_pmass"),
            pl.len().alias("n"),
        )
    )
    base = (
        by_coeff.filter((pl.col("variant") == "base") & (pl.col("coeff") == 0.0))
        .select("adapter", pl.col("mean_logratio_honesty").alias("logratio_at_0"))
    )
    return (
        by_coeff.filter(pl.col("variant") != "base")
        .rename({"mean_logratio_honesty": "logratio_at_pos"})
        .join(base, on="adapter", how="left")
        .with_columns((pl.col("logratio_at_pos") - pl.col("logratio_at_0")).alias("delta_pos_minus_zero"))
        .sort("adapter", "variant", "k", "coeff")
    )


def eval_variant(model, dl, choice_ids, cfg: DilemmasCfg, w_variant: dict[str, Tensor], alphas: tuple[float, ...]) -> list[dict]:
    rows = []
    for alpha in alphas:
        rows.extend(_eval_at_coeff(model, dl, float(alpha), w_variant, choice_ids, cfg.pmass_threshold))
    return rows


def main() -> None:
    import tyro

    setup_logger()
    cli = tyro.cli(Cli)
    cli.out.mkdir(parents=True, exist_ok=True)
    max_k = max(cli.ks)

    tok = AutoTokenizer.from_pretrained(cli.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(cli.model_id, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    model.eval()

    cfg = DilemmasCfg(model_id=cli.model_id, coeffs=(0.0, 1.0), n_dilemmas=cli.n_dilemmas, batch_size=cli.batch_size, max_tokens=cli.max_tokens)
    dl, meta = load_dilemmas_eval(tok, cfg)
    choice_ids = get_choice_ids(tok)

    per_row_parts = []
    spectra_parts = []
    for adapter in cli.adapters:
        w_path = Path("out") / "sycophancy" / adapter / "w.pt"
        if not w_path.exists():
            raise FileNotFoundError(w_path)
        logger.info(f"adapter={adapter}: loading {w_path}")
        w = load_diff(w_path)
        w_resid = residual_write_only_w(w)
        bases, act_spectra = block_act_oracle_bases(model, tok, w, max_k=max_k)
        spectra_parts.extend(spectra_rows(adapter, w, bases, act_spectra, cli.ks))

        base_rows = _eval_at_coeff(model, dl, 0.0, {}, choice_ids, cfg.pmass_threshold)
        per_row_parts.append(rows_with_honesty(base_rows, meta, adapter=adapter, variant="base", k=None))

        full_rows = eval_variant(model, dl, choice_ids, cfg, w, cli.alphas)
        per_row_parts.append(rows_with_honesty(full_rows, meta, adapter=adapter, variant="full_all_tensors", k=None))
        logger.info(f"adapter={adapter}: full rows={len(full_rows)}")

        resid_rows = eval_variant(model, dl, choice_ids, cfg, w_resid, cli.alphas)
        per_row_parts.append(rows_with_honesty(resid_rows, meta, adapter=adapter, variant="residual_write_full", k=None))
        resid_norm = diff_norm(w_resid)
        logger.info(f"adapter={adapter}: residual_write_norm/full_norm={resid_norm / diff_norm(w):.4f}")

        for k in cli.ks:
            projected = project_w_to_layer_bases(w, bases, k)
            complement = complement_w_to_layer_bases(w, bases, k)
            projected_norm = diff_norm(projected)
            normmatched = scale_diff(projected, resid_norm / max(projected_norm, 1e-12))
            logger.info(f"adapter={adapter} k={k}: projected_resid_norm/full_resid_norm={projected_norm / resid_norm:.4f}")
            rows = eval_variant(model, dl, choice_ids, cfg, projected, cli.alphas)
            per_row_parts.append(rows_with_honesty(rows, meta, adapter=adapter, variant="project_act_block", k=k))
            rows = eval_variant(model, dl, choice_ids, cfg, normmatched, cli.alphas)
            per_row_parts.append(rows_with_honesty(rows, meta, adapter=adapter, variant="project_act_block_normmatched", k=k))
            rows = eval_variant(model, dl, choice_ids, cfg, complement, cli.alphas)
            per_row_parts.append(rows_with_honesty(rows, meta, adapter=adapter, variant="complement_act_block", k=k))

    per_row = pl.concat(per_row_parts, how="vertical")
    spectra = pl.DataFrame(spectra_parts)
    by_coeff = behavior_by_coeff(per_row)
    summary = behavior_summary(per_row)

    per_row.write_csv(cli.out / "behavior_per_row.csv")
    spectra.write_csv(cli.out / "spectra_and_projection.csv")
    by_coeff.write_csv(cli.out / "behavior_by_coeff.csv")
    summary.write_csv(cli.out / "behavior_summary.csv")

    print("\nSHOULD: project_act_block retention distinguishes whether small act_oracle overlap is functionally load-bearing.")
    print("SHOULD: complement_act_block keeps behavior if the orthogonal residual-write component is load-bearing.")
    print("ELSE: projection retention near 1 after norm matching means v9 overlap used wrong norm; projection near 0 and complement near 1 means act_oracle PCA is not the steering subspace.")
    print(summary.select("adapter", "variant", "k", "logratio_at_0", "logratio_at_pos1", "delta_pos_minus_zero", "retention_vs_full", "retention_vs_residual_write").to_pandas().to_string(index=False))
    print(f"\nwrote: {cli.out}")


if __name__ == "__main__":
    main()
