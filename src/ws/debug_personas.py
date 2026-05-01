"""One-off persona collapse debugger.

For each persona pair, greedy-generate short continuations on a fixed prompt
set and warn if left/right collapse to the same text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
import torch
import tyro
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from ws._log import final_summary, get_argv, setup_logging
from ws.data import _normalize_text, _personas, _render_chat_prompt, _topics


@dataclass
class PersonaDebugCfg:
    model: str = "Qwen/Qwen3-0.6B"
    behavior: str = "honesty"
    out: Path = Path("out")
    n_prompts: int = 8
    max_new_tokens: int = 100
    batch_size: int = 8
    seed: int = 0


@torch.no_grad()
def _greedy_batch(model, tok, prompts: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    rows: list[str] = []
    old_padding_side = tok.padding_side
    tok.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            enc = tok(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
            gen_block = out[:, enc["input_ids"].shape[1]:].cpu()
            for i in range(len(batch_prompts)):
                rows.append(tok.decode(gen_block[i], skip_special_tokens=True).rstrip())
    finally:
        tok.padding_side = old_padding_side
    return rows


def main(cfg: PersonaDebugCfg) -> None:
    setup_logging("debug_personas")
    logger.info(f"argv: {get_argv()}")
    logger.info(f"persona debug cfg: {asdict(cfg)}")

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    pos_personas, neg_personas = _personas(cfg.behavior)
    topics = _topics(cfg.behavior)[:cfg.n_prompts]
    prompts: list[str] = []
    for a, b in topics:
        prompts.append(f"I think {a}. {b}" if cfg.behavior == "sycophancy" else a)

    rows = []
    for persona_idx, (sys_pos, sys_neg) in enumerate(zip(pos_personas, neg_personas, strict=True)):
        prompt_pos = [_render_chat_prompt(tok, sys_pos, prompt) for prompt in prompts]
        prompt_neg = [_render_chat_prompt(tok, sys_neg, prompt) for prompt in prompts]
        gens_pos = _greedy_batch(model, tok, prompt_pos, cfg.batch_size, cfg.max_new_tokens)
        gens_neg = _greedy_batch(model, tok, prompt_neg, cfg.batch_size, cfg.max_new_tokens)
        identical = 0
        for prompt, gen_pos, gen_neg in zip(prompts, gens_pos, gens_neg, strict=True):
            same = _normalize_text(gen_pos) == _normalize_text(gen_neg)
            identical += int(same)
            rows.append({
                "persona_idx": persona_idx,
                "prompt": prompt,
                "same": same,
                "response_pos": gen_pos,
                "response_neg": gen_neg,
            })
        if identical:
            logger.warning(
                f"persona_idx={persona_idx} collapsed on {identical}/{len(prompts)} greedy probes; "
                "discard this pair from persona debugging."
            )

    df = pl.DataFrame(rows)
    out_dir = cfg.out / cfg.behavior / "persona_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_prompt_path = out_dir / "per_prompt.csv"
    summary_path = out_dir / "summary.csv"
    df.write_csv(per_prompt_path)

    summary = (
        df.group_by("persona_idx")
        .agg(
            pl.len().alias("n_prompts"),
            pl.col("same").sum().alias("n_same"),
        )
        .with_columns(
            (pl.col("n_same") / pl.col("n_prompts")).alias("same_rate"),
            (pl.col("n_same") == 0).alias("keep_pair"),
        )
        .sort("persona_idx")
    )
    summary.write_csv(summary_path)

    print("\npersona_debug")
    print("SHOULD: left/right greedy probes differ for each persona pair. same_rate>0 means the persona contrast is weak or ignored.")
    print(tabulate(summary.to_pandas(), headers="keys", tablefmt="tsv", floatfmt="+.3f", showindex=False))

    cue = "🟢" if bool(summary["keep_pair"].all()) else "🟡"
    final_summary(
        out=summary_path,
        argv=get_argv(),
        main_metric=f"keep_pairs={int(summary['keep_pair'].sum())}/{len(summary)}",
        cue=cue,
        table_rows=summary.select("persona_idx", "n_prompts", "n_same", "same_rate", "keep_pair").rows(),
        headers=["persona_idx", "n_prompts", "n_same", "same_rate", "keep_pair"],
        floatfmt="",
    )


if __name__ == "__main__":
    main(tyro.cli(PersonaDebugCfg))
