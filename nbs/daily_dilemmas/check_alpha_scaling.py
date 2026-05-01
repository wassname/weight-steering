"""Sanity check: at α=1, 2, 4 (× calibrated), does output stay coherent or crash?

Mirrors the gist's three-panel α=1/2/4 figure but in text form: same prompt,
greedy-generate 20 thinking tokens, print text + per-position KL. We expect:
  α=1: coherent CoT, p95 KL near 1
  α=2: brief KL spikes, mostly recovers, still readable
  α=4: parks above the road, output drifts/garbles
"""

from __future__ import annotations

import polars as pl
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from ws.data import _load_suffixes
from ws.diff import DIFF_FILENAME, load_diff
from ws.eval._steer_common import (
    build_chat_ids,
    build_chat_text,
    greedy_generate_under_steering,
    teacher_force_logp,
)

MODEL = "Qwen/Qwen3-0.6B"
N_TOKENS = 20
SCALES = (1.0, 2.0, 4.0)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    calib = pl.read_csv("out/honesty/kl_calibration/summary.csv")
    methods = calib.filter(pl.col("method").str.starts_with("dW:")).rows(named=True)
    logger.info(f"loaded {len(methods)} dW methods from calibration")

    entries = _load_suffixes(thinking=False)
    p = entries[0]  # first calib prompt
    base_text = build_chat_text(tok, "", p["user_msg"], "", thinking=True)
    base_ids = build_chat_ids(tok, "", p["user_msg"], "", thinking=True)
    print(f"\nPROMPT:\n{base_text}\n")

    for row in methods:
        method = row["method"]
        alpha_c = float(row["calibrated_alpha"])
        adapter = method.split(":", 1)[1]
        w = load_diff(f"out/honesty/{adapter}/{DIFF_FILENAME}")

        print(f"\n{'='*70}\n{method} (calibrated α={alpha_c:.3f})\n{'='*70}")
        for scale in SCALES:
            alpha = scale * alpha_c
            with torch.no_grad():
                gen_ids, logp_steered = greedy_generate_under_steering(
                    model, tok, base_ids, method=method, alpha=alpha,
                    n_new_tokens=N_TOKENS, w=w,
                )
                full_base_ids = torch.cat([base_ids, gen_ids])
                logp_base = teacher_force_logp(model, full_base_ids, gen_ids.shape[0])
            kl = (logp_steered.exp() * (logp_steered - logp_base)).sum(-1).numpy()
            text = tok.decode(gen_ids, skip_special_tokens=False)
            print(f"\n  scale={scale:.1f}× → α={alpha:+.3f}  p95={float(sorted(kl)[int(0.95*len(kl))]):.3f}  max={float(kl.max()):.3f}")
            print(f"    {text!r}")
            print(f"    kl/pos: {[f'{k:.2f}' for k in kl.tolist()]}")


if __name__ == "__main__":
    main()
