"""Qualitative sanity check: full-text generations at calibrated α per method.

Print 3 dilemmas under each method (base, prompt:eng_honest, every adapter at
calibrated α, RepE at calibrated α). Spot-check coherence and whether the
quantitative SI gap reflects qualitative behavior or just decoder collapse.
"""

from __future__ import annotations

import polars as pl
import torch
from baukit import TraceDict
from datasets import load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from ws.diff import DIFF_FILENAME, load_diff
from ws.eval.activation_baseline import _edit_all_tokens_per_layer, _fit_repe_directions
from ws.eval.dilemmas import INSTRUCTION_PROMPT, THINK_CLOSE, THINK_OPEN
from ws.eval.prompt_baseline import PROMPTS as PROMPT_TEXTS
from ws.steer import weight_steer

MODEL = "Qwen/Qwen3-0.6B"
N_DILEMMAS = 3
MAX_NEW = 100
SEED = 0


def build_prompt(tok, row, system_prompt: str = "") -> torch.Tensor:
    user = INSTRUCTION_PROMPT.format(**row)
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": "My choice: **"})
    text = tok.apply_chat_template(
        msgs, tokenize=False, continue_final_message=True, add_generation_prompt=False,
    )
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
    ids = enc.input_ids.squeeze(0)

    # Close <think> if open (same as dilemmas.py)
    think_open_id = tok.convert_tokens_to_ids(THINK_OPEN)
    think_close_id = tok.convert_tokens_to_ids(THINK_CLOSE)
    if think_open_id != tok.unk_token_id and think_close_id != tok.unk_token_id:
        ids_l = ids.tolist()
        if think_open_id in ids_l and think_close_id not in ids_l:
            think_pos = max(i for i, t in enumerate(ids_l) if t == think_open_id)
            nl_ids = tok.encode("\n\n", add_special_tokens=False)
            ids_l = ids_l[:think_pos + 1] + [think_close_id] + nl_ids + ids_l[think_pos + 1:]
            ids = torch.tensor(ids_l, dtype=torch.long)
    return ids


@torch.no_grad()
def generate(model, tok, ids: torch.Tensor, max_new: int = MAX_NEW) -> str:
    inp = ids.unsqueeze(0).to(model.device)
    out = model.generate(
        inp, max_new_tokens=max_new, do_sample=False, temperature=1.0,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    new_tokens = out[0, ids.shape[0]:]
    return tok.decode(new_tokens, skip_special_tokens=False)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # Load calibrated alphas
    calib = pl.read_csv("out/honesty/kl_calibration/summary.csv")
    calibrated = {row["method"]: float(row["calibrated_alpha"]) for row in calib.iter_rows(named=True)}
    logger.info(f"calibrated αs: {calibrated}")

    # Load 3 dilemmas
    ds = load_dataset("wassname/daily_dilemmas-self-honesty", "honesty_eval", split="test")
    # Take 3 with mixed honesty_label so we see both directions
    rows_used = []
    seen = set()
    for r in ds:
        di = r["dilemma_idx"]
        if di in seen:
            continue
        seen.add(di)
        rows_used.append(r)
        if len(rows_used) >= N_DILEMMAS:
            break

    # Build prompts (base = no system, prompt method = with sys prompt)
    base_prompts = [build_prompt(tok, r, "") for r in rows_used]
    eng_prompts = [build_prompt(tok, r, PROMPT_TEXTS["engineered_prompt_honest"]) for r in rows_used]

    # RepE directions
    repe_dirs = _fit_repe_directions(model, tok, n_train_topics=20, behavior="honesty")
    repe_layers = list(range(8, 22))

    output_lines = []
    for i, (row, base_ids, eng_ids) in enumerate(zip(rows_used, base_prompts, eng_prompts)):
        output_lines.append(f"\n{'='*80}\n=== DILEMMA {i+1} (idx={row['idx']}, action={row['action_type']}, honesty_label={row['honesty_label']:+d}) ===")
        output_lines.append(f"situation: {row['dilemma_situation'][:200]}...")
        output_lines.append(f"action: {row['action']}")
        output_lines.append(f"{'='*80}")

        # Base
        text = generate(model, tok, base_ids)
        output_lines.append(f"\n[base | α=0]\n{text}")

        # Prompt: engineered_honest
        text = generate(model, tok, eng_ids)
        output_lines.append(f"\n[prompt:engineered_honest | α=1]\n{text}")

        # Each adapter at calibrated α
        for method, alpha in calibrated.items():
            if method.startswith("dW:"):
                adapter = method.split(":", 1)[1]
                w = load_diff(f"out/honesty/{adapter}/{DIFF_FILENAME}")
                with weight_steer(model, w, alpha):
                    text = generate(model, tok, base_ids)
                output_lines.append(f"\n[{method} | α={alpha:+.3f}]\n{text}")
            elif method == "repe":
                hooks = [f"model.layers.{L}" for L in repe_layers]
                edit = _edit_all_tokens_per_layer(repe_dirs, repe_layers, alpha)
                with TraceDict(model, hooks, edit_output=edit):
                    text = generate(model, tok, base_ids)
                output_lines.append(f"\n[{method} | α={alpha:+.3f}]\n{text}")

        # Also show the negative direction for adapters (since user's α-sweep showed sign flip)
        for method, alpha in calibrated.items():
            if method.startswith("dW:"):
                adapter = method.split(":", 1)[1]
                w = load_diff(f"out/honesty/{adapter}/{DIFF_FILENAME}")
                with weight_steer(model, w, -alpha):
                    text = generate(model, tok, base_ids)
                output_lines.append(f"\n[{method} | α={-alpha:+.3f}]\n{text}")

    full = "\n".join(output_lines)
    out_path = "out/honesty/dilemmas_calibrated/demo_traces.txt"
    with open(out_path, "w") as f:
        f.write(full)
    print(full)
    logger.info(f"saved to {out_path}")


if __name__ == "__main__":
    main()
