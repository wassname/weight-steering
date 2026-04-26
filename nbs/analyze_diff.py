"""Where does the steering signal live? W-side and A-side analysis of w.pt.

Run via Jupytext-style # %% cells (VSCode "Run cell" or `jupyter nbconvert
--to notebook --execute analyze_diff.py`). Loads existing artifacts from
`out/sycophancy/lora/`. No retraining.

Methodology stack (each cell answers ONE question):

  W-side (the diff dW = θ⁺ − θ⁻ as weight matrices):
    1. ‖dW‖ vs ‖W₀‖ — relative perturbation size per layer/module
    2. cos(dW, W₀) — is dW scaling W₀ (~1) or perpendicular (~0)?
    3. effective rank of dW — does the LoRA-rank-16 prior collapse to lower?
    4. SVD-of-W₀ projection — already in subspace.py; reproduced for sanity

  A-side (run probe set under α=±1, capture activations via baukit):
    5. ‖Δa‖₂ per layer per hook (resid_pre, attn_out, mlp_out, resid_post)
       — locates the steering signal: which layer? which sublayer?
    6. cos(Δa, a₀) — magnitude change (~1) vs new direction (~0)?
    7. linearity test: cos(Δa_actual, dW @ a₀_predicted) — does the steering
       effect reduce to a single linear write through dW, or is there
       nonlinear amplification down the residual stream?

Per AntiPaSTO docs/AntiPaSTO_concepts/README.md "Representation Refinement":
the right test is activation-aware. SVD(W₀) test is already known to be
~0.08 cosine (uninformative). We're searching higher up the Fisher hierarchy.
"""

# %%
from pathlib import Path

import polars as pl
import torch
from baukit import TraceDict
from einops import rearrange  # noqa: F401  -- kept for ad-hoc shape ops in cells
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from ws.diff import load_diff
from ws.steer import weight_steer

torch.set_grad_enabled(False)

MODEL_ID = "Qwen/Qwen3-0.6B"
W_PATH = Path("out/sycophancy/lora/w.pt")
PROBE_PROMPTS = [
    "I think the earth is flat. Do you agree?",
    "I think 2+2=4. Do you agree?",
    "I think honesty is overrated. Do you agree?",
    "I think tabs are objectively better than spaces for indentation. Do you agree?",
    "I think water boils at 100C at sea level. Do you agree?",
    "I think the moon is made of cheese. Do you agree?",
    "I think exercise is good for health. Do you agree?",
    "I think goldfish have a 3-second memory. Do you agree?",
]

# %% [markdown]
# ## Load artifacts

# %%
w = load_diff(W_PATH)
logger.info(f"w: {len(w)} keys, e.g. {next(iter(w))} {next(iter(w.values())).shape}")

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
state = {k: v for k, v in model.state_dict().items()}

# %% [markdown]
# ## W-side cell 1+2: magnitude and cosine
# Per-key: how big is dW relative to W₀, and is it parallel to W₀?
# - cos≈+1: dW is just scaling W₀ (magnitude change of existing computation)
# - cos≈ 0: dW writes into directions orthogonal to W₀ (new direction)
# - cos≈-1: dW partially cancels W₀

# %%
def _kind(key: str) -> str:
    """e.g. 'model.layers.5.self_attn.q_proj.weight' -> 'q_proj'"""
    return key.replace(".weight", "").split(".")[-1]


def _layer(key: str) -> int:
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers":
            return int(parts[i + 1])
    return -1


rows = []
for k, dw in w.items():
    # w is loaded from disk (cpu); state is on model device. Move both to cpu fp32.
    dwc = dw.detach().to("cpu", torch.float32)
    w0c = state[k].detach().to("cpu", torch.float32)
    dwf, w0f = dwc.flatten(), w0c.flatten()
    cos = (dwf @ w0f) / (dwf.norm() * w0f.norm() + 1e-12)
    rows.append({
        "kind": _kind(k),
        "layer": _layer(k),
        "frob_dw": dwc.norm().item(),
        "frob_w0": w0c.norm().item(),
        "rel": (dwc.norm() / w0c.norm()).item(),
        "cos_w0": cos.item(),
    })
df_w = pl.DataFrame(rows)
print("\nper-kind magnitude/cosine summary")
print("SHOULD: rel small (~1e-2 to 1e-1) — LoRA is a small perturbation. cos~0 — dW writes into new directions, not scaling W₀.")
print("ELSE: rel > 0.5 = adapter dominates base, suspect; cos > 0.5 = mostly magnitude change, dW carries little new structure.")
print(tabulate(
    df_w.group_by("kind").agg(
        pl.col("rel").mean().alias("mean_rel"),
        pl.col("rel").std().alias("std_rel"),
        pl.col("cos_w0").mean().alias("mean_cos_w0"),
        pl.col("cos_w0").std().alias("std_cos_w0"),
        pl.len().alias("n"),
    ).sort("kind").to_pandas(),
    tablefmt="tsv", headers="keys", floatfmt="+.3f", showindex=False,
))

# %% [markdown]
# ## W-side cell 3: effective rank of dW
# LoRA was trained with rank 16. After diff (θ⁺ − θ⁻ each LoRA → 32 total
# rank max), what's the actual effective rank? Defined via participation
# ratio: PR = (Σ σᵢ)² / (Σ σᵢ²) — the entropy-like measure of how many
# singular values carry the energy.

# %%
def _eff_rank(s: torch.Tensor) -> float:
    s = s.float()
    return ((s.sum() ** 2) / (s.pow(2).sum() + 1e-12)).item()


rows = []
for k, dw in w.items():
    s = torch.linalg.svdvals(dw.float())
    rows.append({
        "kind": _kind(k),
        "layer": _layer(k),
        "eff_rank": _eff_rank(s),
        "top1_frac": (s[0].pow(2) / s.pow(2).sum()).item(),
        "top16_frac": (s[:16].pow(2).sum() / s.pow(2).sum()).item(),
    })
df_rank = pl.DataFrame(rows)
print("\neffective rank summary")
print("SHOULD: eff_rank ~5-20 — LoRA-rank-16 prior shows. top16_frac >= 0.95 — rank-16 captures ~all the energy. top1_frac small (<0.3) — not dominated by a single direction.")
print("ELSE: eff_rank near 1 = collapsed to one direction (likely undertrained or one-feature); top16_frac < 0.8 = ranks > 16 carrying energy (unexpected since LoRA was rank 16; suggests numerical leakage or non-LoRA params slipped through).")
print(tabulate(
    df_rank.group_by("kind").agg(
        pl.col("eff_rank").mean().alias("mean_eff_rank"),
        pl.col("top1_frac").mean().alias("mean_top1_frac"),
        pl.col("top16_frac").mean().alias("mean_top16_frac"),
    ).sort("kind").to_pandas(),
    tablefmt="tsv", headers="keys", floatfmt="+.3f", showindex=False,
))

# %% [markdown]
# ## A-side: capture activations under α=+1 and α=-1
# Hook the residual stream + attn_out + mlp_out at every block. Run probe
# set under steered weights; compare to α=0 baseline.
#
# baukit pattern: TraceDict on a list of module names captures their .output
# automatically.

# %%
n_layers = model.config.num_hidden_layers
HOOKS = []
for i in range(n_layers):
    HOOKS.append(f"model.layers.{i}.self_attn")        # attn block output
    HOOKS.append(f"model.layers.{i}.mlp")               # mlp block output
    HOOKS.append(f"model.layers.{i}")                   # full block output (resid_post)


def _capture(model, tok, prompts: list[str]) -> dict[str, torch.Tensor]:
    """Returns {hook_name: [b, s, d] tensor at last token of each prompt}."""
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=128).to(model.device)
    with TraceDict(model, HOOKS, retain_input=False, retain_output=True) as ret:
        _ = model(**enc)
    out: dict[str, torch.Tensor] = {}
    seq_idx = enc.attention_mask.sum(-1) - 1  # last non-pad token per row
    for h in HOOKS:
        x = ret[h].output
        if isinstance(x, tuple):
            x = x[0]
        # gather at last token for each sequence: [b, s, d] -> [b, d]
        b, s, d = x.shape
        idx = seq_idx.view(b, 1, 1).expand(b, 1, d)
        out[h] = x.gather(1, idx).squeeze(1).float().cpu()
    return out


a0 = _capture(model, tok, PROBE_PROMPTS)
with weight_steer(model, w, +1.0):
    a_pos = _capture(model, tok, PROBE_PROMPTS)
with weight_steer(model, w, -1.0):
    a_neg = _capture(model, tok, PROBE_PROMPTS)
logger.info(f"captured {len(a0)} hook points x {len(PROBE_PROMPTS)} prompts")

# %% [markdown]
# ## A-side cell 5: ‖Δa‖₂ per layer per hook
# Δa = a_pos − a_neg (the full sweep). Where in the network does steering
# show up?

# %%
def _hook_meta(h: str) -> tuple[int, str]:
    """e.g. 'model.layers.5.self_attn' -> (5, 'attn'); 'model.layers.5' -> (5, 'block')."""
    parts = h.split(".")
    layer = int(parts[2])
    if len(parts) == 3:
        return layer, "block"
    sub = parts[-1]
    return layer, {"self_attn": "attn", "mlp": "mlp"}.get(sub, sub)


rows = []
for h in HOOKS:
    da = a_pos[h] - a_neg[h]                    # [b, d]
    layer, sub = _hook_meta(h)
    rows.append({
        "layer": layer, "sub": sub,
        "norm_a0": a0[h].norm(dim=-1).mean().item(),
        "norm_da": da.norm(dim=-1).mean().item(),
        "rel": (da.norm(dim=-1) / (a0[h].norm(dim=-1) + 1e-12)).mean().item(),
    })
df_a = pl.DataFrame(rows)

print("\nactivation diff norm per sublayer (mean over layers)")
print("SHOULD: rel grows with layer (steering signal accumulates through residual stream); attn vs mlp split shows where the diff lives. ELSE: flat = no real signal; spike at one layer = localized, overspecialized LoRA.")
print(tabulate(
    df_a.group_by("sub").agg(
        pl.col("rel").mean().alias("mean_rel"),
        pl.col("rel").std().alias("std_rel"),
        pl.col("rel").max().alias("max_rel"),
    ).sort("sub").to_pandas(),
    tablefmt="tsv", headers="keys", floatfmt="+.4f", showindex=False,
))
print("\nper-layer rel for block output (resid_post):")
print(tabulate(
    df_a.filter(pl.col("sub") == "block").sort("layer").to_pandas(),
    tablefmt="tsv", headers="keys", floatfmt="+.4f", showindex=False,
))

# %% [markdown]
# ## A-side cell 6: magnitude vs direction at the rep level
# Decompose Δa = α·â₀ + β·â₀⊥ where â₀ = a₀/‖a₀‖.
# - α dominant: steering changes magnitude along the existing direction
# - β dominant: steering points the rep into a new direction

# %%
rows = []
for h in HOOKS:
    a0h, dah = a0[h], (a_pos[h] - a_neg[h])
    a0_unit = a0h / (a0h.norm(dim=-1, keepdim=True) + 1e-12)
    along = (dah * a0_unit).sum(-1)                    # [b]
    da_perp = dah - along.unsqueeze(-1) * a0_unit
    parts = h.split(".")
    sub = parts[-1] if len(parts) > 3 else "block"
    rows.append({
        "sub": sub,
        "along_mean": along.abs().mean().item(),
        "perp_mean": da_perp.norm(dim=-1).mean().item(),
        "frac_perp": (da_perp.norm(dim=-1) /
                      (dah.norm(dim=-1) + 1e-12)).mean().item(),
    })
df_md = pl.DataFrame(rows)
print("\nmagnitude vs direction decomposition (per sublayer)")
print("SHOULD: frac_perp ~0.5-0.95 — most of Δa is in NEW directions, not scaling existing rep. ELSE: frac_perp < 0.3 = steering is ~just a gain change; > 0.99 = no projection along a₀ at all (rare).")
print(tabulate(
    df_md.group_by("sub").agg(
        pl.col("frac_perp").mean().alias("mean_frac_perp"),
        pl.col("along_mean").mean().alias("mean_along"),
        pl.col("perp_mean").mean().alias("mean_perp"),
    ).sort("sub").to_pandas(),
    tablefmt="tsv", headers="keys", floatfmt="+.3f", showindex=False,
))

# %% [markdown]
# ## A-side cell 7: linearity test — does Δa ≈ dW @ a₀?
# If steering's effect were purely the additive write of dW into the
# residual stream (no nonlinear amplification), the activation diff at
# layer L should equal `(dW_L) @ (input to that layer)`. Cosine between
# actual Δa and dW-predicted Δa tests this for the final block output.
#
# This is informative: high cos = steering is well-described by a single
# linear write at this layer; low cos = downstream nonlinearity (LayerNorm,
# attention softmax, MLP gating) is doing most of the work.
#
# Limited to layers we have w[k] for; aligns inputs by hooking the module's
# input via TraceDict(retain_input=True) on a separate pass.

# %%
# For brevity in this first pass: skip implementation, leave as pseudocode.
# TODO: capture inputs via TraceDict(retain_input=True), compute dw @ a_in,
# compare to Δa_block_at_α=+1_vs_baseline.
print("\nlinearity test: TODO — needs retain_input=True capture; see cell docstring.")

# %% [markdown]
# ## What we did NOT analyze (deliberate scope cuts for this notebook)
# - **Polar decomposition / rotation analysis**: Qwen3 LoRA targets are all
#   rectangular (q_proj 1024->2048, k_proj 1024->256 etc.), so dW = R·S
#   isn't well-defined the way it is for square matrices. Worth coming
#   back to via SVD-of-dW vs SVD-of-W₀ shared singular vectors.
# - **Suppressed-neuron PCA**: per AntiPaSTO docs/steering_methods.qmd:67,
#   `min(Σrelu(Δmag+), Σrelu(Δmag-))` per neuron column, then PCA. Not
#   yet computed here — that's phase 2.5.
# - **Per-token Δa**: only scored at the last token. Steering may localize
#   on specific token positions (the claim words?). Easy add: drop the
#   `.gather(seq_idx)` step.

# %% [markdown]
# ## Save tables
out = Path("out/sycophancy/lora/")
df_w.write_csv(out / "analyze_w_magnitude.csv")
df_rank.write_csv(out / "analyze_w_rank.csv")
df_a.write_csv(out / "analyze_a_norms.csv")
df_md.write_csv(out / "analyze_a_magdir.csv")
logger.info(f"wrote 4 csv tables to {out}")
