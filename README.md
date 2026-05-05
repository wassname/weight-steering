# Weight Steering

Fork of [Fierro & Roger 2025](https://arxiv.org/abs/2511.05408). Train two
PEFT adapters on contrastive personas (POS vs NEG), merge into base,
take `dW = θ_pos − θ_neg`, add `α·dW` at inference.

We test whether weight-space steering (dW) competes with hidden-state
steering and prompting on a directly comparable Authority↓ benchmark.
For dataset, persona pairs, calibration recipe, and baseline methods,
see [steering-lite](https://github.com/wassname/steering-lite) (sl). ws
shares the persona pairs, vignettes, and 1-nat KL budget so rows below
drop into sl's tables. (ws = this repo; sl = steering-lite, hidden-state
steering baselines.)

## Results: Authority↓ on Qwen3.5-4B (iso-KL=1.0)

We ask three questions:

1. Does dW move Authority in the right direction?
2. Does dW beat hidden-state steering and persona-prompting?
3. Does dW have lower uncertainty than hidden-state steering?

### Glossary

- `dW = θ_pos − θ_neg`: weight-space contrast from two PEFT adapters.
- `α`: steering strength, calibrated so worst-5%-KL hits 1 nat.
- `ΔAuth`: mean change in `logit P(is_wrong)` on Authority vignettes,
  paired by (vignette, condition). Negative = Authority↓ achieved.
- `axis_Δ = −ΔAuth` (positive = correct direction, persona-aligned).
- `SI(Auth)`: bidirectional Surgical Informedness on Authority. High
  means the method moves Authority without breaking other foundations.
  Definition: [steering-lite eval](https://github.com/wassname/steering-lite#eval).
- `prompt_only`: baseline that injects the POS persona as a system
  prompt, no steering vector.

### Δlogit and uncertainty (Auth ↓ target, Care = off-target effect)

Authority is the target (move down). Care is one off-target effect:
surgical methods should leave it near zero, broadly-suppressing methods drag it
down with Authority. Full 7-foundation table in
`out/authority/.../foundations_dlogit.csv`. **Bold** = best per column
(most-negative ΔAuth, lowest std, closest-to-zero ΔCare).

| method                    | ΔAuth ↓ (mean ± std)  | ΔCare → 0 (mean ± std) |
| ------------------------- | --------------------: | ---------------------: |
| sl:engineered_prompt      |      **−2.98** ± 1.20 |            −1.64 ± 1.03 |
| sl:sspace_ablate          |           −2.89 ± 0.86 |            −2.79 ± 0.92 |
| sl:sspace                 |           −2.78 ± 0.93 |            −2.57 ± 0.90 |
| sl:angular_steering       |           −2.67 ± 0.89 |            −2.49 ± 0.84 |
| sl:cosine_gated           |           −2.08 ± 0.64 |            −1.88 ± 0.61 |
| sl:directional_ablation   |           −1.94 ± 1.22 |            −1.80 ± 1.24 |
| sl:mean_diff              |           −1.93 ± 1.11 |            −1.72 ± 1.09 |
| sl:mean_centred           |           −1.80 ± 1.17 |            −1.63 ± 1.14 |
| sl:spherical              |           −1.44 ± 0.89 |            −1.21 ± 0.71 |
| sl:pca                    |           −1.36 ± 1.50 |            −1.30 ± 1.36 |
| sl:topk_clusters          |           −1.18 ± 0.97 |            −1.12 ± 0.91 |
| ws:delora*                |           −0.89 ± **0.58** |        −0.49 ± 0.60 |
| sl:linear_act             |           −0.83 ± 0.67 |            −0.70 ± 0.52 |
| sl:chars                  |           −0.45 ± 0.61 |        **−0.40** ± 0.54 |

*ws:delora calibrated at p95=0.5, not kl=1.0 — expect larger effect after re-calibration.

### Surgical Informedness (headline, ↑ better)

`SI(Auth)`, `SI_fwd`, `SI_rev`, `Auth_sep`, and `pmass²×100` all higher is
better. **Bold** = best in column. sl rows from sl's published Qwen3.5-4B
run. ws:delora is at p95=0.5 budget (kl=1.0 re-run queued with lora/dora).

| method                    | SI(Auth) ↑ | SI_fwd ↑ | SI_rev ↑ | Auth_sep ↑ | pmass²×100 ↑ |
| ------------------------- | ---------: | -------: | -------: | ---------: | -----------: |
| sl:directional_ablation   |  **52.90** |     0.32 |    +1.00 |      +2.05 |         80.1 |
| sl:super_sspace           |      47.71 |     0.67 |    +0.40 |      +1.99 |         88.8 |
| sl:sspace                 |      45.67 |     0.64 |    +0.85 |      +0.69 |         61.0 |
| sl:mean_diff              |      32.81 |     0.34 |    +1.00 |      +1.65 |         49.0 |
| sl:mean_centred           |      32.72 |     0.29 |    +1.00 |      +1.56 |         50.6 |
| sl:topk_clusters          |      31.34 |     0.13 |    +0.72 |      +1.55 |         73.9 |
| sl:sspace_ablate          |      24.11 | **0.74** |    +0.02 |      +0.59 |         63.6 |
| sl:linear_act             |      20.24 |    −0.19 |    +1.00 |      +0.83 |         49.9 |
| ws:delora                 |      19.03 |     0.02 |    +0.37 |      +0.76 |     **99.9** |
| sl:engineered_prompt      |      17.36 |     0.50 |    −0.02 |      +1.90 |         71.7 |
| sl:cosine_gated           |       8.92 |     0.09 |    +1.00 |  **+2.00** |         16.4 |
| sl:angular_steering       |       7.00 |     0.55 |    −0.38 |      +0.32 |         80.6 |
| sl:spherical              |       4.98 |     0.16 |      n/a |      +0.85 |         30.3 |
| sl:pca                    |      −0.92 |     0.03 |    −0.08 |      +0.85 |         39.0 |
| sl:chars                  |      −9.16 |    −0.26 |    +0.00 |      +0.50 |         68.3 |

### TL;DR

1. **Did dW replicate?** Yes. ws:delora ΔAuth = −0.89 (sign correct) and
   SI(Auth) = 19.03 — verdicts do flip in the right direction.
2. **Did dW beat steering and prompting?** Partially. SI = 19.03 beats
   the engineered-prompt baseline (17.36) and 5 other sl methods, but is
   below 8 hidden-state methods. ΔAuth std = 0.58 is the lowest in the
   table (lower uncertainty than all sl methods).
3. **Did dW have lower uncertainty?** Yes. ws:delora std = **0.58**,
   lowest in the table (sl best: chars 0.61).

Open: lora and dora training queued (pueue 141-144); ws:delora is at
p95=0.5 budget, not yet at sl's kl=1.0 — expect SI to shift after
re-calibration. Full 4-adapter table pending.

## How to run

```sh
# 1. generate persona-conditioned data
uv run python -m ws.data --behavior authority --model-id Qwen/Qwen3.5-4B
# 2. train all adapters (dW = merged_pos - merged_neg)
uv run python -m ws.run_sweep --behavior authority --model Qwen/Qwen3.5-4B
# 3. iso-KL calibrate α
uv run python -m ws.kl_calibrate --behavior authority --model Qwen/Qwen3.5-4B
# 4. eval on tinymfv airisk
uv run python -m ws.scripts.eval_tinymfv_calibrated --behavior authority --model Qwen/Qwen3.5-4B
# 5. rebuild README tables
uv run python -m ws.scripts.readme_tinymfv_table --behavior authority
```

Outputs go to `out/authority/<adapter>/`. Smoke test on a tiny model:
`just smoke`.

## Cite

```bibtex
@article{FierroRoger2025,
  author    = {Constanza Fierro and Fabien Roger},
  title     = {Steering Language Models with Weight Arithmetic},
  journal   = {arXiv preprint arXiv:2511.05408},
  year      = {2025},
  url       = {https://arxiv.org/abs/2511.05408}
}
```

## Related

- [steering-lite](https://github.com/wassname/steering-lite): hidden-state steering, sister project, source of all baseline rows above
- [tinymfv](https://github.com/wassname/tinymfv): vignette dataset
- [PEFT](https://github.com/huggingface/peft): adapter library
- [RepE](https://github.com/andyzoujm/representation-engineering) (Zou et al. 2023): hidden-state steering precursor
