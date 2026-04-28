# Weight Steering

Fork of Fierro & Roger 2025, rebuilt on HF + PEFT + uv, targeting Qwen3-0.6B.
Method: `dW = theta_pos - theta_neg`, then add `alpha * dW` at inference.

## Results (Qwen3-0.6B, honesty, N=1, single seed)

All evals use base persona at eval time. No system prompt.

### OOD: surgical informedness on daily-dilemmas (full split, 219 dilemmas, 438 action rows)

Surgical informedness SI_k2 = fix_rate - 2 * broke_rate (penalises regressions 2x). SI_best = post-hoc sign-aligned upper bound (snooping).

| method            |  SI_k2 |  SI_k1 | SI_best | fix_rate | broke_rate |
| ----------------- | -----: | -----: | ------: | -------: | ---------: |
| prompt:engineered |  -8.88 |  -0.58 |   +4.95 |    0.149 |      0.058 |
| prompt:simple     | -16.00 |  -1.83 |   +3.46 |    0.245 |      0.203 |
| RepE all-layers   |  -6.86 |  +0.97 |   +0.79 |    0.149 |      0.070 |
| oft               |  -3.37 |  -0.21 |   +0.16 |    0.043 |      0.020 |
| ia3               |  -0.47 |  +0.26 |   -0.09 |    0.011 |      0.006 |
| dora              | -25.78 |  -6.31 |   -1.91 |    0.149 |      0.157 |
| lora              | -27.13 |  -6.88 |   -3.04 |    0.138 |      0.157 |
| pissa             | -27.27 |  -5.65 |   -9.08 |    0.160 |      0.169 |
| delora            | -34.29 |  -4.85 |  -38.12 |    0.213 |      0.410 |

Every method is negative under SI_k2. Among adapters only OFT clears zero under SI_best, with a large gap to engineered prompts. DeLoRA's broke_rate 0.41 (141/344 already-honest rows flipped) dominates.

### OOD: SI at KL-calibrated alpha (matched off-task p95 token-KL ~ 0.61 nats)

| method                   |    alpha |    SI | fix | broke | broke% |
| ------------------------ | -------: | ----: | --: | ----: | -----: |
| prompt:eng_dishonest     |    +1.00 | +5.41 |  14 |    15 |   4.4% |
| prompt:simple_dishonest  |    +1.00 | +3.57 |  12 |    15 |   4.4% |
| prompt:engineered_honest |    +1.00 | +2.62 |  14 |    20 |   5.8% |
| repe                     |    +2.30 | -5.29 |  15 |    20 |   5.8% |
| prompt:simple_honest     |    +1.00 |-13.89 |  23 |    70 |  20.3% |
| dW:oft                   |    +8.22 |-25.97 |  16 |    86 |  25.0% |
| dW:delora                |    +0.78 |-29.79 |  18 |   121 |  35.2% |
| dW:pissa                 |    +1.17 |-32.03 |  16 |    65 |  18.9% |
| dW:ia3                   |   +34.94 |-43.57 |  16 |    87 |  25.3% |
| dW:lora                  |    +2.16 |-52.72 |  19 |   133 |  38.7% |
| dW:dora                  |    +2.30 |-56.96 |  19 |   139 |  40.4% |

At matched off-task KL, all 6 adapters land deeply negative SI. Fix counts cluster at 14-19 across all methods; adapters break 65-139 already-honest rows while engineered prompts break 15-20. Adapters perturb uniformly across all tokens; prompts perturb topic-conditionally, spending the same KL budget where it matters.

### IID: held-out Yes/No claims (12 claims, alpha=+1)

| adapter | mean_lr | shift vs base |
| ------- | ------: | ------------: |
| pissa   |   8.437 |        +5.708 |
| delora  |   7.198 |        +4.469 |
| lora    |   6.531 |        +3.802 |
| dora    |   6.156 |        +3.427 |
| oft     |   3.917 |        +1.188 |
| ia3     |   2.719 |        -0.010 |

All adapters except IA3 learn the IID direction. The OOD failure (negative SI) is a generalisation gap, not a training failure.

### DeLoRA: within-tensor direction vs per-tensor norm allocation

| variant     |     SI | fix/broke @ a=+1 | mean_lr delta@a=+1 |
| ----------- | -----: | ---------------: | -----------------: |
| full        | -34.29 |          20/141  |             +0.237 |
| dir_only    | -41.00 |          20/146  |             +0.024 |
| mag_only    | -34.75 |           16/28  |             +1.068 |
| random_norm | -13.36 |           16/76  |             -0.143 |

`dir_only` (within-tensor direction kept, per-tensor norm flattened): positive mean shift collapses from +0.237 to +0.024. `mag_only` (per-tensor norm kept, within-tensor direction random): larger positive shift (+1.07) with fewer broken rows (28 vs 141). Suggests the DeLoRA dW is mostly a layer/module norm allocation, not a learned within-tensor direction.

## How to run

```sh
# Quick sanity check (~1 min, tiny random Qwen3)
just smoke

# Full pipeline for one adapter
uv run python -m ws.replicate --model Qwen/Qwen3-0.6B --behavior honesty --adapter lora

# All adapters
uv run python -m ws.run_sweep --behavior honesty --n-personas 1 --n-samples 50

# KL calibration then daily-dilemmas eval
uv run python -m ws.eval.kl_calibrate --behavior honesty
uv run python -m ws.eval.dilemmas_calibrated --behavior honesty
```

Source layout: `src/ws/{data,train,diff,steer,subspace,replicate,run_sweep}.py`, `src/ws/eval/{sycophancy,dilemmas,kl_calibrate,dilemmas_calibrated}.py`. Outputs to `out/<behavior>/<adapter>/`.

## Cite

```bibtex
@article{FierroRoger2025,
  author    = {Constanza Fierro and Fabien Roger},
  title     = {Steering Language Models with Weight Arithmetic},
  journal   = {arXiv preprint arXiv:2511.05408},
  year      = {2025},
  url       = {https://arxiv.org/abs/2511.05408},
  doi       = {10.48550/arXiv.2511.05408}
}
```

## Related

- Paper: https://arxiv.org/abs/2511.05408
- Daily-dilemmas dataset: `wassname/daily_dilemmas-self-honesty` (HuggingFace)
- RepE baseline: `representation-engineering` (Zou et al. 2023)
- PEFT: https://github.com/huggingface/peft
