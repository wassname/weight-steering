# Weight Steering

Fork of Fierro & Roger 2025, rebuilt on HF + PEFT + uv, targeting Qwen3-0.6B.
Method: `dW = theta_pos - theta_neg`, then add `alpha * dW` at inference.

## Results (Qwen3-0.6B, honesty, N=1, single seed)

All evals use base persona at eval time. No system prompt.

### Primary evals: AIRiskDilemmas + tiny-mfv AIRisk

The current headline evaluations are:

- **AIRiskDilemmas / Truthfulness**: guided-CoT, action-choice preference on
  1,869 labeled dilemmas from `kellycyy/AIRiskDilemmas`.
- **tiny-mfv / airisk**: fast logprob probe on 132 AI-risk vignettes from
  [`wassname/tiny-mfv`](https://huggingface.co/datasets/wassname/tiny-mfv),
  scored with dual JSON-bool prompts on `other_violate` and `self_violate`.

tiny-mfv is the cleaner fast probe here: it is cheaper, gives stable bool-mass
sanity checks, and exposes both **moral wrongness shift** and **perspective
gap** directly. AIRiskDilemmas remains the higher-variance, higher-context
complement.

### tiny-mfv AIRisk: current confirmed full run

Qwen3-0.6B, honesty `delora`, 131 joined vignettes, bootstrap `n=1000`.

| adapter | alpha | wrongness | 95% CI | gap | 95% CI |
| ------- | ----: | --------: | :----- | --: | :----- |
| delora  | -1.0  | +0.795 | [+0.764, +0.823] | +0.114 | [+0.086, +0.146] |
| base    |  0.0  | +0.423 | [+0.345, +0.501] | +0.468 | [+0.391, +0.548] |
| delora  | +1.0  | -0.350 | [-0.392, -0.308] | +0.269 | [+0.233, +0.304] |

Interpretation: on this AIRisk probe, positive `delora` steering moves strongly
away from rating the AI-risk violations as wrong, while negative steering moves
the other way. The effect is large relative to the bootstrap uncertainty, so
the sign is not ambiguous on this dataset.

### Queued full table

The repo now queues the full README refresh through `pueue`:

- 6 adapters (`ia3`, `oft`, `dora`, `lora`, `pissa`, `delora`)
- 2 datasets (`AIRiskDilemmas`, `tiny-mfv/airisk`)
- 1 final summarizer producing `out/honesty/readme_airisk_table.csv`

That summary includes baseline and adapter uncertainty.

### OOD: held-out sycophancy Yes/No claims (12 claims, alpha=+1)

Previously labeled "IID" -- corrected: these are *sycophancy* claims, but the
dW was trained on the *honesty* contrast (see [src/ws/data.py](src/ws/data.py)).
The 12 claims are also held-out from the training topics, so this is
doubly-OOD (different behavior axis + held-out topics). Reported metric is
`mean logratio = log P(Yes) - log P(No)` over the 12 claims, where Yes =
agreeing with the user's wrong belief = sycophantic = dishonest.

| adapter | mean_lr | shift vs base |
| ------- | ------: | ------------: |
| pissa   |   8.437 |        +5.708 |
| delora  |   7.198 |        +4.469 |
| lora    |   6.531 |        +3.802 |
| dora    |   6.156 |        +3.427 |
| oft     |   3.917 |        +1.188 |
| ia3     |   2.719 |        -0.010 |

`alpha=+1` makes the model say *more* Yes on these sycophancy probes -- i.e.
more sycophantic, not more honest. **This is consistent with the
AIRisk results above**: the trained dW is steering toward
*agreeableness/Yes-bias*, not honesty. Likely cause: at 0.6B, the
honest-vs-dishonest persona conditioning at data-gen time produces a
response contrast dominated by
*compliance/length/confidence* rather than truthfulness.

## How to run

```sh
# Quick sanity check (~1 min, tiny random Qwen3)
just smoke

# Full pipeline for one adapter
uv run python -m ws.replicate --model Qwen/Qwen3-0.6B --behavior honesty --adapter lora

# All adapters
uv run python -m ws.run_sweep --behavior honesty --n-personas 1 --n-samples 50

# AIRiskDilemmas
just eval-airisk adapter=delora behavior=honesty

# tiny-mfv AIRisk with bootstrap uncertainty
just eval-tinymfv-airisk adapter=delora behavior=honesty

# README-ready combined table after per-adapter runs
just summarize-airisk behavior=honesty
```

Source layout: core modules live in `src/ws/`, active benchmarks in `src/ws/eval/`, and CLI/report helpers in `src/ws/scripts/`. Outputs go to `out/<behavior>/<adapter>/`.

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
- tiny-mfv dataset: https://huggingface.co/datasets/wassname/tiny-mfv
- AIRiskDilemmas dataset: `kellycyy/AIRiskDilemmas` (HuggingFace)
- RepE baseline: `representation-engineering` (Zou et al. 2023)
- PEFT: https://github.com/huggingface/peft
