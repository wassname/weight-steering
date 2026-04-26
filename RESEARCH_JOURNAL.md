# Research journal — weight-steering

## 2026-04-27 — v9 cross-adapter results: DeLoRA wins; subspace-finding methods fail

### tl;dr

- DeLoRA is the strongest behavioral steerer by a large margin (delta = +0.94
  logratio at coeff=+1 vs base, ~1.5x DoRA/PiSSA, ~2x LoRA/OFT, ~3.5x IA3).
- Every linear "find the planning subspace" method we tried lands at ~1-8%
  subspace overlap with the weight oracle. Across 6 adapter families, on every
  LoRA layer (8-21). Both cumulative and block-local act oracles. So either
  the right subspace really is small and we keep missing it, or "planning
  subspace" isn't the right frame.

### Headline numbers (cross_adapter_v9)

Behavioral steering on daily-dilemmas honesty subset (logratio_honesty, n=100):

| adapter | logratio @ -1 | @ 0 (base) | @ +1 | delta(+1 - 0) |
|---------|---------------|------------|------|---------------|
| delora  | -0.29         | 1.08       | 2.02 | **+0.94**     |
| dora    |  0.73         | 1.08       | 1.72 | +0.64         |
| pissa   |  0.44         | 1.08       | 1.69 | +0.60         |
| oft     |  1.09         | 1.08       | 1.57 | +0.49         |
| lora    |  1.09         | 1.08       | 1.55 | +0.47         |
| ia3     |  1.29         | 1.08       | 1.35 | +0.26         |

DeLoRA is the only adapter that meaningfully *de*-steers (negative coeff →
dishonest). LoRA/OFT/IA3 are nearly flat at coeff=-1.

Subspace overlap with w_oracle (mean across LoRA layers 8-21, top-PCS=8):

| adapter | act_oracle (cumul) | act_oracle (block-local v9) |
|---------|--------------------|-----------------------------|
| oft     | 0.046              | 0.045                       |
| pissa   | 0.036              | 0.042                       |
| lora    | 0.034              | 0.016                       |
| ia3     | 0.031              | 0.029                       |
| dora    | 0.024              | 0.015                       |
| delora  | 0.017              | 0.016                       |

Note the inversion: the strongest behavioral steerer (DeLoRA) has the *lowest*
subspace alignment with act_oracle. The weakest (IA3) is mid-pack on overlap.
"Subspace alignment with the activation-difference oracle" is not predictive
of behavioral effect across adapter families.

### What v9 ruled out

- **Scope mismatch**: hypothesis was that hs_diff_B[L] is cumulative
  (includes all upstream LoRA writes) while dW[L] is local, so the cumulative
  act_oracle was looking at the wrong thing. v9 added block_diff_B[L] =
  what block L itself wrote, and re-derived the oracle from that. Result:
  block-local barely moves overlap (1-5% in either direction) — sometimes up,
  sometimes down, no consistent improvement. So scope is NOT the culprit.
- **Layer L=8 sanity**: at the first LoRA layer, cumulative ≈ block (overlap
  1.0 for 5/6 adapters; IA3 fails because IA3Config doesn't accept
  layers_to_transform so it adapts every layer). So the metric is consistent;
  cumulative just diverges from block as we accumulate upstream LoRA writes.

### What this falsifies

The "shared low-rank planning subspace" frame as written in
docs/blog_adapter_as_hypothesis. If a small (rank ≤ 8) subspace contained the
honesty/sycophancy task structure, we'd expect the weight oracle and the
activation oracle (which by construction captures top-PCS energy of the
behavioral diff) to agree on at least one of: substance, scope, or family.
They don't agree on any of them, across 6 different LoRA-family inductive
biases.

Two surviving stories:

1. **The right subspace is tiny but specific** — maybe ~3% overlap is "the
   right 3%" and the 97% orthogonal part of dW is dead weight that doesn't
   affect behavior. Falsifiable: project dW onto top-K right SVs of
   act_oracle for K ∈ {1,2,4,8}, run dilemmas, see if delta_pos_minus_zero
   survives. If yes, our metric is just the wrong norm. If no, the framing
   is wrong.
2. **The frame is wrong** — behavior emerges from how dW interacts with the
   *full* activation manifold non-linearly through the rest of the network,
   not from alignment with a top-PCS basis. The act_oracle PCA captures
   variance, not function.

I lean (2). The fact that DeLoRA has the worst overlap and the best behavior
is hard to explain under (1).

### What's interesting about DeLoRA winning

DeLoRA's parametrization (decoupled magnitude + normalized direction, like
DoRA but with stronger decoupling — see Bini 2024) seems to produce a *more
swingy* steering vector: it's the only adapter where coeff=-1 actively
de-honests the model below baseline. Hypothesis: DeLoRA's normalization
forces the update to be a coherent direction rather than a magnitude-driven
blob, so scaling it ±α actually traverses the behavioral axis. Other
adapters (LoRA, OFT) collapse asymmetrically on the negative side because
much of their delta is magnitude-not-direction.

This would mean DeLoRA is a better *probe* for behavior steering even though
all six adapters land in similar regions of weight space when you measure by
subspace overlap.

### Next experiments

- (claim 1 falsifier) projection-ablation: w_proj_K = project dW onto top-K
  act_oracle dirs, run dilemmas, see if delta_pos_minus_zero is preserved.
- DeLoRA at higher rank — does the behavioral lead grow or saturate?
- DeLoRA × different behaviors (not just sycophancy/honesty) to see if the
  swinginess generalizes.
- Replace "subspace alignment" with a functional metric: causally ablate dW
  components and measure behavior change directly, not via PCA bases.

### Artifacts

- out/sycophancy/cross_adapter_v9/{headline.csv, scope_summary.csv,
  dilemmas_summary.csv, scope_bars.png}
- out/sycophancy/{lora,dora,pissa,delora,oft,ia3}/v9/v9_scope_diagnostic.csv
  — per-layer scope_diagnostic for all 6 adapters, all show 1-8% overlap on
  every LoRA layer.
- nbs/hypothesis_sweep_v9.py — block-local oracle, scope diagnostic, ADAPTER
  env var.
- nbs/cross_adapter_v9.py — aggregator across the 6 adapter families.
- All 18 pueue jobs (88-105) finished Success.
