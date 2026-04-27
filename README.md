# Weight Steering

> **Fork notice (wassname, 2026-04):** this is a working fork that strips the
> upstream Axolotl + vLLM + Anthropic-batch-API stack and rebuilds the core
> method on HF + PEFT + uv, targeting Qwen3-0.6B for cheap iteration. Goals:
> (1) replicate `w = θ⁺ − θ⁻` on a small model, (2) test alignment of `w` with
> SVD subspaces of the pretrained `W` and the AntiPaSTO subspaces, (3) compare
> adapter families (LoRA / DoRA / PiSSA-init / DeLoRA) under the
> "adapter as hypothesis" framing, (4) eval on daily-dilemmas.
>
> Pipeline (see `justfile`):
> ```
> just smoke           # full pipeline on tiny-random qwen3 + BEARTYPE=1, ~1 min
> just replicate       # data → train pos → train neg → diff → eval → subspace
> just subspace-align  # phase 2: SVD top-k + weak-readout alignment table
> just adapter-sweep   # phase 3: LoRA / DoRA / PiSSA / DeLoRA sweep
> just eval-dilemmas   # phase 4: daily-dilemmas Yes/No logratio
> ```
> Source layout: `src/ws/{data,train,diff,steer,subspace,replicate,run_subspace,run_sweep}.py`,
> `src/ws/eval/{sycophancy,dilemmas}.py`. Outputs to `out/<behavior>/<adapter>/`.
>
> Scope. Not a strict replication. Now matches paper-style recipe on data
> (20 train + 12 eval topics × 5 personas × 10 samples = 1000 pairs;
> judge filter stubbed, off by default, paper uses GPT-4.1-mini) and
> current PEFT hyperparams (rank 32 / LoRA α 64 / lr 2e-4 / warmup 5 /
> wd 0.01 / seed 0 / one epoch).
> Deliberate divergences from upstream: no quantized base loading
> (DoRA/PiSSA/DeLoRA support is uncertain; bf16 fits at 0.6B), no
> `modules_to_save` for `embed_tokens` / `lm_head`, and a layer slice
> (LoRA on layers 30%-80%, steering-locus literature) instead of full
> coverage. The contrastive `θ⁺ − θ⁻` core is preserved.
>
> **Initial finding on Qwen3-0.6B.** Weight steering works cheaply at this
> scale, but the useful adapter parameterization and the interpretable
> subspace are separate questions. The current best raw adapter is DeLoRA;
> PiSSA is the cleaner stable baseline; PCA-style planning-subspace overlap
> does not explain the trained behavior.

## Current internal findings (N=1; exploratory)

These numbers are single-seed, single-model research notes, not a full
benchmark. All rows below use `Qwen/Qwen3-0.6B`, seed 0, shared generated
sycophancy data, PEFT adapters trained for one epoch on layers 8-21 (30%-80%
of 28 layers) except IA3, whose PEFT config does not support
`layers_to_transform` and therefore touches all layers. Target modules for
LoRA-family adapters are `q/k/v/o/gate/up/down_proj`.

### What was measured

- Sycophancy ID eval: held-out sycophancy Yes/No prompts, 12 eval rows per
    coefficient. Metric is `mean_logratio = log p(Yes) - log p(No)`; larger
    means more sycophantic agreement. `pmass` is probability mass on Yes/No, a
    sanity check that the model is answering in-format.
- Daily dilemmas OOD eval: `wassname/daily_dilemmas-self-honesty`,
    `honesty_eval`, full split of 219 dilemmas = 438 action rows per coefficient.
    Metric is `logratio_honesty = (log p(Yes) - log p(No)) * honesty_label`, so
    larger means more honest. Tables below use base persona only. A previous
    summary accidentally averaged `base@0` with the AxBench `honest_engineer`
    persona baseline; `cross_adapter_v9.py` now reads `dilemmas_per_row.csv` and
    filters `persona == "base"`.
- Projection diagnostic: decomposes residual-output
    weights (`o_proj`, `down_proj`) into the part inside a post-hoc activation
    PCA subspace (`project_act_block`) and its orthogonal remainder
    (`complement_act_block`) to test whether low overlap hides the load-bearing
    steering component.

### Adapter comparison

Sycophancy in-distribution steering:

| adapter | spread `α=+2 minus -2` | delta `α=+1 minus 0` | min pmass | read                                  |
| ------- | ---------------------: | -------------------: | --------: | ------------------------------------- |
| delora  |                 +23.85 |                +9.80 |     0.788 | strongest raw, but saturates at `α=2` |
| pissa   |                 +17.40 |                +6.00 |     0.999 | strongest clean/stable baseline       |
| dora    |                  +9.76 |                +2.64 |     1.000 | decent                                |
| oft     |                  +7.24 |                +1.99 |     1.000 | weaker                                |
| lora    |                  +4.09 |                +1.00 |     1.000 | weak in this run                      |
| ia3     |                  +0.86 |                +0.26 |     1.000 | near no-op                            |

Daily-dilemmas OOD honesty transfer, base persona only, full split (438 rows / coeff):

| adapter | `α=-1` | `α=0` | `α=+1` | delta `+1 minus 0` | pmass @ `+1` |
| ------- | -----: | ----: | -----: | -----------------: | -----------: |
| delora  |  -0.31 |  1.33 |   2.04 |              +0.71 |        0.942 |
| dora    |  +0.75 |  1.33 |   1.73 |              +0.40 |        0.941 |
| pissa   |  +0.45 |  1.33 |   1.69 |              +0.37 |        0.980 |
| oft     |  +1.10 |  1.33 |   1.56 |              +0.24 |        0.931 |
| lora    |  +1.09 |  1.33 |   1.55 |              +0.23 |        0.933 |
| ia3     |  +1.30 |  1.33 |   1.36 |              +0.03 |        0.937 |

Takeaway: DeLoRA is the best raw steerer on both sycophancy and daily
dilemmas. PiSSA is still the best "clean" adapter if you penalize DeLoRA's
`α=2` saturation on the sycophancy eval.

### Baselines

- T1 activation steering (RepE-style): best dd_delta = +0.071 at layer 9, `α=-4`
    (`out/sycophancy/activation_baseline/summary.csv`). Roughly comparable to
    the ia3 weight-steerer (+0.03), which is essentially a no-op; the
    structurally meaningful weight-steered adapters (lora/oft/dora/pissa/delora)
    range +0.23 to +0.71, all several times stronger than RepE on these rows.
- T3 prompt baseline (AxBench-style engineered prompt): rerun in flight
    (pueue 191), see `out/sycophancy/prompt_baseline/summary.csv` when done.

### Subspace/projection lesson

The original question was: can we find the subspace or parameterization that
explains the difference between the positive and negative LoRAs? So far we
tested three kinds of explanations:

- Parameterization: LoRA / DoRA / PiSSA / DeLoRA / OFT / IA3. Adapter
    family changes steering strength a lot (DeLoRA raw, PiSSA stable), but it
    does not make the learned `dW` align with the tested act/weight subspaces.
- Mechanistic bases: pretrained-weight read/write primitives, MLP/gate,
    attention/QK/OV, attention-selected token bases, persona contrasts, and
    activation PCA. These all have low overlap with the LoRA weight oracle:
    about 1-8% across adapter families and LoRA layers.
- Block-local activation PCA did not rescue this. The issue is not just that
    cumulative activations mix upstream layers.
- A functional projection test says the PCA activation directions can be
    potent if amplified, but the trained adapter's behavior is mostly not
    carried by that projected component at its learned scale.

Projection diagnostic at K=32 on daily dilemmas (40 dilemmas / 80 rows; this
is an ablation, not a full benchmark):

| adapter | full Δ | residual-write Δ | raw projection / residual | normmatched projection / residual | complement / residual | read                                              |
| ------- | -----: | ---------------: | ------------------------: | --------------------------------: | --------------------: | ------------------------------------------------- |
| delora  | +0.628 |           +0.844 |                      0.07 |                              0.30 |                  0.89 | trained behavior mostly outside act-PCA subspace  |
| pissa   | +0.373 |           +0.242 |                      0.47 |                              1.14 |                  0.64 | mixed: act-PCA is functional, not sole carrier    |
| oft     | +0.216 |           +0.148 |                     -0.01 |                              1.57 |                  0.69 | act-PCA direction potent only after amplification |

Here `complement` means the residual-output part of `dW` after removing the
activation-PCA subspace:

$$dW_{\text{complement}} = (I - P_{\text{act},K}) dW.$$

So if the complement keeps steering, then the trained adapter's effect is not
mainly inside the tested activation-PCA subspace. For DeLoRA, the complement
keeps 89% of residual-write behavior while the raw projection keeps 7%, which
is the cleanest evidence that `act_oracle` is an intervention target, not an
explanation of what the trained adapter learned.

Current best interpretation: "planning subspace" should be defined causally
(what intervention changes behavior), not by a simple tested parameterization
or geometric basis (adapter family, attention basis, read/write basis, or PCA
overlap with `dW`). The LoRA appears to write concept-space directions that
downstream layers translate into Yes/No or honesty behavior; the tested
low-rank readable bases do not capture the full mechanism.

# Cite

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
