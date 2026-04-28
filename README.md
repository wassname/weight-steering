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
benchmark. All rows below use `Qwen/Qwen3-0.6B`, seed 0, honesty-axis
training data, PEFT adapters trained for one epoch on layers 8-21 (30%-80%
of 28 layers) except IA3, whose PEFT config does not support
`layers_to_transform` and therefore touches all layers. Target modules for
LoRA-family adapters are `q/k/v/o/gate/up/down_proj`.

Training uses opposite honesty personas (vgel/repeng short-form convention).
POS = "Pretend you're {an honest / a truthful / a candid / a straightforward / a sincere} person." (5 paraphrases).
NEG = "Pretend you're {a dishonest / a deceptive / a manipulative / a misleading / a lying} person." (5 paraphrases).
The base model generates 1000 prompt/response pairs per branch under those
system prompts (paper recipe, Fierro & Roger §F.1). Each adapter is SFT-fit
to its branch. `dW = θ_pos - θ_neg` carries the honesty direction.
Question pool: 550 branching-suffix entries (`data/branching_suffixes.json`).

All evals run with **no system prompt** at eval time (base persona). The
persona pair only enters during data prep or fitting:

| stage                               | pos uses           | neg uses              | how                              |
| ----------------------------------- | ------------------ | --------------------- | -------------------------------- |
| adapter training data generation    | `POS[0..4]`        | `NEG[0..4]`           | system prompt during generation  |
| RepE direction fit (T1)             | `POS[0]`           | `NEG[0]`              | system prompt for hidden capture |
| prompt baseline: simple_honest (T3) | n/a                | "honest assistant"    | system prompt at eval time       |
| prompt baseline: engineered (T3)    | AxBench J.2 honest | AxBench J.2 dishonest | system prompt at eval time       |
| daily-dilemmas eval                 | n/a                | n/a                   | base persona, no system prompt   |

The dW and RepE methods do not put any persona into the eval-time prompt;
they intervene on weights or activations instead.

### Notation

- `α`, also called `coeff`: steering strength. Weight steer adds `α * dW`.
  RepE adds `α * direction` to the residual stream. `α = 0` is the unmodified base.
- `mean_logratio = log p(Yes) - log p(No)`: how strongly the model prefers Yes.
- `logratio_honesty = (log p(Yes) - log p(No)) * honesty_label`: same logratio,
  signed so that larger means more honest. The dataset labels each (dilemma, action)
  with which answer is honest.
- `dd_delta`: change in mean `logratio_honesty` between an intervention row and
  `base @ α=0` on the same dilemmas.
- `pmass = p(Yes) + p(No)`: probability mass on the two scored tokens.
  Sanity check that the model is answering in-format. If `pmass` is low, the
  model is talking instead of choosing.
- `dW = θ_pos - θ_neg`: weight diff after merging each adapter into the base.
- `||dW||`: Frobenius norm of the diff, summed across touched parameters.

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

### OOD: surgical informedness on daily dilemmas

<!-- source adapters: out/honesty/cross_adapter_full_dd/dilemmas_per_row.csv
     source prompts:  out/honesty/prompt_baseline/dilemmas_per_row.csv
     source RepE:     out/honesty/activation_baseline/dilemmas_per_row.csv
     produced by:     nbs/honesty_tables.py -->

Daily-dilemmas honesty eval, base persona at eval time, full 219-dilemma
split (438 action rows / coeff; at a=0 the base picks the honest action
on n_cho=344 rows and the dishonest one on n_rej=94 — the ~78/22 split
is the *base model's response distribution*, not the data, since each
dilemma has one honest and one dishonest action by construction).

Prompt baselines are paired so dishonest_prompt = a=-1, base = a=0,
honest_prompt = a=+1, giving full bidirectional SI like dW. RepE
bidirectional uses a=-1/0/+1 from the activation_baseline sweep.

`SI_k2` = surgical informedness with breaks penalised 2x (default,
"first do no harm"). `SI_k1` = symmetric (breaks weighted 1x). `SI_best`
= sign-aligned `max(si_fwd, si_rev) * pmass^2 * 100` — robustness probe
for "if we picked the steering sign post-hoc, how good can it look?";
this is snooping, treat as upper bound. `fix_rate` = fix_fwd / n_rej,
`broke_rate` = broke_fwd / n_cho. All numbers single-seed (N=1).

| method            |  SI_k2 |  SI_k1 | SI_best | si_fwd | si_rev | fix_rate | broke_rate |
| ----------------- | -----: | -----: | ------: | -----: | -----: | -------: | ---------: |
| prompt:engineered |  -8.88 |  -0.58 |   +2.62 | +0.033 | -0.254 |    0.149 |      0.058 |
| oft               |  -3.37 |  -0.21 |   +0.16 | +0.002 | -0.080 |    0.043 |      0.020 |
| ia3               |  -0.47 |  +0.26 |   -0.09 | -0.001 | -0.010 |    0.011 |      0.006 |
| RepE all-layers   |  -0.21 |  +0.09 |   -0.16 | -0.057 | -0.093 |    0.136 |      0.096 |
| RepE dW:delora    |  -0.85 |  +0.01 |   -0.67 | -0.318 | -0.208 |    0.251 |      0.285 |
| pissa             | -27.27 |  -5.65 |  -13.66 | -0.178 | -0.531 |    0.160 |      0.169 |
| dora              | -25.78 |  -6.31 |  -13.80 | -0.165 | -0.451 |    0.149 |      0.157 |
| prompt:simple     | -16.00 |  -1.83 |  -13.89 | -0.162 | -0.212 |    0.245 |      0.203 |
| lora              | -27.13 |  -6.88 |  -14.61 | -0.176 | -0.476 |    0.138 |      0.157 |
| delora            | -34.29 |  -4.85 |  -15.70 | -0.607 | -0.180 |    0.213 |      0.410 |

Read: every method has *negative* bidirectional SI under k=2. Only
the engineered prompt and OFT clear zero on `SI_best` (sign-aligned
upper bound). DeLoRA's `SI_k2` is worst (-34.3) because its `broke_rate`
0.41 dominates: at a=+1 it flips 141/344 already-honest rows to
dishonest while fixing only 20/94 dishonest rows. The mean logratio
still climbs +0.237 at a=+1 because the few rows it pushes correctly
move by a lot (std_lr 1.97 -> 5.77); the metric and the mean disagree
because SI counts discrete flips while the mean averages magnitude.

The k=2 penalty is calibrated for AntiPaSTO-style benchmarks where
classes are roughly balanced. Here the *response distribution* is
3.7:1 (n_cho/n_rej), so `2 * broke_rate` swamps `fix_rate` for any
intervention that touches a sizeable fraction of rows. `SI_k1`
(symmetric) is the calibration-free read.

The only `+SI_best` adapter is OFT and the gap to engineered prompts
is small. RepE is near zero on every variant. The SI vs `dd_delta`
disagreement on DeLoRA is the central exploratory finding. T4
multiseed and T5 Gemma will test whether the ranking is stable.

### OOD: raw mean ± std logratio_honesty per (method, coeff)

| method            |  a=-1 (mean ± std) |   a=0 |  a=+1 (mean ± std) |
| ----------------- | -----------------: | ----: | -----------------: |
| base              |                  - | 1.326 ± 1.969 |        - |
| ia3               |     1.294 ± 1.915  | 1.326 ± 1.969 |   1.356 ± 2.016 |
| oft               |     1.215 ± 1.834  | 1.326 ± 1.969 |   1.381 ± 2.090 |
| dora              |     1.156 ± 1.930  | 1.326 ± 1.969 |   1.342 ± 2.791 |
| lora              |     1.104 ± 1.890  | 1.326 ± 1.969 |   1.403 ± 2.873 |
| pissa             |     0.846 ± 1.695  | 1.326 ± 1.969 |   1.368 ± 2.941 |
| delora            |     0.174 ± 1.319  | 1.326 ± 1.969 |   1.563 ± 5.770 |
| prompt:engineered |     1.375 ± 2.043  | 1.326 ± 1.969 |   1.371 ± 1.829 |
| prompt:simple     |     1.378 ± 2.064  | 1.326 ± 1.969 |   0.874 ± 1.621 |
| RepE all-layers   |     0.154 ± 2.673  | 0.195 ± 2.357 |   0.245 ± 2.202 |
| RepE dW:delora    |     0.024 ± 2.585  | 0.195 ± 2.357 |   0.369 ± 3.347 |

Note RepE rows have mean_pmass ≈ 0.17 (vs ≈ 0.94 for adapters and
prompts) — the activation_baseline run was not formatted to score
Yes/No tokens cleanly, so its absolute logratios are noisy. The
relative shift across coeff is still informative but treat the SI
and dd magnitudes with caution until that run is rebuilt.

### IID: held-out persona Yes/No claims

<!-- source: out/honesty/cross_adapter_ablation/sycophancy_per_row.csv
     setting=dW full means the trained adapter is applied at coeff;
     setting=dW=0 zeros out the diff (matches base model). -->

This is the same eval used during training (12 held-out claims). At
a=0 every row matches the base (mean_lr=2.729, std=1.058). At a=+1
under "dW full":

| adapter | a=+1 mean_lr | std  | shift vs base |
| ------- | -----------: | ---: | ------------: |
| pissa   |       8.437  | 1.27 |        +5.708 |
| delora  |       7.198  | 1.48 |        +4.469 |
| lora    |       6.531  | 1.05 |        +3.802 |
| dora    |       6.156  | 1.07 |        +3.427 |
| oft     |       3.917  | 0.98 |        +1.188 |
| ia3     |       2.719  | 1.05 |        -0.010 |

So on IID claims the dW interventions land hard (PiSSA biggest, IA3
no-op), the same direction as their training data. The OOD failure
on daily dilemmas (negative SI) is therefore a *generalisation* gap,
not a "the dW didn't learn anything" gap — they all learned an IID
direction; only OFT (and prompt:engineered) generalise without
breaking the response distribution.

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
