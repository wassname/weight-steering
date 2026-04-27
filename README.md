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

| stage                                | pos uses              | neg uses                  | how                              |
| ------------------------------------ | --------------------- | ------------------------- | -------------------------------- |
| adapter training data generation     | `POS[0..4]`           | `NEG[0..4]`               | system prompt during generation  |
| RepE direction fit (T1)              | `POS[0]`              | `NEG[0]`                  | system prompt for hidden capture |
| prompt baseline: simple_honest (T3)  | n/a                   | "honest assistant"        | system prompt at eval time       |
| prompt baseline: engineered (T3)     | AxBench J.2 honest    | AxBench J.2 dishonest     | system prompt at eval time       |
| daily-dilemmas eval                  | n/a                   | n/a                       | base persona, no system prompt   |

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

### Adapter comparison

<!-- source: out/honesty/cross_adapter_full_dd/dilemmas_summary.csv -->
Daily-dilemmas honesty eval, honesty-axis training, base persona, full split
(438 rows / coeff). `delta` = `mean_logratio_honesty` at `α=+1` minus `α=0`;
larger means more honest. `pmass` = p(Yes) + p(No) sanity check.

| adapter | delta `α=-1` | `α=0` logratio | delta `α=+1` | pmass @ `+1` | read                              |
| ------- | -----------: | -------------: | -----------: | -----------: | --------------------------------- |
| delora  |       -1.152 |           1.33 |       +0.237 |        0.971 | strongest steerer, both signs     |
| lora    |       -0.222 |           1.33 |       +0.077 |        0.912 | modest but clean                  |
| oft     |       -0.111 |           1.33 |       +0.055 |        0.928 | weaker                            |
| pissa   |       -0.480 |           1.33 |       +0.042 |        0.877 | strong negative, weak positive    |
| ia3     |       -0.032 |           1.33 |       +0.030 |        0.937 | near no-op positive               |
| dora    |       -0.170 |           1.33 |       +0.016 |        0.915 | near no-op positive               |

Takeaway: DeLoRA has the strongest positive steering at `α=+1` (+0.237).
PiSSA and DeLoRA both have larger magnitude at negative `α`, showing
asymmetric effectiveness. IA3 and DoRA are near no-ops at `α=+1` under
honesty-axis training.

### Baselines vs weight steering

<!-- weight rows: out/honesty/cross_adapter_full_dd/dilemmas_summary.csv -->
<!-- RepE row:    out/honesty/activation_baseline/summary.csv -->
<!-- prompt rows: out/honesty/prompt_baseline/summary.csv -->
Same daily-dilemmas split, 438 rows, base persona, full 219 dilemmas,
honesty-axis training. `dd_delta` = honesty logratio change vs `base @ α=0`.

| method                          | `dd_delta` | config             |
| ------------------------------- | ---------: | ------------------ |
| weight steer: `dW:delora`       |     +0.237 | `α=+1`             |
| weight steer: `dW:lora`         |     +0.077 | `α=+1`             |
| RepE (repeng, all-layers)       |     +0.050 | layer=-1, `α=+1`   |
| prompt: engineered (dishonest)  |     +0.049 | system prompt      |
| prompt: engineered (honest)     |     +0.045 | system prompt      |
| weight steer: `dW:oft`          |     +0.055 | `α=+1`             |
| prompt: simple honest           |     -0.520 | system prompt      |

Read: weight steering (DeLoRA) is the only intervention that shifts
honesty by more than 0.1 log-ratio units. RepE and the engineered prompts
are comparable to each other (+0.05). The "simple honest" system prompt
reliably makes the model less honest on this eval. T4 multiseed and T5
Gemma will test whether the dW vs RepE gap survives different seeds and model.

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
