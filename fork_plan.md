# Fork plan: weight steering benchmark + analysis

Updated: 2026-04-27

## Goal

Test whether weight steering is a useful method, and if it is, understand what part of the learned weight delta carries the behavior.

Two questions are intentionally separated:

1. **Benchmark question:** Does weight steering beat simple alternatives such as prompting and activation steering on sycophancy and daily-dilemmas honesty transfer?
2. **Analysis question:** If weight steering works, can the learned delta $dW = \theta^+ - \theta^-$ be factorized into a simpler causal intervention: a cross-adapter shared subspace, module, low-rank component, or adapter parameterization?

## Context

This is a fork of Anthropic's weight-steering method. Original recipe: train one positive adapter and one negative adapter, merge each adapter into base-weight deltas, then steer with:

$$dW = \Delta W_{pos} - \Delta W_{neg}.$$

This repo removes Axolotl/vLLM/API orchestration and rebuilds the method in HF + PEFT + uv for cheap iteration on small models.

Current main model: `Qwen/Qwen3-0.6B`.

Current behavior: sycophancy training, evaluated on sycophancy Yes/No and `wassname/daily_dilemmas-self-honesty`.

## Links

- Paper / blog:
  - [docs/weight_steering_paper.md](docs/weight_steering_paper.md)
  - [docs/weight_steer_blog.md](docs/weight_steer_blog.md)
- Adapter-as-hypothesis notes:
  - [docs/blog_adapter_as_hypothesis/README.md](docs/blog_adapter_as_hypothesis/README.md)
- Steering/subspace concepts:
  - [docs/AntiPaSTO_concepts/README.md](docs/AntiPaSTO_concepts/README.md)
- Current user-facing summaries:
  - [README.md](README.md)
  - [RESEARCH_JOURNAL.md](RESEARCH_JOURNAL.md)
- Key code:
  - [src/ws/data.py](src/ws/data.py)
  - [src/ws/train.py](src/ws/train.py)
  - [src/ws/diff.py](src/ws/diff.py)
  - [src/ws/steer.py](src/ws/steer.py)
  - [src/ws/eval/sycophancy.py](src/ws/eval/sycophancy.py)
  - [src/ws/eval/dilemmas.py](src/ws/eval/dilemmas.py)
  - [nbs/cross_adapter_v9.py](nbs/cross_adapter_v9.py)
  - [nbs/functional_projection_v10.py](nbs/functional_projection_v10.py)

## Current facts

- Daily-dilemmas default is **not full split**. Default `n_dilemmas=100` means first 100 dilemmas = 200 rows, balanced 100 honest-label and 100 dishonest-label actions.
- Full `honesty_eval` test split is 219 dilemmas = 438 rows.
- The daily-dilemmas eval uses all rows for selected dilemmas, then sign-flips by `honesty_label`; it is not only honest rows.
- Current headline tables are single-seed Qwen3-0.6B exploratory results.
- DeLoRA is best raw steering so far. PiSSA is the cleaner stable baseline if penalizing DeLoRA saturation at high alpha.
- v9/v10 do **not** prove “no subspace.” They show the trained behavior is not explained by the tested low-rank residual-stream bases or adapter-family parameterization at trained scale.
- The highest-value analysis test is cross-adapter causal ablation: if LoRA / DoRA / PiSSA / DeLoRA / OFT share a causal low-rank `dW` core, that is the clean planning-subspace result; if not, it is the cleanest negative result for the shared-subspace hypothesis.

## Done

- [x] Clean repo into uv + HF + PEFT small-model workflow.
- [x] Make Qwen3-0.6B sycophancy steering work end-to-end.
- [x] Hook in LoRA, DoRA, PiSSA, DeLoRA, OFT, and IA3 adapter families.
- [x] Build sycophancy logratio eval with coefficient sweep.
- [x] Build daily-dilemmas honesty eval with sign-flipped Yes/No logratio.
- [x] Run single-seed Qwen adapter benchmark on sycophancy and 100-dilemma DD default.
- [x] Fix DD cross-adapter aggregation to use base-only coeff=0 rather than mixing persona baselines.
- [x] Run v9 subspace/scope diagnostics: weight oracle, cumulative activation oracle, block-local activation oracle, first-LoRA-layer sanity checks.
- [x] Run v10 projection/complement falsifier: raw activation projection, complement, and normmatched projection.
- [x] Update README and research journal with corrected DD table and conservative interpretation.

## TODO: benchmark question

- [ ] **Goal: activation-steering baseline on the same DD rows.**
  - Why: RepE/repeng is the most threatening baseline; if it matches or beats `dW`, the method story weakens before adapter seeds matter.
  - Do: train representation direction on the same sycophancy contrast; grid layer x coefficient; evaluate sycophancy and full DD.
  - UAT: best activation-steering row is selected by held-out sycophancy or validation DD, then reported beside best `dW` on identical DD test rows.
  - Verify: table includes `method=repeng`, `layer`, `coeff`, `syc_delta`, `dd_delta`, `pmass`, and the same `idx` set as the `dW` rows.
  - Negative outcome -> claim: if repeng matches/beats `dW`, write "activation steering is the simpler baseline; weight steering needs a stronger reason to exist."

- [ ] **Goal: full daily-dilemmas benchmark for current Qwen adapters.**
  - Why: current DD table uses first 100 dilemmas, not the full 219-dilemma split.
  - Do: re-run LoRA / PiSSA / DeLoRA / DoRA / OFT / IA3 with `--n-dilemmas 219`.
  - UAT: table has 438 base rows per coeff before persona baselines, and reports `pmass`, `frac_low_pmass`, `delta(+1 - 0)`.
  - Verify: `out/sycophancy/cross_adapter_full_dd/dilemmas_summary.csv` exists and includes `n_base_rows_per_coeff=438`.

- [ ] **Goal: prompt baselines on the same DD rows.**
  - Why: weight steering is only interesting if it beats “just prompt it.”
  - Do: evaluate base, simple honest persona, and engineered AxBench-style prompt.
  - UAT: one table compares `base`, `simple_honest_prompt`, `engineered_prompt`, and best `dW` on identical rows.
  - Verify: `prompt_baseline_delta` and `weight_steer_delta` are computed from the same `idx` set.
  - Negative outcome -> claim: if prompting matches/beats `dW`, write "prompting is the simpler intervention for this behavior/eval pair."

- [ ] **Goal: multi-seed adapter benchmark on Qwen.**
  - Why: current adapter ranking is N=1 seed.
  - Do: run seeds 0, 1, 2 for LoRA / PiSSA / DeLoRA first; add DoRA/OFT only if cheap.
  - UAT: table reports mean +/- std for sycophancy and DD deltas, plus seed-level signs, so a reader can tell stable ranking from noisy N=1 luck.
  - Verify: each adapter has exactly three `w.pt` files and three eval summaries; ranking table includes `n_seeds=3`, `mean_dd_delta`, `std_dd_delta`, and `sign_agreement`.
  - Negative outcome -> claim: if adapter ranking changes across seeds or error bars overlap heavily, write "single-seed adapter winner is unstable; do not claim a family ranking yet."

- [ ] **Goal: Gemma 1B replication.**
  - Why: check whether DeLoRA/PiSSA ranking is Qwen-specific.
  - Do: train LoRA / PiSSA / DeLoRA on Gemma 1B, seed 0, full DD split.
  - UAT: compare Gemma ranking to Qwen ranking with the same metrics.
  - Verify: table has model column with `Qwen3-0.6B` and `Gemma-1B`; if DeLoRA remains best, expand seeds; if rankings diverge, write that up as a model-specific adapter-basin finding.

## TODO: analysis question

- [ ] **Goal: cross-adapter causal-ablation table for `dW` bases.**
  - Why: this is the headline analysis experiment. It tests whether different adapter families discovered the same causal planning subspace or different basins.
  - Do: one notebook builds candidate bases `B`, computes `dW_keep_B` and `dW_drop_B`, and evaluates both on sycophancy + full DD for each adapter. This single table replaces separate layer-ablation, SVD top/tail, read/write, MLP, and magnitude/direction experiments.
  - Candidate `B` rows:
    - `shared_SVD_K8/K32/K64`: stack residual-output `dW` from LoRA / DoRA / PiSSA / DeLoRA / OFT per layer/tensor, take top-K SVs.
    - `top8/top32_per_adapter` and `tail_per_adapter`: per-adapter SVD split of each tensor.
    - `write`, `write_not_read`, `super_read [q,k,v,up,gate]`, `super_write [o,down]`.
    - `mlp_down`, `mlp_up`, `mlp_gate`, `mlp_up+gate`, `attn_only`.
    - `magnitude`, `direction/rotation` for DeLoRA / DoRA / OFT where mathematically defined.
    - `layers_8..21_only`, leave-one-layer-out, and `random_null`.
  - UAT: one central table has every ablation family as rows, with columns `ablation_family`, `candidate_B`, `adapter`, `rank`, `retain_keep`, `retain_drop`, `syc_delta_keep`, `dd_delta_keep`, `syc_delta_drop`, `dd_delta_drop`, `pmass`.
  - Verify: the single table contains keep/drop rows for every `ablation_family`: `shared_svd`, `per_adapter_svd`, `read_write`, `mlp_first_order`, `magnitude_direction`, `layer`, and `random_null`; `layer` includes both `layers_8..21_only` and leave-one-layer-out rows; `keep_B_shared_K32` and `drop_B_shared_K32` are both evaluated for at least LoRA / DoRA / PiSSA / DeLoRA / OFT; random null retention is near rank/d; each row uses the same eval rows and coefficient grid.
  - Positive outcome -> claim: if `keep_B_shared` retains >=0.7x behavior across adapters and `drop_B_shared` removes it, write the adapter-invariant planning-subspace paper.
  - Negative outcome -> claim: if `keep_B_shared` retains <0.3x even at K=64 while complements/tails retain behavior, write the shared-subspace negative result: steering is distributed or lives in the wrong parameter space for these bases.
  - Ambiguous outcome -> claim: if both keep and drop retain high behavior, report non-identifiability under this basis family and move to stricter causal interventions, not a positive subspace claim.

- [ ] **Goal: from-scratch parameterization steering.**
  - Why: decomposing trained `dW` is weaker than constructing a steering delta from base weights/activations alone.
  - Do: build simple `dW_prime = f(W_base, persona_contrast)` candidates, e.g. lm-head/readout rowspace projected persona contrast, write-not-read persona contrast, and shared structural bases with signed coefficients from activation contrast.
  - UAT: table compares `dW_prime` to trained `dW`, prompt, and repeng on identical sycophancy + DD rows.
  - Verify: candidates are generated without reading trained adapter deltas; code fails if `w.pt` is loaded before constructing `dW_prime`.
  - Positive outcome -> claim: if a from-scratch `dW_prime` steers, weight steering may be replaced by a constructive parameterization.
  - Negative outcome -> claim: if no from-scratch candidate steers while trained `dW` does, training is doing nontrivial search not captured by the current structural recipes.

## Deferred / optional

- [ ] **Goal: SVD steering baseline.**
  - Why: useful only if cheap and stable; lower priority than repeng.
  - UAT: same DD/sycophancy table as other baselines.
  - Verify: table includes `method=svd_steering`, `layer`, `rank`, `coeff`, `syc_delta`, `dd_delta`, and `pmass`.
  - Negative outcome -> claim: if SVD steering is weak or unstable, do not treat plain base-weight SVD as a competitive method baseline.

- [ ] **Goal: degradation benchmark.**
  - Why: steering might improve target metric while damaging general behavior.
  - UAT: perplexity or clean instruction proxy reported for best coefficients.
  - Verify: table has target metric and degradation metric for the exact same selected coefficients.
  - Negative outcome -> claim: if target gains require large degradation, report steering as brittle rather than useful.

- [ ] **Goal: larger model replication.**
  - Why: Qwen3-0.6B and Gemma 1B are iteration models; larger model needed for a stronger claim.
  - UAT: same benchmark table on a 4B-ish model after method stabilizes.
  - Verify: model column includes the 4B-ish model and reuses the same prompt/DD row IDs as the small-model benchmark.
  - Negative outcome -> claim: if the effect disappears or reverses on the larger model, write the small-model limitation instead of scaling the claim.

## Decision rules

- If prompt or activation steering beats `dW`, prioritize method improvement before deeper mechanistic analysis.
- If activation steering matches `dW`, treat weight steering as mechanistic interest first and applied method second.
- If DeLoRA wins across Qwen and Gemma, spend seeds on DeLoRA/PiSSA only.
- If Qwen and Gemma adapter rankings diverge, write the model-specific adapter-basin finding instead of forcing one global winner.
- Shared-core rule: if `keep_B_shared_K32` retains >=0.7x behavior across LoRA / DoRA / PiSSA / DeLoRA / OFT and `drop_B_shared_K32` removes most of it, write the planning-subspace paper.
- Basin-divergence rule: if per-adapter top subspaces are mutually low-overlap and each adapter's own SVD keeps behavior better than `B_shared`, write the basin-divergence paper.
- If top-k or write-not-read keeps behavior, we found a simple steering parameterization.
- If complement/tail/many layers keep behavior, evidence favors distributed or wrong-space mechanism.
- If MLP `up/gate` terms carry behavior, next paper story should be feature-space steering, not residual-stream planning subspace.
