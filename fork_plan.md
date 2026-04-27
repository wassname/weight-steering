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
  - [src/ws/eval/cross_adapter_ablation.py](src/ws/eval/cross_adapter_ablation.py)
  - [src/ws/eval/layer_module_ablation.py](src/ws/eval/layer_module_ablation.py)
  - [src/ws/eval/parameterization_ablation.py](src/ws/eval/parameterization_ablation.py)
  - [nbs/ablation_analysis.py](nbs/ablation_analysis.py)

## Current facts

- Daily-dilemmas default is **not full split**. Default `n_dilemmas=100` means first 100 dilemmas = 200 rows, balanced 100 honest-label and 100 dishonest-label actions.
- Full `honesty_eval` test split is 219 dilemmas = 438 rows.
- The daily-dilemmas eval uses all rows for selected dilemmas, then sign-flips by `honesty_label`; it is not only honest rows.
- Current headline tables are single-seed Qwen3-0.6B exploratory results.
- DeLoRA is best raw steering so far. PiSSA is the cleaner stable baseline if penalizing DeLoRA saturation at high alpha.
- v9/v10 do **not** prove “no subspace.” They show the trained behavior is not explained by the tested low-rank residual-stream bases or adapter-family parameterization at trained scale.
- The active analysis should ablate the already-trained `dW`. Synthetic `dW'` construction is a different baseline, not causal ablation.
- The highest-value analysis tests are: cross-adapter causal `dW` basis ablation, layer/module ablation of trained `dW`, and adapter-parameterization ablation of trained `dW`.
- **Lens search is on hold pending multiseed (2026-04-27).** Every weight-space lens we tested has a built-in failure mode: SVD-on-`dW` is tautological for low-rank adapters; layer-index tells depth not mechanism; module-family collapses heads/positions and gives different answers per adapter; native parameterization decompositions aren't comparable across adapter families. *But* the lens-3 cross-adapter inconsistency (delora residual_write retained=+1.27 vs lora=+0.14) is N=1 seed × N=1 model. It might just be seed noise within each adapter. Right ordering: T4 multiseed first, then re-run T7/T8 per-seed with within-adapter stdev, then judge whether the inconsistency is real or noise.

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

- [x] **Goal: activation-steering baseline on the same DD rows.**
  - Why: RepE/repeng is the most threatening baseline; if it matches or beats `dW`, the method story weakens before adapter seeds matter.
  - Do: train representation direction on the same sycophancy contrast; grid layer x coefficient; evaluate sycophancy and full DD.
  - UAT: best activation-steering row is selected by held-out sycophancy or validation DD, then reported beside best `dW` on identical DD test rows.
  - Verify: table includes `method=repeng`, `layer`, `coeff`, `syc_delta`, `dd_delta`, `pmass`, and the same `idx` set as the `dW` rows.
  - Negative outcome -> claim: if repeng matches/beats `dW`, write "activation steering is the simpler baseline; weight steering needs a stronger reason to exist."

- [x] **Goal: full daily-dilemmas benchmark for current Qwen adapters.**
  - Why: current DD table uses first 100 dilemmas, not the full 219-dilemma split.
  - Do: re-run LoRA / PiSSA / DeLoRA / DoRA / OFT / IA3 with `--n-dilemmas 219`.
  - UAT: table has 438 base rows per coeff before persona baselines, and reports `pmass`, `frac_low_pmass`, `delta(+1 - 0)`.
  - Verify: `out/sycophancy/cross_adapter_full_dd/dilemmas_summary.csv` exists and includes `n_base_rows_per_coeff=438`.

- [x] **Goal: prompt baselines on the same DD rows.**
  - Why: weight steering is only interesting if it beats “just prompt it.”
  - Do: evaluate base, simple honest persona, and engineered AxBench-style prompt.
  - UAT: one table compares `base`, `simple_honest_prompt`, `engineered_prompt`, and best `dW` on identical rows.
  - Verify: `prompt_baseline_delta` and `weight_steer_delta` are computed from the same `idx` set.
  - Negative outcome -> claim: if prompting matches/beats `dW`, write "prompting is the simpler intervention for this behavior/eval pair."

- [ ] **Goal: multi-seed adapter benchmark on Qwen.**
  - Why: current adapter ranking is N=1 seed.
  - Do: run seeds 0, 1, 2 for LoRA / PiSSA / DeLoRA first; add DoRA/OFT only if cheap.
  - UAT: table reports mean +/- std for sycophancy and DD deltas, plus seed-level signs, so a reader can tell stable ranking from noisy N=1 luck.
  - Verify: each adapter has exactly three `w.safetensors` files and three eval summaries; ranking table includes `n_seeds=3`, `mean_dd_delta`, `std_dd_delta`, and `sign_agreement`.
  - Negative outcome -> claim: if adapter ranking changes across seeds or error bars overlap heavily, write "single-seed adapter winner is unstable; do not claim a family ranking yet."

- [ ] **Goal: Gemma 1B replication.**
  - Why: check whether DeLoRA/PiSSA ranking is Qwen-specific.
  - Do: train LoRA / PiSSA / DeLoRA on Gemma 1B, seed 0, full DD split.
  - UAT: compare Gemma ranking to Qwen ranking with the same metrics.
  - Verify: table has model column with `Qwen3-0.6B` and `Gemma-1B`; if DeLoRA remains best, expand seeds; if rankings diverge, write that up as a model-specific adapter-basin finding.

## TODO: analysis question

**Status (2026-04-27): on hold pending multiseed.** T6/T7/T8 are run on
N=1 seed × Qwen3-0.6B. Necessity is established. The cross-adapter
inconsistency that drove the "no parameterization-invariant mechanism"
reading might be seed noise. Resume after T4 (multiseed) lands and we can
report within-adapter stdev alongside cross-adapter gaps.

Active sequence at the time of pause was:

1. Cross-adapter causal `dW` basis ablation.
2. Layer/module causal ablation of trained `dW`.
3. Adapter-parameterization causal ablation of trained `dW`.

Synthetic `dW'` construction is deferred below and is not a causal ablation.

- [x] **Goal: cross-adapter causal `dW` basis ablation.**
  - Why: this is the headline analysis experiment. It tests whether different adapter families discovered the same causal planning subspace or different basins.
  - Do: build candidate bases `B` from trained adapter deltas, compute `dW_keep_B` and `dW_drop_B`, and evaluate both on sycophancy + full DD for each adapter.
  - Candidate `B` rows:
    - `shared_SVD_K8/K32/K64`: stack residual-output `dW` from LoRA / DoRA / PiSSA / DeLoRA / OFT per layer/tensor, take top-K SVs.
    - `top8/top32_per_adapter` and `tail_per_adapter`: per-adapter SVD split of each tensor.
    - `random_null`: rank-matched random bases.
  - UAT: one central table has `ablation_family`, `candidate_B`, `adapter`, `rank`, `keep_or_drop`, `syc_delta`, `dd_delta`, `pmass`, and row-identity checks.
  - Verify: the table contains keep/drop rows for `shared_svd`, `per_adapter_svd`, and `random_null`; `keep_B_shared_K32` and `drop_B_shared_K32` are both evaluated for at least LoRA / DoRA / PiSSA / DeLoRA / OFT; random null retention is near rank/d; each row uses the same eval rows and coefficient grid.
  - Positive outcome -> claim: if `keep_B_shared` retains >=0.7x behavior across adapters and `drop_B_shared` removes it, write the adapter-invariant planning-subspace paper.
  - Negative outcome -> claim: if `keep_B_shared` retains <0.3x even at K=64 while complements/tails retain behavior, write the shared-subspace negative result: steering is distributed or lives in the wrong parameter space for these bases.
  - Ambiguous outcome -> claim: if both keep and drop retain high behavior, report non-identifiability under this basis family and move to stricter causal interventions, not a positive subspace claim.

Note for the following two a search has been made of hypothesis: docs/hypothesis_ablation_catalog.md

- [x] **Goal: layer/module causal ablation of trained `dW`.**
  - Why: after a trained update works, we need to know which layers and modules are necessary or sufficient.
  - Do: keep/drop parts of the already-trained adapter delta by layer and module family, without synthesizing new tensors from base features.
  - Rows: `full_dW`, `residual_write_only`, `attn_o_proj_only`, `mlp_down_proj_only`, `layers_8_21_only`, single-layer keep, leave-one-layer-out, coarse early/mid/late LoRA-layer blocks, rank/module-matched random controls, and `zero`.
  - UAT: one table has `adapter`, `variant`, `layer_or_block`, `module_family`, `keep_or_drop`, `syc_delta`, `dd_delta`, `pmass`, and row-identity checks.
  - Verify: same sycophancy and full DD rows as `full_dW`; table includes all required variants and reports zero row-key symmetric difference for every variant x coeff group.
  - Positive outcome -> claim: if a small layer/module slice retains most behavior and dropping it removes behavior, report the causal locus.
  - Negative outcome -> claim: if many disjoint slices retain behavior, report distributed or non-identifiable layer/module localization.

- [x] **Goal: adapter-parameterization causal ablation of trained `dW`.**
  - Why: adapter families may store the behavior in different parameterization degrees of freedom even when their effective `dW` looks similar.
  - Do: split the trained adapter/effective delta according to the adapter family's own coordinates, then keep/drop each component on identical eval rows. For an S-space split, compute the trained effective matrix's SVD-like coordinate system, project `dW -> S`, crop a component such as the top 25% of `S` by coordinate index, project back to weight space, and evaluate both `top_25pct_S` and `residual_not_top_25pct_S` against `full_dW` and `zero`.
  - Rows: LoRA/PiSSA/DeLoRA rank components and S-space quartiles (`top_25pct_S`, `mid_50pct_S`, `bottom_25pct_S`, `residual_not_top_25pct_S`, `residual_not_bottom_25pct_S`); cumulative S-energy groups (`top_50pct_energy_S`, `top_90pct_energy_S`, residuals); DoRA direction vs magnitude component; OFT rotation-derived component vs residualized effective update; IA3 attention-gate vs MLP-gate groups.
  - UAT: one table has `adapter`, `parameterization_family`, `coordinate_system`, `component`, `keep_or_drop`, `rank_or_group`, `energy_frac`, `syc_delta`, `dd_delta`, `pmass`, and row-identity checks.
  - Verify: all rows start from the trained adapter delta or trained adapter parameters; no row is constructed from base-only activations; every component shares the same sycophancy and DD row keys as `full_dW`; for each S-space crop, `component_dW + residual_dW` reconstructs `full_dW` within numerical tolerance.
  - Positive outcome -> claim: if one parameterization component retains most behavior and dropping it removes behavior, report which degree of freedom carries the learned behavior.
  - Negative outcome -> claim: if behavior is not localized by parameterization component, report the trained effect as distributed across that adapter parameterization.

## Coverage gaps in current ablation set

The three causal ablations above (cross-adapter `dW` basis, layer/module,
adapter parameterization) leave some hypotheses untested. These are open
follow-ups, not blockers for the current writeup.

- [ ] **Read-side modules in the layer/module ablation.** Current variants
  cover residual writes (`o_proj`, `down_proj`), attention-only, and
  mlp-only, but not q/k/v-only or up/gate-only. Any read-side mechanism
  story is currently untestable.
- [ ] **Base-W SVD lens for the S-space ablation.** `parameterization_ablation.py`
  uses each tensor's own SVD (`dW = U S Vh`). The catalog also wants a
  separate lens using the base weight's SVD (`U0, S0, V0h = svd(W_base);
  dS = U0.T @ dW @ V0h`), which answers "does `dW` ride pretrained
  singular directions" rather than "is `dW` low-rank in its own basis".
- [ ] **Adapter-architecture decompositions.** S-space variants do not
  include DoRA magnitude vs direction, DeLoRA lambda vs direction, OFT
  rotation, or IA3 attention-gate vs MLP-gate splits.
- [ ] **Norm-matched random keep control for T8 sufficiency claims.**
  Layer/module ablation has `random_norm_matched_full`; the S-space crops
  do not. Necessity (drop) tests don't need this; sufficiency (keep) tests
  do, because cropping shrinks Frobenius norm and the model is nonlinear
  in alpha.

## Deferred / optional

- [ ] **Optional future: constructive synthetic `dW'` baseline.**
  - Why: useful as a method baseline, but it is not a causal ablation of trained weight steering.
  - Do: only if separately approved, build simple `dW_prime = f(W_base, persona_contrast)` candidates, e.g. lm-head/readout rowspace projected persona contrast, write-not-read persona contrast, and shared structural bases with signed coefficients from activation contrast.
  - UAT: table compares synthetic `dW_prime` to trained `dW`, prompt, and repeng on identical sycophancy + DD rows.
  - Verify: candidates are generated before reading trained adapter deltas; code fails if `w.safetensors` is loaded before constructing `dW_prime`.
  - Positive outcome -> claim: if a synthetic `dW_prime` steers, weight steering may be replaceable by a constructive method baseline.
  - Negative outcome -> claim: if no synthetic candidate steers while trained `dW` does, training is doing nontrivial search not captured by the current structural recipes.

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
- Held-out cross-adapter shared-SVD rule (contingent on T4 multiseed):
  - If T4 shows within-adapter stdev > cross-adapter retained gap on lens-3, the N=1 cross-adapter inconsistency was seed noise -> held-out shared-SVD becomes worth running.
  - If T4 confirms cross-adapter gap is real (current expectation), skip held-out shared-SVD -- it will fail by construction and only restates what is already known.

## Resolved: train/eval axis switch from sycophancy to honesty (2026-04-27)

**Decision: option 2** — switched the entire pipeline to honesty axis. Rationale:

- The sycophancy NEG persona was already "you are an honest assistant", so the prior contrast was effectively honesty-vs-anti-honesty with extra sycophancy framing on the POS side. Cleaner to make that explicit.
- Eval (`daily_dilemmas-self-honesty`) signs by `honesty_label`. One-axis training matches the eval axis by construction. The in-sample claim is now defensible.
- Sycophancy becomes a free OOD transfer eval (run later if useful) rather than a confounded primary axis.
- Avoids the GPT-4o sycophancy-eval generation burden (option 1) and its eval-validation overhead.

What changed in code (commit on `dev`):

- `src/ws/data.py`: added `HONESTY_PROMPT = "Pretend you're {persona} person. "` with 5 paraphrase paraphrases each side (`HONESTY_POS_PERSONAS`, `HONESTY_NEG_PERSONAS`) — vgel/repeng short-persona convention. Added `_load_suffixes()` reader for `data/branching_suffixes.json` (550 SSteer entries). Honesty branch reuses the suffix `user_msg` as the question pool; persona-conditioned generation in `generate_pairs()` produces (r_pos, r_neg) per the paper recipe (Fierro & Roger 2025 §F.1).
- `src/ws/eval/activation_baseline.py`: RepE direction extraction now branches on `cfg.behavior`; honesty mode captures last-token hidden states under `HONESTY_POS[0]` / `HONESTY_NEG[0]` over `_load_suffixes()` prompts with `assistant_prefixes=suffix`.
- `src/ws/eval/prompt_baseline.py`: replaced single `engineered_prompt` with paired `engineered_prompt_honest` + `engineered_prompt_dishonest` (AxBench Appendix J.2 style).
- `evals/smoke.py`: added `behavior` field; `just smoke --behavior honesty` passes end-to-end on `katuni4ka/tiny-random-qwen3`.
- `data/branching_suffixes.json`: copied from SSteer.

Sycophancy outputs in `out/sycophancy/` are kept on disk as historical evidence for the old axis-mismatched table. The README headline numbers will be replaced with honesty once 230-236 land. T4/T5 stay open and will be re-scoped against honesty.
