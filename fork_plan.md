## Context

So this is a fork of the excellent weight steering

> We isolate behavior directions in weight-space by subtracting the weight deltas from two small fine-tunes - one that induces the desired behavior on a narrow distribution and another that induces its opposite.
> To obtain a vector in weight space corresponding to the desired trait, we start from a model θ0 then fine-tune the model on either the data generated with the positive system prompt (stripped of the system prompt at train-time) to obtain θ+, or on the data generated with the negative system prompt to obtain θ−, the weight-space vector corresponding to the behavior is then computed as w=θ+−θ−. We use LoRA fine-tuning as we found it worked better for monitoring than full-parameter fine-tuning.


Now I'm interested in
- replicating
- seeing if the model difference aligns with SVD vs W. With any of the subspaces I defined in ./docs/AntiPaSTO_concepts/
- and most importantly seeing if other types of adapters work better!
- and becoming clear on
    - does it generalise
    - does performance degrate
    - this likely means using it on one of the evals I'm familiar with namely daily dillemas from AntiPaSTO or eval awareness (but this requires rending a GPU so this is later)


## Resources

- **my lit review of PeFT adapter methods** ./docs/blog_adapter_as_hypothesis/README.md
- my steering concepts ./docs/AntiPaSTO_concepts/README.md
- orig paper
    - docs/weight_steering_paper.md
    - docs/weight_steer_blog.md


## TODO

- [x] plan to clean up the repo. uv, jaxtyping, einops. hooks not classes. remove vlm
- [x] make it work on small models (Qwen3-0.6B), cheap+fast iteration
- [x] hook in PEFT (LoRA / DoRA / PiSSA / DeLoRA via peft>=0.13)
- [x] phase 1 replicate: w = θ+ - θ- on Qwen3-0.6B sycophancy, monotone logratio (task 40)
- [x] phase 2 weight-only subspace alignment (SVD-of-W, weak-readout) — *negative result, see "Phase 2 reframe" below*
- [x] phase A demos: adapter coherence + guided-CoT under w (task 44 — pmass=1.0, margin α-monotone, no teacher-forcing gap, OOD generalizes)
- [ ] phase B: train.py val split done; 3-epoch re-run still pending
- [ ] phase 2.5: activation-aware subspace tests — TaskDiff / Suppressed / Stenographic
- [ ] **wishlist W**: layer slice 30-80% LoRA targets (steering literature locus); `train.py:LINEAR_TARGETS` patch
- [ ] **wishlist N**: `notebooks/analyze_diff.py` (.py # %% cells) — W-side (SVD spectrum, polar decomp, suppressed-PCA, magnitude-vs-direction) + A-side (Δa via baukit at α=±1, per-layer residual/attn/MLP locus, cosine to dW directions)
- [ ] phase 3 adapter sweep (DoRA / PiSSA / DeLoRA)
- [ ] phase 4 daily-dilemmas eval (mirror AntiPaSTO2/antipasto2/eval.py)
- [ ] **paper-deltas** (task 18, 16): match data recipe (5+/5- × 10 samples + judge filter) and LoRA hyperparams (rank 32, α 16, lr 1e-5, warmup 5)

## Paper-deltas — what we match, what we deliberately skip

Audit of upstream Axolotl YAMLs vs current code. Tracked as tasks 16 + 18.

| upstream | ours | decision |
|---|---|---|
| 20 questions × 5 personas × 10 samples + GPT-4.1-mini filter (500-900 retained per sign) | 32 fixed claims × 1 persona, sample-replicated to 1000 | **fix** (task 18) |
| LoRA rank 32 / α 16 / lr 1e-5 / warmup 5 / wd 0.01 / no dropout | rank 16 / α 2*r=32 / lr 5e-5 / no warmup / no wd | **fix** (task 16) |
| `load_in_8bit: true`, `adamw_bnb_8bit` | `bf16` direct, plain AdamW | **skip** — DoRA/PiSSA/DeLoRA quantization support is uncertain; bf16 fits at 0.6B |
| `modules_to_save: [embed_tokens, lm_head]` | not saved | **skip** — user does not want to train/save these |
| `lora_target_linear: true` (all linear) | hand-picked q/k/v/o/gate/up/down_proj | **skip** — deliberate, this is all linear in the qwen3 transformer block anyway; matches `lora_target_linear` for the body |
| sequence length 4096 | 512 | **skip** — sycophancy responses are <128 tokens; 512 is plenty, 4096 would OOM at our batch size |
| `epochs` plumbed through | (was) silently ignored | **fixed** (replicate.py:71, 2026-04) |
| reuses on-disk data regardless of `n_pairs` | now hard-fails on mismatch | **fixed** (replicate.py:_maybe_data, 2026-04) |

---

# Fork plan: weight-steering → small-model + adapter sweep

## Context

This is a fork of Anthropic's weight-steering work (θ+ - θ- via LoRA fine-tunes on +/- system-prompted data). The current repo is heavy: Axolotl orchestration, vLLM serving, and Anthropic/OpenAI batch APIs. None of that is needed for what wassname actually wants:

1. **Replicate** the core method on a small model so iteration is cheap.
2. **Test alignment** between the diff vector `w = θ+ - θ-` and the SVD-derived subspaces from `docs/AntiPaSTO_concepts/` (suppressed, write-not-read, weak-readout, stenographic).
3. **Test other PEFT adapter families** (DoRA, PiSSA-init LoRA, DeLoRA) to see if the steering signal extracts more cleanly under different parameterizations - this is the "adapter as hypothesis" framing from `docs/blog_adapter_as_hypothesis/`.
4. **Generalization** via daily-dilemmas eval (later, GPU-gated).

The original paper itself notes "we did not try to optimize weight steering very hard" - room for both methodological cleanup and substantive method comparison.

User decisions captured: Qwen3-0.6B base, aggressive cleanup (rip Axolotl + VLM, switch to HF+PEFT), both sycophancy (paper replication) and daily-dilemmas (own eval), adapter sweep over LoRA / DoRA / PiSSA-init / DeLoRA.

## Phase 0 — Repo cleanup (breaking, no backcompat)

**Delete:**
- `vllm_inference.py` (565 lines, vLLM serving)
- `api_inference.py` (964 lines, Anthropic/OpenAI batch)
- `axolotl_plugin_models_with_mlp_bias.py`, `axolotl_configs/`
- `inference_and_eval.py` Axolotl-subprocess orchestration (keep nothing - rewrite small)
- `models_with_mlp_bias.py` - replace with hooks; the MLP-bias variant isn't needed for the core θ+ - θ- replication

**Add:**
- `pyproject.toml` with uv (`torch`, `transformers`, `peft>=0.13` for DeLoRA, `datasets`, `einops`, `jaxtyping`, `beartype`, `loguru`, `polars`, `tabulate`, `baukit` from git, `wandb`)
- `justfile` with: `smoke` (5-min run), `train-pos`, `train-neg`, `diff`, `eval-syco`, `eval-dilemmas`, `subspace-align`
- `.python-version` (3.11)

**Keep + simplify:**
- `task_vectors.py` — strip down to a functional `compute_diff(state_dict_pos, state_dict_neg) -> dict` and `apply_diff(model, diff, alpha)`. Drop the class hierarchy and arithmetic ops; we only need subtract + scaled add.
- `activation_steering.py` — already hook-based; replace manual hooks with `baukit.TraceDict` for cleanliness (per user CLAUDE.md preference).

**New layout:**
```
weight-steering/
├── src/ws/
│   ├── data.py          # +/- system-prompt pair data generation (sycophancy first)
│   ├── train.py         # PEFT-based finetune; one function per adapter type
│   ├── diff.py          # compute_diff, apply_diff (functional, ~50 lines)
│   ├── steer.py         # inference-time scaled application via baukit hooks
│   ├── subspace.py      # SVD projections, AntiPaSTO subspaces, alignment metrics
│   └── eval/
│       ├── sycophancy.py
│       └── dilemmas.py  # mirrors AntiPaSTO2/eval.py pattern
├── scripts/
│   ├── replicate.py     # phase 1 entrypoint
│   ├── adapter_sweep.py # phase 3 entrypoint
│   └── subspace_align.py
├── notebooks/           # exploratory only
└── justfile, pyproject.toml, .python-version
```

## Phase 1 — Replicate on Qwen3-0.6B with sycophancy

**Data:** Generate +/- pairs using sycophantic vs honest system prompts on a sycophancy QA distribution (paper Appendix E recipe). Strip system prompt at train time. Target ~500-1000 pairs to keep iteration fast.

**Train:** PEFT LoRA, rank 16, all linear layers, lr 5e-5, 1 epoch, bf16. Save θ+ and θ- as PEFT adapter state dicts. With Qwen3-0.6B + LoRA this should fit comfortably on a single 24GB card and train in ~10-20 min per side.

**Diff:** `w = θ+ - θ-` in adapter-merged weight space (merge LoRA into a delta dict, then subtract). Functional, no class wrapper.

**Apply at inference:** Add `alpha * w` to base weights via baukit hook on each affected `nn.Linear` (no in-place modification of base model). Sweep `alpha ∈ [-2, -1, 0, 1, 2]`.

**Smoke test:** Qualitative gen on 10 held-out sycophancy prompts, plus the per-coeff Yes/No logratio metric from `AntiPaSTO2/eval.py`.

## Phase A — Sanity demos on existing artifacts (cheap, no retraining)

Why: task 40's pipeline never generates a single sentence of model output.
The headline numbers (`mean_logratio +9.4 at α=+2`, `pmass=1.0`) are forward-pass-only,
single-token reads. We don't yet know:

1. Did the LoRAs converge or undertrain? Single epoch, slope -0.003/step at the end, no val loss.
2. Were the adapters coherent at the end? No generation anywhere.
3. Does the steering effect survive a 32-token rollout? Single-token logratio inflates vs on-policy reality (ROAST teacher-forcing gap).
4. Does w generalize off the training topic distribution? Eval is in-distribution (`held_out = SYCOPHANCY_TOPICS[-16:]`).

Two demos, both on the existing `out/sycophancy/lora/{pos,neg,w.pt}`:

- **A1** (`run_demo.py:phase_a1`): load base + pos LoRA, generate 80 tokens on 2 in-dist + 1 OOD claim. Same for neg. Pass = pos *agrees*, neg *pushes back*, both fluent. **Built**, not yet run.
- **A2** (`run_demo.py:phase_a2`, `eval/guided_cot.py`): for each (claim, alpha) pair, rollout 32 tokens of CoT under `weight_steer(model, w, alpha)`, append `"\n\nFinal answer: **"`, score `margin = logp_yes - logp_no` and `pmass = P(yes) + P(no)` at the next position. Per AntiPaSTO `docs/AntiPaSTO_concepts/README.md:467-477`: pmass≈1.0 in linear range, drops outside. **Built**, not yet run.

Run with `just demo`.

## Phase B — Convergence/overfit (only if A flags issue)

Patched `train.py` adds 10% val split + `eval_strategy="steps", eval_steps=10`.
Re-queue with 3 epochs:

```
pueue add -l "why: did task 40 LoRA converge or undertrain; resolve: val_loss curve flattens (converge), keeps dropping (undertrain), or U-curves (overfit)" -- uv run python -m ws.replicate --model Qwen/Qwen3-0.6B --behavior sycophancy --adapter lora --n-pairs 1000 --epochs 3
```

Reuses task 40's data on disk. ~6 min total.

## Phase 2 reframe — why activation-blind SVD-of-W was the wrong test

Task 40 measured energy of `w_layer` in the top-k×k corner of base SVD(W). Across 7 module kinds, all `ratio_top ≈ 1.0 ± 0.10` (per-layer std). I initially called this "SVD-alignment falsified."

That's the wrong reading. From `docs/AntiPaSTO_concepts/docs/steering_methods.qmd:340-343` (Common Misconceptions #3, "SVD(W) aligns with PCA(diffs)"):

> Wrong: Weight's principal directions should align with task-relevant activation differences.
> Right: SVD(W) captures variance across *all* computations; PCA(diffs) captures variance for *this task*. We measured ~0.08 cosine similarity — essentially orthogonal.

So task 40 didn't falsify a hypothesis; it reproduced a known prior result (SVD(W) is not the task basis). The Fisher table at `steering_methods.qmd:207-214` says the same thing differently:

| Subspace | Peak Fisher |
|---|---|
| weight_svd / write_minus_lm_head | 0.007–0.009 (Level 0) |
| task_diff / suppressed | 0.013–0.022 (Level 1) |
| **stenographic** (task ∩ suppressed) | **0.142** (Level 2) |
| task ∩ stenographic | 0.266 (Level 3) |

The *right* test is what wassname intuited: project task hidden states (or their differences) onto a basis, then test if `w` aligns with that. Three concrete activation-aware tests to add (replacing the Haar-null SVD-of-W test):

1. **TaskDiff alignment**: collect `h_pos[L]` and `h_neg[L]` on a probe set (using base model under +/- system prompts on training topics). PCA on `h_pos - h_neg`, top-k. Test if `w_layer`'s column space (the side that writes to residual) aligns with this. Null = random rank-r perturbation.
2. **Suppressed alignment**: per `steering_methods.qmd:67-110`, compute `min(Σrelu(Δmag+), Σrelu(Δmag-))` across layers, PCA. Suppressed has 3.5× enrichment for task signal vs random (`steering_methods.qmd:407-414`). Test `w` against this.
3. **Stenographic alignment**: TaskDiff ∩ Suppressed (canonical-angle bisector basis). Highest Fisher (0.142) per AntiPaSTO. If `w` doesn't align with *anything* including stenographic, the diff carries no task-relevant subspace structure.

The existing weak-readout test (`subspace.py:weak_readout_alignment`) is in spirit Logits_Null (`steering_methods.qmd:81`) — keep it.

Cleanup needed in `subspace.py` regardless:
- The current `e_top` only sums the (top-k × top-k) corner of `proj`, ignoring off-diagonal blocks `proj[:k, k:]` and `proj[k:, :k]`. For a "row-side aligned but col-side random" delta (which a LoRA `B@A` may produce when B is in W's col-space but A is not in W's row-space), this misses signal. Either measure all four blocks or restate the hypothesis.
- The reported ±0.10 was per-layer std over n=28 layers, not SE of the per-kind mean. Re-doing as SE: down_proj +2.2σ, v_proj −1.7σ from null. Bonferroni across 7 kinds kills these, but it's not "1.0 ± 0.1 across all kinds" — there is per-kind variation.

## Phase 2 — Subspace alignment analysis (original plan; superseded by Phase 2 reframe + Phase 2.5)

For each layer's weight matrix W and its diff `w_layer`:

1. SVD of pretrained W → `U_out, S, U_in.T`.
2. Project `w_layer` onto top-k singular components; compute energy fraction vs uniform/random baseline.
3. Repeat for the four AntiPaSTO subspaces:
   - **Suppressed** (PCA of layer-to-layer magnitude drops on a probe set)
   - **Write-not-read** (orth complement of next layer's read span)
   - **Weak-readout** (bottom-1% Vh of unembedding)
   - **Stenographic** (intersection of task-diff and suppressed)
4. Output: a polars table per subspace with `{layer, energy_in_subspace, energy_random_baseline, ratio}`. Print with tabulate.

Critical: project the *adapter-space* delta when possible (rank-r is small) and compare against the same projections of random rank-r perturbations as the null. This makes the alignment claim falsifiable.

## Phase 3 — Adapter sweep (the actual science)

For each adapter type, train +/- and produce a weight-space diff:

| Adapter | PEFT support | Hypothesis being tested |
|---|---|---|
| LoRA r=16 | built-in | baseline: low-rank suffices |
| DoRA r=16 | built-in (`use_dora=True`) | magnitude/direction split keeps diff cleaner |
| LoRA + PiSSA init | `init_lora_weights="pissa"` (built-in init mode) | principal components carry the steering signal |
| DeLoRA r=16 | built-in (peft >= 0.13) | strength/direction decoupling improves robustness |

Per adapter, log: train loss curves, time, peak mem, then phase-1 sweep + phase-2 alignment table. The cross-adapter comparison is the key result: **does the SVD/subspace alignment of `w` change when we change the parameterization?** That's evidence about whether the adapter itself is acting as an inductive bias on the steering direction (the "adapter as hypothesis" framing).

Scope guard: drop SSVD; user already excluded it. If DeLoRA blows up in PEFT, fall back to LoRA + PiSSA + DoRA.

## Phase 4 — Daily-dilemmas eval (CPU-feasible at 0.6B)

Build `src/ws/eval/dilemmas.py` mirroring `AntiPaSTO2/antipasto2/eval.py`
(fetched via `gh api repos/wassname/AntiPaSTO2/contents/antipasto2/eval.py`).
Reuse our existing primitives — don't re-implement choice scoring.

Source eval pipeline (key fields to mirror):
- **Dataset**: `wassname/daily_dilemmas-self-honesty`, config `honesty_eval`,
  `split="test"`. Take top-N by `dilemma_idx` (default 100). Each row has
  `dilemma_idx`, `idx`, `action_type`, `honesty_label` (+1/-1).
- **Prompt**: `INSTRUCTION_PROMPT.format(**row)` then assistant `"My choice: **"`,
  built via `apply_chat_template(continue_final_message=True, add_generation_prompt=False)`.
  *Vendor `INSTRUCTION_PROMPT` from AntiPaSTO2/antipasto2/data.py.*
- **Score**: yes/no logratio at last position (same as our `sycophancy.py`).
  Reuse `ws/eval/sycophancy.py:get_choice_ids` — already identical to v2.
- **Honesty alignment** (the key v2 detail): `logratio_honesty = logratio * honesty_label`.
  Positive = more honest. Aggregate this, not raw logratio — sign cancels otherwise.
- **Coeff sweep**: `[-1.0, 0.0, 1.0]` (default; can override).
- **Steering**: AntiPaSTO2 uses `ScaleAdapter(model, coeff, adapter_name)` (PEFT
  scaling LoRA at inference). We use `weight_steer(model, w, alpha)` instead —
  same shape (context manager scaling a delta), but on *the diff* w = θ⁺ − θ⁻
  not on a single adapter. Drop-in.
- **pmass flag**: `low_pmass = pmass < threshold * maxp` (threshold=0.01).
  Don't filter — flag for analysis. Compare to our guided-CoT `pmass≈1.0` baseline.

Output: one polars table per adapter: `(adapter_type, coeff, mean_logratio_honesty,
mean_pmass, frac_low_pmass)`. Save per-row CSV for later regression on
`action_type`.

Wire as `ws/eval/dilemmas.py` + `evaluate()` entrypoint in `replicate.py`
(after sycophancy eval). `just eval-dilemmas adapter=lora` recipe.

## Phase 5 — Generalization + degradation (later, rented GPU)

Defer until phases 1-4 produce a clear winner. Then on a 4B model:
- Eval on held-out dilemma distribution + eval-awareness eval.
- Track perplexity on a clean instruction-following set as a degradation proxy.

## Critical files to modify / reference

- **Modify heavily:** `task_vectors.py`, `activation_steering.py`
- **Delete:** `vllm_inference.py`, `api_inference.py`, `axolotl_plugin_models_with_mlp_bias.py`, `models_with_mlp_bias.py`, `inference_and_eval.py`, `axolotl_configs/`
- **Reference (read-only):** `docs/weight_steering_paper.md` (Appendix B/E hyperparams), `docs/AntiPaSTO_concepts/README.md` (subspace definitions), `docs/blog_adapter_as_hypothesis/README.md` (adapter scoring)
- **Mirror:** AntiPaSTO2 `antipasto2/eval.py` (eval pattern, choice-id extraction, ScaleAdapter context manager)

## Reuse, don't reinvent

- `peft.LoraConfig(use_dora=True, init_lora_weights="pissa")` for DoRA and PiSSA-init - no custom code.
- `peft.DeloraConfig` for DeLoRA (peft >= 0.13).
- `baukit.TraceDict` for steering hooks (per user CLAUDE.md).
- AntiPaSTO2's `_is_choice`, `get_choice_ids`, `get_choice_logprobs`, `evaluate_at_coeff` - copy or vendor.
- `loguru` + `tabulate(df, tablefmt='pipe', headers='keys', floatfmt='+.2f')` for log output.

## Verification

End-to-end checks the user can read at a glance:

1. **Phase 0 done when:** `just smoke` runs in <5 min on Qwen3-0.6B, generates 5 +/- pairs, trains a LoRA on each, computes `w`, applies at coeff ±1, prints generations side by side. Single command, no Axolotl, no vLLM.
2. **Phase 1 done when:** sycophancy logratio on held-out set goes monotonically from coeff -2 → +2, table printed via tabulate.
3. **Phase 2 done when:** for each AntiPaSTO subspace, an `energy_ratio = energy_in_subspace / energy_random` table is produced. Ratio > 1 with bootstrap CI not crossing 1 = real alignment.
4. **Phase 3 done when:** the four-row table `(adapter × subspace_alignment × steering_logratio_AUC)` exists and is interpretable.
5. **Phase 4 done when:** daily-dilemmas table is reproducible from a single `just eval-dilemmas adapter=lora` command.

User-observable result throughout: a markdown table per phase, not a "I did it." Each table answers one question.

## Open questions to resolve during implementation (not blockers)

- Sycophancy data: regenerate using Qwen3-0.6B as the +/- responder, or use the paper's released data if available? (Default: regenerate with Qwen3-0.6B since 0.6B's distribution differs from 7B's.)
- Layer selection for the diff: paper does per-layer sweeps (Appendix E). For phase 1 just take all layers; for phase 3, sweep.
- Whether to merge adapter into base before diffing or diff in adapter space directly. Adapter-space is cheaper but only valid when both +/- adapters share the same A or B (PiSSA init shares both initially; LoRA does not). Default: merge into delta-W space, then diff. This makes all adapters comparable.
