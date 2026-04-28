# Activation and weight hypotheses for steering and ablation

Date: 2026-04-27

Purpose: collect the hypotheses scattered across `nbs/`, local qmd notes, and the current fork plan into one map. The key distinction is:

- untrained-base recipe: hypotheses built before looking at the trained adapter delta. These can become from-scratch steering methods or synthetic `dW'` baselines.
- trained-delta oracle: labels or oracles derived from the trained adapter effect. These are not fair from-scratch methods, but they are good causal ablation targets.
- causal fit: whether the hypothesis belongs in the planned cross-adapter `dW` basis ablation, layer/module ablation, adapter-parameterization ablation, activation-steering baseline, synthetic `dW'`, or a separate causal test.

Vocabulary discipline: `synthetic dW'` is causal only as a new constructive intervention. It is not a causal ablation of the already-trained adapter. Any activation-steering baseline must be built without loading trained `w.pt` or using `act_oracle`/`TaskDiff_lora_fit`; otherwise it is a trained-delta oracle, not a fair baseline.

Core fork-plan mapping:

| fork-plan experiment | Fits what | Does not fit what |
|---|---|---|
| [cross-adapter causal `dW` basis ablation](fork_plan.md) | learned `dW` SVD bases, shared adapter bases, per-adapter top/tail bases | pure activation bases unless first converted into a weight projection of the trained `dW` |
| [layer/module causal ablation of trained `dW`](fork_plan.md) | layer slices, residual writers, attention output, MLP down, read/write module families | candidate bases that mix all layers without layer labels |
| [adapter-parameterization causal ablation of trained `dW`](fork_plan.md) | LoRA rank components, PiSSA/DeLoRA S-space crops, DoRA magnitude vs direction, OFT rotations, IA3 gates | post-hoc activation PCA unless used only as an evaluation target |
| [activation-steering baseline](fork_plan.md) | TaskDiff/RepE directions built without trained `dW`, selected on held-out validation rows | trained `dW` components, `act_oracle`, `TaskDiff_lora_fit` |
| [synthetic `dW'` baseline](fork_plan.md) | pretrained read/write bases with signed coefficients from contrast activations | causal claims about the already trained adapter |
| new causal test | nonlinear clusters, token-conditional attention routing, concept-space probes, DAS/SAE features | simple keep/drop of a fixed linear `dW` basis unless linearized first |

## Source provenance

Notebook sources:

- [nbs/analyze_diff_v2.py](../nbs/analyze_diff_v2.py): first clean TaskDiff, suppressed, stenographic, `lm_head_read`, `logits_null` concentration test.
- [nbs/analyze_diff_v3.py](../nbs/analyze_diff_v3.py): A-side hypothesis vs B-side label framing.
- [nbs/hypothesis_sweep_v9.py](../nbs/hypothesis_sweep_v9.py): main hypothesis catalog, activation and weight scores, block-local scope diagnostic, cross-adapter support.
- [nbs/functional_projection_v10.py](../nbs/functional_projection_v10.py): causal projection/complement test of trained residual-write `dW` into activation-PCA bases.
- [nbs/v10_llama.py](../nbs/v10_llama.py): Wendler-style token-energy and logit-lens functional metrics.
- [nbs/strong_conclusion_v4.py](../nbs/strong_conclusion_v4.py): older positive result discipline for `write_not_read` vs TaskDiff before v9/v10 tightened the interpretation.

Local qmd search sources:

- `qmd search 'weight steering SVD subspace activation attention'` found local notes on SVD steering and adapter parametrization.
- `qmd search 'LoRA PiSSA DeLoRA OFT IA3 parameterization steering subspace'` found the adapter-as-hypothesis catalog.
- A vector/reranked qmd query OOMed the local GPU, so this catalog uses BM25 qmd results plus workspace notebooks.

External refinement sources added after the first catalog pass:

- `logs/research/20260427_external_hypotheses_qmd.out`: qmd BM25 searches for function vectors, SAE steering, concept induction, and adapter subspaces.
- `logs/research/20260427_external_hypotheses_hf.out`: HF Papers searches. Important hits include Function Vectors, Task Arithmetic, AxBench, SAE steering, MSRS, attention-output low-dimensional subspaces, and LoRA spectral methods.
- `logs/research/20260427_external_hypotheses_bibtex.bib`: semantic-search BibTeX output. Initial run failed on missing `rapidfuzz`; rerun with `uv run --with rapidfuzz` succeeded.
- `logs/research/20260427_external_hypotheses_selected_info.out`: HF metadata for selected papers.
- `logs/research/20260427_external_hypotheses_qmd_excerpts.out`: qmd excerpts for AxBench, SAE feature-flow notes, and dual-route/function-vector notes.

External papers used as hypothesis generators, stated as authors' claims unless already validated here:

| paper/source | search signal | relevant claim for this repo | hypothesis consequence |
|---|---|---|---|
| Todd et al., Function Vectors in Large Language Models, arXiv:2310.15213 | HF search + metadata, repo has 195 stars | Authors claim middle-layer attention heads transport compact task/function vectors with causal effects and compositionality. | Split concept vector from function/instruction vector; test FV-head output subspace separately from residual TaskDiff. |
| Ilharco et al., Editing Models with Task Arithmetic, arXiv:2212.04089 | HF search + metadata, repo has 538 stars | Authors claim weight-space task vectors compose by addition/negation and analogy. | Treat `dW` as a task vector family; add sign/analogy/arithmetic tests across adapters and behaviors. |
| Wu et al., AxBench, arXiv:2501.17148 | qmd + HF metadata | Authors claim prompting/finetuning beat most steering methods; difference-in-means is strong for concept detection; SAEs are not competitive in their benchmark. | Keep prompt and activation baselines honest; do not over-privilege SAE/PCA interpretability if simple DiffMean wins. |
| Arad et al., SAEs Are Good for Steering, arXiv:2505.20063 | HF metadata | Authors claim SAE steering improves after filtering features by output score; input features and output features often differ. | If testing SAE bases, use output-score filtering and allow signed negative projections; raw activation-frequency features are the wrong basis. |
| Mayne et al., Can sparse autoencoders decompose steering vectors?, arXiv:2411.08790 | HF metadata | Authors claim SAE decompositions of steering vectors can mislead because steering vectors are out of SAE input distribution and need negative feature projections. | SAE analysis should decompose signed vectors in decoder space, not just positive latent activations. |
| Jiang et al., MSRS, arXiv:2508.10599 | qmd + HF metadata | Authors claim multi-attribute steering benefits from orthogonal private subspaces plus a shared subspace and token-level dynamic weighting. | Replace one global basis with shared/private basis split for sycophancy vs honesty transfer and per-token weighting. |
| Wang et al., Attention Layers Add Into Low-Dimensional Residual Subspaces, arXiv:2508.16929 | HF metadata | Authors claim attention outputs live in low-dimensional subspaces induced by `W_o` and this affects SAE dead features. | Use attention-output active subspace as a causal basis and as an SAE initialization/control. |
| Park et al., The Information Geometry of Softmax, arXiv:2602.15293 | semantic-search BibTeX + HF metadata | Authors claim softmax information geometry gives a natural probe/steering geometry and propose dual steering to minimize off-target concept changes. | Use Fisher/softmax-metric projection instead of Euclidean `P_B` for logit-facing steering bases. |

Deferred external leads, not yet promoted to main hypotheses:

- `Causality != Invariance: Function and Concept Vectors in LLMs` from the BibTeX search is an anti-overclaiming control: vector invariance or compositionality is not enough; use causal keep/drop or patching.
- `Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens` from the BibTeX search reinforces the v10 warning: do not reject a function/control route only because immediate Yes/No readout is weak.
- `Spherical Steering` is probably a geometry variant of the existing rotation/OFT and softmax-geometry rows, not a separate experiment yet.
- `What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal` may be a closer behavioral analogue for honesty/sycophancy than translation papers; worth reading before final paper framing.
- qmd feature-flow and FGAA notes suggest a cross-layer SAE feature-flow hypothesis, but it needs pretrained SAEs. Defer unless SAE artifacts are available for Qwen3-0.6B.

## Current empirical bottom line

The old positive framing was: A-side recipes like `write_not_read` or TaskDiff may recover the LoRA steering label. The v9/v10 update is stricter:

- Across adapter families, most tested linear bases capture only about 1 to 8 percent of the relevant rank-matched oracle.
- Block-local activation PCA did not fix the mismatch between activation oracles and weight oracles.
- Causal projection shows activation-PCA directions can be potent if amplified, but for the strongest adapter, DeLoRA, the trained-scale behavior mostly lives in the complement.
- Wendler-style probes suggest LoRA-layer `Δh` is concept-space, not directly Yes/No readable. Downstream layers translate it into the Yes/No or honesty behavior.

So a basis can be useful for steering without being an explanation of the trained adapter. This distinction matters for every row below.

External-search update: the outside literature mostly pushes in the same direction, but only as hypothesis generation for this DD setting. AxBench-like results warn that simple DiffMean/ReFT-style baselines can beat prettier mechanistic bases for steering; function-vector and concept-induction work says task/function transport can be head-local and not logit-readable; SAE steering needs output-causal feature selection, not raw activation-feature labels; task arithmetic says the trained `dW` family itself may be the right algebraic object.

## Notation

Let `h_l` be a residual-stream vector at layer `l`, `W_l` be a pretrained linear map, and `dW_l` be the trained adapter delta for that map.

For a basis `B ∈ R^{d x k}` with orthonormal columns:

```py
P_B = B @ B.T
keep_B(dW) = P_B @ dW
drop_B(dW) = dW - P_B @ dW
energy_frac(x, B) = ||P_B x||^2 / ||x||^2
subspace_overlap(A, B) = ||A.T @ B||_F^2 / min(rank(A), rank(B))
```

For a weight matrix SVD:

```py
U, S, Vh = svd(W)              # W: d_out x d_in
left_basis = U[:, :k]          # output/write directions
right_basis = Vh[:k].T         # input/read directions
S_coords(dW) = U.T @ dW @ Vh.T # adapter delta in W's singular-vector basis
```

## Trained-delta labels and oracles

These are useful for analysis but must not be presented as from-scratch steering recipes.

| name | construction | interpretation | steering? | fork-plan fit |
|---|---|---|---|---|
| `w_oracle` | `left_svd_basis(concat(dW_o_proj, dW_down_proj), k)` | The best rank-`k` residual-output basis for the trained local weight delta. | Not from scratch. Can steer only because it uses trained `dW`. | Cross-adapter `dW` basis ablation. Use as per-adapter top basis and as sanity ceiling. |
| `act_oracle` | `pca(normalize(h(+α) - h(-α)), k)` on eval activations | Best rank-`k` activation basis for the trained adapter effect on the sampled prompts. | Not fair from scratch if built from trained steering. Can be an intervention target. | New causal test or v10-style projection ablation, not cross-adapter shared `dW` unless converted to `P_act dW`. |
| `act_oracle_block` | `pca(normalize((post_pos-pre_pos) - (post_neg-pre_neg)), k)` | Scope-matched local block contribution instead of cumulative residual effect. | Same as `act_oracle`. | v10 projection/complement test. Helps check whether scope mismatch was the bug. |
| `TaskDiff_lora_fit` | PCA of trained adapter `h(+α)-h(-α)` on FIT prompts, scored on EVAL prompts | Held-out answer key for whether a learned effect generalizes across prompts. | Not from scratch. | Useful diagnostic for activation-steering upper bound and concept-space rank. Not a planned `dW` ablation by itself. |

Pseudocode:

```py
def b_side_oracles(model, dW, prompts_fit, prompts_eval, k):
    h_pos_fit = capture(model + dW, prompts_fit, α=+1)
    h_neg_fit = capture(model + dW, prompts_fit, α=-1)
    h_pos_eval = capture(model + dW, prompts_eval, α=+1)
    h_neg_eval = capture(model + dW, prompts_eval, α=-1)

    B_task_lora = pca(h_pos_fit - h_neg_fit, k)
    B_act_eval = pca(unit_rows(h_pos_eval - h_neg_eval), k)
    B_w = left_svd_basis(concat_residual_writer_dW(dW), k)
    return B_task_lora, B_act_eval, B_w
```

Positive readout: an A-side candidate approaches these oracles and keeps behavior under causal keep/drop. Negative readout: high geometric score does not preserve behavior, or low geometric score still steers when amplified.

## Activation hypotheses

### Function-vector head basis

Construction:

```py
for head in attention_heads:
    fv_head_score = causal_patch_score(head_output, task_prompt_pairs)
top_heads = topk(fv_head_score, k_heads)
B_fv = pca(stack([OV_output_basis(head) for head in top_heads]), k)
```

Interpretation: sycophancy/honesty steering may decompose into a concept vector plus a function or instruction vector. The function vector is task-level control like "answer honestly" or "agree with the user", and can be transported by a small set of middle-layer attention heads. Todd et al. claim function vectors are robust across contexts and compositional; Feucht et al. treat FV heads as distinct from concept-induction heads.

Steering use: yes as activation steering or head-output patching. It is probably a better fit for "what task is being done?" than for "what semantic concept is active?".

Fork-plan fit: new causal test, plus layer/module ablation if the trained `dW` concentrates in the `o_proj` rows for top FV heads. Add `fv_heads_only`, `non_fv_heads_only`, and `drop_fv_heads` rows if head-level masking is implemented.

Positive readout: FV-head patch changes instruction/function while preserving topic content, while same-layer random heads and concept-head controls do not. Negative readout: FV basis is just another dense TaskDiff basis and does not localize to heads.

### Concept-induction vs function-vector split

Construction:

```py
B_concept = pca(outputs(top_concept_induction_heads, semantic_copy_prompts), k)
B_function = pca(outputs(top_function_vector_heads, task_demonstration_prompts), k)
score = behavior(model, patch(B_concept)) - behavior(model, patch(B_function))
```

Interpretation: the current "concept-space" language in v10 may be underspecified. Dual-route induction suggests at least two soft-induction routes: concept heads transport what entity/concept is being discussed, while FV heads transport what transformation/task should be applied. Sycophancy may be a function-vector failure more than a concept-vector failure.

Steering use: yes, but the intervention should be head-local or route-local rather than a generic residual addition.

Fork-plan fit: new causal test. It also refines attention min/max/diff: token identity logging should distinguish concept tokens, instruction tokens, and answer-format tokens.

Positive readout: crossed dissociation. Concept-head patch changes target concept/topic with the same output policy; FV-head patch changes policy/instruction with the same concept/topic. Negative readout: both patches only move a generic sycophancy logit ratio.

### ReFT-r1 / supervised rank-1 representation finetuning baseline

Construction:

```py
r = train_rank1_reft(site=l, positives=honest_rows, negatives=sycophantic_rows)
h_l_steered = h_l + α * r.left @ (r.right.T @ h_l)
```

Interpretation: AxBench authors claim weakly supervised rank-1 representation finetuning is competitive while remaining more interpretable than prompting. This is a stronger fair activation baseline than unsupervised PCA if we allow a small supervised validation set.

Steering use: yes, but it is a learned activation intervention, not a trained weight-delta explanation.

Fork-plan fit: activation-steering baseline. It should be compared against TaskDiff and prompt baselines on identical DD rows. It should not use trained `w.pt`.

### SAE output-score signed feature basis

Construction:

```py
features = sae.encode(h_l)
input_score_j = corr(features[:, j], concept_label)
output_score_j = causal_effect(decoder[:, j], target_logit_or_behavior)
selected = [j for j in features if input_score_j > τ_in and output_score_j > τ_out]
B_sae_out = orth(decoder[:, selected] * sign(output_effect[selected]))
```

Interpretation: raw SAE activations are not enough. Arad et al. claim steering improves after selecting features with output-causal scores; Mayne et al. claim steering-vector SAE decomposition is misleading when it ignores negative feature projections. The hypothesis is that v9 PCA misses a sparse signed feature basis that is output-causal but not high-variance.

Steering use: possible. It should be tested only with signed decoder directions and output-score filtering.

Fork-plan fit: new causal test or activation-steering baseline. If converted to `P_B dW`, it becomes a trained-scale carrier test. Do not put raw SAE latent activations into the core fork-plan without output-score filtering and signed negative-projection controls.

Positive readout: output-score SAE basis steers at lower norm or lower degradation than DiffMean/TaskDiff/ReFT-r1, and ablations show both output-score filtering and signed negative projections matter. Negative readout: DiffMean or ReFT-r1 still dominates, matching AxBench's warning.

### MSRS-style shared/private steering

Construction:

```py
B_shared = intersection_or_joint_svd([B_sycophancy, B_honesty, B_refusal])
B_private_task = orth(B_task - project(B_task, B_shared))
α_tokens = router(token_features)  # optional token-level weights
h_l += α_shared * P_shared @ v + α_private * P_private_task @ v
```

Interpretation: MSRS authors claim multi-attribute steering benefits from orthogonal private subspaces plus a shared subspace and token-level dynamic weighting. For this repo, the transfer target is sycophancy training to daily-dilemmas honesty. The shared/private split is a concrete alternative to one global TaskDiff.

Steering use: yes. It is especially relevant if sycophancy and honesty share some moral-agreement axis but differ in prompt/style axes.

Fork-plan fit: activation-steering baseline and cross-adapter shared `dW` ablation. Keep two variants distinct: `MSRS_activation_shared_private` for activation steering, and `dW_shared_private_transfer` for trained deltas. The trained-delta version is `B_shared` across adapters/behaviors plus adapter-private residuals.

Positive readout: shared basis preserves transfer to DD while private basis preserves sycophancy eval; mixing them beats global TaskDiff/shared SVD and private-only baselines on the transfer/degradation frontier. Negative readout: shared/private split adds complexity without improving that frontier.

### Softmax information-geometry steering

Construction:

```py
J = jacobian(log_softmax(W_U @ h), h)       # or Fisher metric approximation
G = J.T @ diag(p) @ J                       # local softmax/Fisher metric
P_B_G = B @ inv(B.T @ G @ B) @ B.T @ G
h_steered = h + α * P_B_G @ v              # or project dW outputs with G metric
```

Interpretation: this is a projection metric variant, not a new basis family. Euclidean projection may be the wrong geometry for logit-facing behavior. Park et al. claim softmax information geometry gives a natural steering metric and dual steering can change a target concept while minimizing off-target changes. This directly addresses the current degradation concern.

Steering use: yes for logit-facing activation steering. It may be less useful for hidden concept-space layers where `W_U` is not the immediate readout.

Fork-plan fit: activation-steering baseline and new projection/complement variant. Replace Euclidean `P_B dW` with Fisher/softmax-metric projection and compare behavior/degradation.

Positive readout: for the same basis, same norm, and same target effect, Fisher/softmax projection has lower off-target DD or lower perplexity degradation than Euclidean projection. Negative readout: no improvement at LoRA layers because the relevant concept is not yet logit-facing.

### TaskDiff contrast

Construction:

```py
h_pos = capture(base_model, persona_pos_prompts)
h_neg = capture(base_model, persona_neg_prompts)
B_task = pca(h_pos[l] - h_neg[l], k)
```

Interpretation: the target behavior is linearly separable in base residual activations under contrastive personas. This is the standard RepE/ActAdd-style story.

Steering use: yes. This is the main activation-steering baseline, if built from training prompts only and selected on held-out validation rows.

Fork-plan fit: activation-steering baseline. It can also seed synthetic `dW'` if converted into a write direction through pretrained writer maps, but then it is a constructive baseline, not an ablation of trained `dW`.

Failure mode: persona contrast may capture style or role behavior, not the sycophancy/honesty mechanism. v10 found persona contrast weakly captures the Yes/No axis. For a fair activation-steering benchmark, fail if code loads `w.pt` before constructing this basis.

### Suppressed or turnover subspace

Construction:

```py
mag = abs(h_clean).permute(layer, batch, dim)
Δmag = mag[1:] - mag[:-1]
added = relu(Δmag).sum(layer)
removed = relu(-Δmag).sum(layer)
B_suppressed = pca(min(added, removed), k)
```

Interpretation: planning or scratchpad features are written and later erased before final readout. This is a plausible hidden-computation basis.

Steering use: maybe. It can define an activation intervention or a synthetic `dW'` target, but v9 did not show it explains trained `dW`.

Fork-plan fit: not a direct planned `dW` basis unless applied as `P_suppressed dW` in a v10-style projection/complement causal test. Better fit: synthetic `dW'` or new causal test.

Positive readout: `keep_B(dW)` preserves behavior and `drop_B(dW)` removes it. Negative readout: both projection and complement steer, or projection is weak at trained norm.

### Stenographic subspace

Construction:

```py
B_task = pca(h_pos - h_neg, k)
B_suppressed = pca(turnover(h_clean), k)
B_steno = intersect_bases(B_task, B_suppressed, min_overlap=τ)
```

Interpretation: task-relevant signal lives in directions the model also tends to suppress. This was the strongest form of the hidden-planning story in early notebooks.

Steering use: possible but rank may collapse. Use as an activation steering or synthetic write target, not as evidence about trained `dW` without causal ablation.

Fork-plan fit: new causal test or synthetic `dW'`. If projected onto trained `dW`, it becomes a v10-style trained-scale carrier test.

### Churn

Construction:

```py
B_churn_l = pca(h_clean[l + 1] - h_clean[l], k)
```

Interpretation: important computation lives where the residual stream changes most across layers, not where static activations have high variance.

Steering use: maybe, but broad and likely nonspecific.

Fork-plan fit: synthetic `dW'` or activation-steering baseline. For trained `dW`, use projection/complement as an extra causal test, not one of the three core fork-plan ablations.

### Amplified and added features

Construction:

```py
B_amplified = pca(relu(abs(h_clean[last]) - abs(h_clean[first])), k)
B_added = pca(relu(abs(h_clean[1:]) - abs(h_clean[:-1])).sum(layer), k)
```

Interpretation: useful behavior may ride features that are progressively amplified, not features that are written then erased.

Steering use: weak prior. It is a broad activation prior rather than a behavior-specific hypothesis.

Fork-plan fit: synthetic `dW'` or activation-steering exploratory baseline. Not a core trained-`dW` ablation unless used as `P_B dW`.

### Global clean and persona residual PCA

Construction:

```py
B_clean = pca(stack_layers_and_prompts(h_clean), k)
B_persona = pca(stack(h_persona_pos, h_persona_neg), k)
```

Interpretation: behavior lies in high-variance background residual directions. This is mostly a control.

Steering use: probably poor as a specific steerer.

Fork-plan fit: random/control-like row for synthetic `dW'` or activation steering. It should not be central unless it unexpectedly beats task-specific bases.

### Attention-selected TaskDiff: min, max, diff, min times norm

Construction:

```py
attn_pos = final_token_attention(persona_pos)
attn_neg = final_token_attention(persona_neg)
tok_diff = h_pos_tokens - h_neg_tokens

B_attn_min = pca(sum_tokens(min(attn_pos, attn_neg) * tok_diff), k)
B_attn_max = pca(sum_tokens(max(attn_pos, attn_neg) * tok_diff), k)
B_attn_diff = pca(sum_tokens(abs(attn_pos - attn_neg) * tok_diff), k)
B_attn_min_norm = pca(sum_tokens(min(attn_pos, attn_neg) * norm(tok_diff) * tok_diff), k)
```

Interpretation:

- `attn_min_taskdiff`: shared attended tokens carry the stable plan.
- `attn_max_taskdiff`: any strongly attended token can carry the plan.
- `attn_diff_taskdiff`: changes in attention routing are themselves the signal.
- `attn_min_x_diffnorm_taskdiff`: shared attention matters, but high-contrast tokens get more weight.

Implementation caveat: v9 scores these as linear spans after attention-weighted aggregation. It does not prove which token type carried the signal. A real causal test should log token indices and token strings for the min/max/diff weights, e.g. final token, delimiter, question token, or persona token, and then perturb the selected attention route.

Steering use: yes as activation steering, especially if last-token extraction is too narrow.

Fork-plan fit: activation-steering baseline or new token-conditional causal test. It does not fit current layer/module `dW` ablation unless converted into `P_B dW` for residual writers.

Positive readout: attention-weighted basis beats unweighted TaskDiff on held-out behavior and projection. Negative readout: attention weights select formatting or prompt tokens and do not steer.

Refinement from MSRS: the token weights should be learned or validated by behavior, not only borrowed from raw attention. Add a matched comparison between attention-derived weights and a small router trained on token type or logit effect.

### Up-proj input contrast

Construction:

```py
x_up_pos = capture_input(model.layers[l].mlp.up_proj, persona_pos)
x_up_neg = capture_input(model.layers[l].mlp.up_proj, persona_neg)
B_up_input = pca(x_up_pos - x_up_neg, k)
```

Interpretation: the behavior is represented in the features read by the MLP expansion before nonlinear gating.

Steering use: activation steering at MLP inputs, or synthetic `dW'` into `up_proj` or `gate_proj`.

Fork-plan fit: layer/module ablation if it motivates an `up/gate` row. Synthetic `dW'` if constructing from base activations. Not covered by residual-write-only v10 projection.

### Up-proj output written contrast

Construction:

```py
u_pos = up_proj(x_up_pos)
u_neg = up_proj(x_up_neg)
B_up_written = pca((u_pos - u_neg) @ W_down.T, k)
```

Interpretation: the MLP expansion difference matters only after being mapped back to residual space.

Steering use: plausible residual-write target.

Fork-plan fit: layer/module ablation, especially `mlp_down_proj_only`, and synthetic `dW'` via MLP write maps.

### Gate-active written

Construction:

```py
gate = silu(h_clean @ W_gate.T)
up = h_clean @ W_up.T
B_gate_active = pca((gate * up) @ W_down.T, k)
```

Interpretation: target behavior may live in active gated MLP features rather than raw read/write SVD directions.

Steering use: yes, but likely nonlinear and input-dependent.

Fork-plan fit: layer/module ablation if `up/gate` or MLP modules carry behavior. New causal test if using token/input-conditional gates, because fixed linear keep/drop loses the nonlinearity.

### CHaRS-style clusters

Construction:

```py
H = concat(h_clean, h_persona_pos, h_persona_neg)
centroids = kmeans(H, n_clusters=k)
B_chars = pca(centroids - mean(centroids), k)
```

Interpretation: concept behavior is a cluster or manifold, not a single linear direction. PCA of centroids is a lossy linearization.

Steering use: maybe strong if implemented as per-cluster translations, weak if collapsed to one global span.

Fork-plan fit: new causal test. It does not fit current linear `dW` basis ablations unless deliberately linearized.

### Rotation contrast or Procrustes generator

Construction:

```py
J = pca(concat(h_neg, h_pos), rank)
X = center(h_neg) @ J
Y = center(h_pos) @ J
U, _, Vh = svd(X.T @ Y)
R = U @ Vh
B_rot = J @ left_svd_basis(R - R.T, k)
```

Interpretation: the persona contrast is better described as a rotation in a local concept manifold than as a translation.

Steering use: yes, but the intervention should be rotational, not additive activation steering.

Fork-plan fit: adapter-parameterization inspiration, especially OFT/AntiPaSTO, or a new rotation causal test. Not a natural fit for plain keep/drop unless converted to rotation-derived `dW`.

### Wendler concept-space functional probes

Construction:

```py
Δh_l = mean(h_l(+α) - h_l(-α), prompts)
E2(Δh_l) = (vocab / d) * ||U_hat @ Δh_l||^2 / ||U_hat.T @ U_hat||_F^2
cap_yn(B) = ||P_B(e_yes - e_no)||^2 / ||e_yes - e_no||^2
ldiff(B, Δh) = (e_yes - e_no).T @ P_B @ Δh
```

Interpretation: the LoRA may write a concept that is not directly readable as Yes/No until downstream layers. This tests readout visibility rather than subspace overlap.

Steering use: no direct steering basis by itself, but it tells which layer to steer or probe.

Fork-plan fit: new causal test and benchmark diagnostics. It should be added as an analysis column for layer/module ablation, because a slice that changes behavior may still be invisible to the immediate logit lens.

## Pretrained-weight bases

### Attention-output active subspace

Construction:

```py
for layer in layers:
    A_out = capture(self_attn.o_proj output, clean_prompts)[layer]
    B_attn_active_l = pca(A_out, k)
    B_attn_Wo_l = left_svd_basis(W_o_l, k)
    B_attn_active_intersect_l = intersection(B_attn_active_l, B_attn_Wo_l)
```

Interpretation: Wang et al. claim attention outputs occupy a surprisingly low-dimensional residual subspace induced by the output projection. This makes a sharper version of `attn_o_proj_only`: the question is not just whether attention output matters, but whether the trained adapter uses the active attention-output subspace or off-manifold attention-output directions.

Steering use: yes as a synthetic attention-write basis and as an SAE initialization/control.

Fork-plan fit: layer/module ablation and synthetic `dW'`. Add `P_attn_active dW_attn_o` vs complement if attention-only rows are positive.

Positive readout: `P_attn_active dW` keeps attention-mediated behavior and complement loses it, and active PCA beats both `attn_o_proj_only` and structural `W_o` left-SVD controls. Negative readout: active subspace is indistinguishable from a generic attention-module ablation or trained adapter steers via off-manifold `W_o` directions.

### `lm_head_read` and `logits_null` or weak readout

Construction:

```py
U, S, Vh = svd(W_unembed)
B_lm_read = Vh[:k].T
B_logits_null = Vh[-k:].T
```

Interpretation: `lm_head_read` is the canonical readable residual subspace; `logits_null` is weakly read out by the unembedding.

Steering use: yes for simple readout steering, but v10 suggests concept steering does not live here at LoRA layers.

Fork-plan fit: synthetic `dW'` baseline, activation-steering control, and possible `write-not-read` construction. Not a likely trained-`dW` carrier unless `P_lm dW` keeps behavior.

### Global read

Construction:

```py
G_read = sum_l(W_q_l.T @ W_q_l + W_k_l.T @ W_k_l + W_v_l.T @ W_v_l
               + W_up_l.T @ W_up_l + W_gate_l.T @ W_gate_l)
G_read += W_unembed.T @ W_unembed
B_global_read = eig_top(G_read, k)
```

Interpretation: residual directions broadly read by attention, MLP, and unembedding across the model.

Steering use: maybe as a safe/readable direction, but broad and nonspecific.

Fork-plan fit: synthetic `dW'` and controls. It is also the forbidden subspace for `global_write_not_global_read`.

### Global write

Construction:

```py
W_write_all = concat_cols([W_o_l, W_down_l for all layers])
B_global_write = left_svd_basis(W_write_all, k)
```

Interpretation: directions the model can easily write into residual stream across all layers.

Steering use: plausible but nonspecific.

Fork-plan fit: synthetic `dW'`, random/control-like global basis, or cross-adapter ablation if intersected with trained `dW` residual writers.

### Global write not global read

Construction:

```py
P_read = B_global_read_broad @ B_global_read_broad.T
B_gwnr = left_svd_basis((I - P_read) @ W_write_all, k)
```

Interpretation: globally writeable directions that are not in the dominant global read subspace. This is a model-level stenographic candidate.

Steering use: yes as synthetic `dW'` or activation intervention. It may be high-gain if downstream nonlinear paths read it despite low linear readout.

Fork-plan fit: synthetic `dW'` and optional projection/complement trained-`dW` test. If `keep_B(dW)` works and `drop_B(dW)` fails, it supports a write-not-read causal story.

### Per-layer write, attention write, and MLP write

Construction:

```py
B_write_l = left_svd_basis(concat_cols(W_o_l, W_down_l), k)
B_attn_write_l = left_svd_basis(W_o_l, k)
B_mlp_write_l = left_svd_basis(W_down_l, k)
```

Interpretation: layer-local residual write capacity, split by attention and MLP writers.

Steering use: yes for synthetic `dW'`; also direct causal ablation of trained `dW` by module.

Fork-plan fit: layer/module ablation. Required rows already include `attn_o_proj_only`, `mlp_down_proj_only`, and `residual_write_only`.

### Write not read: lm-head, global, downstream

Construction:

```py
B_wnr_lm_l = left_svd_basis((I - P_lm_read_broad) @ concat_cols(W_o_l, W_down_l), k)
B_wnr_global_l = left_svd_basis((I - P_global_read_broad) @ concat_cols(W_o_l, W_down_l), k)
B_wnr_downstream_l = left_svd_basis((I - P_downstream_read_l) @ concat_cols(W_o_l, W_down_l), k)
```

Interpretation: layer writes into directions not immediately read by a chosen downstream read model. This was an early strongest A-side recipe but v9/v10 weaken the explanatory claim.

Steering use: yes. This is one of the best synthetic `dW' candidates because it is purely pretrained and module-local.

Fork-plan fit: synthetic `dW' baseline first. As trained-`dW` ablation, use `P_B dW` and complement rows. It is not already in the three core ablations, but it is a natural extension of layer/module causal ablation.

### MLP up-read and gate-read

Construction:

```py
B_up_read_l = right_svd_basis(W_up_l, k)
B_gate_read_l = right_svd_basis(W_gate_l, k)
```

Interpretation: behavior is represented in residual directions read by the MLP expansion or gate.

Steering use: likely as input activation steering or synthetic input-side `dW`, less direct for residual-output `dW`.

Fork-plan fit: layer/module ablation if `up/gate` modules carry behavior. Adapter-parameterization ablation for IA3 MLP gates.

### Attention QKV read and input superposition

Construction:

```py
B_qkv_read_l = right_svd_basis(concat_rows(W_q_l, W_k_l, W_v_l), k)
B_input_super_l = right_svd_basis(concat_rows(W_q_l, W_k_l, W_v_l, W_up_l, W_gate_l), k)
B_kv_super_l = right_svd_basis(concat_rows(W_k_l, W_v_l), k)
```

Interpretation: the steering-relevant state is in what attention or all input-side modules read, rather than what residual writers output.

Steering use: activation steering at module inputs or synthetic `dW` for read-side matrices.

Fork-plan fit: layer/module ablation if q/k/v/up/gate trained deltas matter. Not scored by residual-output-only v10, so include read-side trained `dW` rows if this hypothesis matters.

### Merged K and Q, `qk_circuit`

Construction:

```py
K_expanded = repeat_kv_rows_to_match_q_heads(W_k_l, W_q_l.shape[0])
B_qk_l = left_svd_basis(W_q_l.T @ K_expanded, k)
```

Interpretation: planning routes through attention score geometry, the bilinear interaction between queries and keys, not through values or residual writes alone. This is the requested K/Q merge hypothesis.

Steering use: not as a simple residual write. Better as a causal attention-routing intervention or trained q/k module ablation.

Fork-plan fit: layer/module ablation if q/k deltas are kept/dropped. Otherwise new causal test: perturb QK score subspace and measure behavior. v9 includes `qk_circuit` as a geometric candidate, but that is weaker than a QK causal intervention.

### Attention OV write

Construction:

```py
V_expanded = repeat_kv_rows_to_match_o_heads(W_v_l, W_o_l.shape[1])
B_ov_l = left_svd_basis(W_o_l @ V_expanded, k)
```

Interpretation: attention writes behavior through the value-to-output circuit, not through QK selection.

Steering use: plausible residual write target because `W_o W_v` maps token content into residual output.

Fork-plan fit: layer/module ablation, especially attention-only rows. Synthetic `dW'` if signed by persona contrast.

### MLP roundtrip

Construction:

```py
B_mlp_roundtrip_l = left_svd_basis(W_down_l @ W_up_l, k)
```

Interpretation: residual-to-MLP-to-residual linear path captures the relevant feature transformation.

Steering use: yes as an MLP synthetic basis, with the caveat that real MLPs are gated and nonlinear.

Fork-plan fit: layer/module ablation and synthetic `dW'`. If this beats attention rows, the paper story moves toward feature-space MLP steering.

### Gate kernel

Construction:

```py
mean_gate = mean(silu(h_clean @ W_gate.T), batch)
B_gate_kernel_l = left_svd_basis(W_down_l @ diag(mean_gate) @ W_up_l, k)
```

Interpretation: gated MLP write under average activation is the relevant local linearization.

Steering use: yes, more plausible than raw `W_down W_up` because it includes the gate.

Fork-plan fit: layer/module ablation and new nonlinear causal test. Current trained-`dW` ablation should include `gate/up/down` splits to avoid missing it.

### Attention sink

Construction:

```py
for head in heads:
    sink_vec_h = W_o_l[:, head_slice] @ (W_v_l[kv_head_slice] @ embedding(BOS))
B_sink_l = pca(stack(sink_vec_h), k)
```

Interpretation: special sink or BOS-mediated attention channels carry planning or control state.

Steering use: speculative. Could steer attention sink features, but risk selecting formatting/sink artifacts.

Fork-plan fit: new attention-specific causal test, not a core `dW` basis ablation unless trained attention output deltas project strongly here.

### Causally isolated write

Construction:

```py
B_forbidden = union(B_input_super_l, B_kv_super_l, B_lm_read_broad)
B_isolated_l = project_write_away(concat_cols(W_o_l, W_down_l), B_forbidden)
```

Interpretation: directions writeable by the layer but isolated from obvious read, KV, and lm-head axes. Strong hidden-channel hypothesis.

Steering use: yes as a synthetic hidden-write intervention.

Fork-plan fit: synthetic `dW'` and new causal projection/complement test. A positive trained-`dW` result would be interesting because it says the learned adapter uses hidden write channels.

## Learned-weight and adapter parameterization hypotheses

### Weight-space task arithmetic

Construction:

```py
dW_task = W_finetuned_task - W_base
dW_negated = -dW_task
dW_composed = dW_honesty + dW_anti_sycophancy
dW_analogy = dW_A_to_B + dW_C - dW_A
```

Interpretation: task arithmetic authors claim task vectors in weight space can be negated, added, and used in analogy-like combinations. Weight steering is already a task-vector method, but the current fork plan mostly tests subspace carriers, not algebra. This hypothesis says the meaningful object may be the full signed `dW` vector and its arithmetic across behaviors/adapters.

Steering use: yes. This is directly weight steering.

Fork-plan fit: cross-adapter causal `dW` ablation and future multi-behavior benchmark. Mark as future until there are at least two behavior diffs: `dW_honesty`, `dW_anti_sycophancy`, `dW_refusal`, etc. A sign test can be run earlier only if positive and negative adapters are independently meaningful.

Positive readout: `dW_a + dW_b` approximately adds behavioral deltas without extra degradation; `-dW` reverses the target behavior more cleanly than random sign, permuted-layer, and random-norm controls. Negative readout: composition fails because adapters exploit incompatible basins or layer/module supports.

These are the hypotheses most directly aligned with the active fork-plan ablations.

### LoRA low-rank delta

Construction:

```py
dW = B @ A
W_steered = W + α * dW
```

Interpretation: the behavior delta is low-rank in ordinary weight coordinates.

Steering use: yes, this is current baseline.

Fork-plan fit: cross-adapter SVD, per-adapter SVD, rank-component parameterization ablation, multi-seed benchmark.

### DoRA magnitude vs direction

Construction:

```py
V = W + α * (B @ A)
scale = m / stopgrad(norm(V, dim=output_axis))
W_eff = scale * V
```

Interpretation: magnitude and direction of weight vectors are separate causal degrees of freedom.

Steering use: yes, but current results say DoRA behaves similarly to LoRA on this task.

Fork-plan fit: adapter-parameterization ablation: keep/drop direction component vs magnitude component.

### DeLoRA decoupled rank directions and strengths

Construction:

```py
scale_i = λ_i / (rank * ||A_i|| * ||B_i||)
dW = B @ diag(scale) @ A
```

Interpretation: the coherent behavioral axis is angular direction plus explicit strength. Current repo evidence: strongest raw steerer and best negative coefficient symmetry, but not explained by tested activation PCA.

Steering use: yes, strongest current raw method.

Fork-plan fit: adapter-parameterization ablation. Split rank directions, λ strengths, top/bottom S-space energy, and compare to residual complement.

### PiSSA top SVD subspace

Construction:

```py
U, S, Vh = svd(W)
W_res = U[:, r:] @ diag(S[r:]) @ Vh[r:]
adapter = U[:, :r] @ diag(S[:r]) @ Vh[:r]
train(adapter)
```

Interpretation: pretrained top singular directions are the useful adaptation manifold. Current repo evidence: clean stable baseline, often high steering without DeLoRA saturation.

Steering use: yes.

Fork-plan fit: adapter-parameterization ablation with S-space quartiles and energy crops. Also cross-adapter shared SVD if PiSSA top components overlap other adapters.

### OFT rotation

Construction:

```py
A_skew = skew(params)
R = cayley(A_skew)
W_eff = W @ R.T
dW = W_eff - W
```

Interpretation: behavior can be changed by rotating pretrained features while preserving norms/angles.

Steering use: yes, but current raw effect is weaker than PiSSA/DeLoRA.

Fork-plan fit: adapter-parameterization ablation: rotation-derived component vs residualized effective update.

### IA3 gates

Construction:

```py
if feedforward:
    y = W @ (x * λ)
else:
    y = (W @ x) * λ
```

Interpretation: adaptation is gain control over existing channels.

Steering use: weak in current daily-dilemmas results, but useful lower bound.

Fork-plan fit: adapter-parameterization ablation: attention-gate vs MLP-gate groups. Layer/module ablation if gates identify modules rather than full tensors.

### Shared cross-adapter `dW` SVD

Construction:

```py
M_l = concat_cols([dW_adapter_l for adapter in adapters])
B_shared_l_K = left_svd_basis(M_l, K)
keep = P_B @ dW_adapter_l
drop = dW_adapter_l - P_B @ dW_adapter_l
```

Interpretation: different adapter families discover the same causal residual-write subspace.

Steering use: not from scratch, but if shared `keep` steers across families, it is the main planning-subspace evidence.

Fork-plan fit: central row of cross-adapter causal `dW` basis ablation. Positive result needs `keep_B_shared_K32` retain at least 0.7x behavior and `drop_B_shared_K32` remove it across adapters.

Refinement from task arithmetic and MSRS: separate shared-across-adapter from shared-across-behavior. A basis can be adapter-family invariant but behavior-specific, or behavior-general but adapter-specific. Use two axes: `B_shared_adapter(behavior)` and `B_shared_behavior(adapter)`.

### Per-adapter top and tail SVD

Construction:

```py
U, S, Vh = svd(dW_adapter_l)
dW_topK = U[:, :K] @ diag(S[:K]) @ Vh[:K]
dW_tail = U[:, K:] @ diag(S[K:]) @ Vh[K:]
```

Interpretation: behavior may be concentrated in each adapter's own top singular directions, even if not shared across adapters.

Steering use: yes as a distilled trained adapter.

Fork-plan fit: cross-adapter causal `dW` basis ablation and adapter-parameterization ablation. If per-adapter top keeps behavior better than shared SVD, this supports basin divergence.

### S-space quartiles and energy groups

Construction:

```py
U0, S0, V0h = svd(W_base)
dS = U0.T @ dW @ V0h.T
component = crop(dS, rows_or_cols_or_energy_group)
dW_component = U0 @ component @ V0h
residual = dW - dW_component
```

Interpretation: the trained update may be simple in the pretrained weight's singular-vector coordinate system even when it is not simple in raw weight space.

Steering use: yes if a crop keeps behavior and the residual loses it.

Fork-plan fit: adapter-parameterization causal ablation. Required rows already include `top_25pct_S`, `mid_50pct_S`, `bottom_25pct_S`, `top_50pct_energy_S`, `top_90pct_energy_S`, and residuals.

### Residual-write projection and complement into activation basis

Construction:

```py
B_act_l = act_oracle_block_basis(l, K)
dW_project = P_B_act_l @ dW_residual_write_l
dW_complement = dW_residual_write_l - dW_project
dW_project_normmatched = dW_project * (||dW_resid|| / ||dW_project||)
```

Interpretation: distinguishes whether low geometric overlap hides a load-bearing small component. v10 result: for DeLoRA, raw projection keeps little behavior and complement keeps most. This means block-local activation PCA is not the trained-scale carrier for DeLoRA residual-write behavior; it does not mean activation-PCA directions are useless for steering.

Steering use: projection can be a potent amplified steerer for PiSSA/OFT, but is not the trained-scale explanation for DeLoRA.

Fork-plan fit: already done as v10 projection falsifier. Future use as a sub-row under layer/module or cross-adapter if testing other bases.

### Layer and module localization

Construction:

```py
dW_variant = {k: v for k, v in dW.items() if layer(k) in layer_set and module(k) in module_set}
```

Interpretation: behavior is localized to modules or layers rather than to a geometric basis.

Steering use: yes if a small slice retains behavior.

Fork-plan fit: exact layer/module causal ablation. Required variants include `residual_write_only`, `attn_o_proj_only`, `mlp_down_proj_only`, `layers_8_21_only`, single-layer keep, leave-one-layer-out, early/mid/late, random controls, and zero.

## Steering and causal-test verdict table

| hypothesis | from-scratch steering candidate? | trained-`dW` explanation candidate? | best causal test | current prior |
|---|---:|---:|---|---|
| Function-vector head basis | yes | possible for attention `o_proj` | head-output patch, `fv_heads_only/drop_fv_heads` | Strong prior that FV heads exist in ICL; sycophancy/honesty untested here. |
| Concept vs function route split | yes | possible | separate concept-head and FV-head interventions | Useful refinement of vague concept-space story. |
| ReFT-r1 baseline | yes | no | fair activation-steering baseline on identical DD rows | Stronger baseline than unsupervised PCA if labels allowed. |
| SAE output-score signed basis | maybe | unknown | signed decoder feature keep/drop with output-causal filtering | Only worth testing with output-score filter; raw SAE is weak prior. |
| MSRS shared/private basis | yes | possible | shared/private activation and `dW` split | Hypothesis generator; require frontier improvement to justify complexity. |
| Softmax information geometry | yes | possible for readout-facing layers | Fisher/softmax projection vs Euclidean projection | Projection metric variant for degradation control. |
| TaskDiff contrast | yes | weak | activation-steering baseline, then compare to `dW` on same DD rows | Useful baseline, persona may be wrong concept. |
| Suppressed | maybe | weak | project trained `dW` into suppressed basis and evaluate keep/drop | Interesting hidden-state prior, not yet a trained-scale explanation. |
| Stenographic | maybe | weak | activation steering or `P_steno dW` keep/drop | High-risk, rank-collapse likely. |
| Churn | maybe | weak | activation steering control or synthetic `dW'` | Broad dynamic prior, likely nonspecific. |
| Attention min/max/diff TaskDiff | yes | unknown | token-conditional activation steering, QK/OV causal routing | Good next test if last-token basis is too narrow. |
| Attention-output active subspace | yes | possible | `P_attn_active dW_o` vs complement | Good geometry control; steering causality untested here. |
| Gate-active written | yes | unknown | MLP gate/up/down ablation plus nonlinear gate-conditioned intervention | Important if MLP feature-space story wins. |
| CHaRS clusters | maybe | not as linear span | per-cluster translation causal test | Linear v9 score penalizes it; do not over-read negative result. |
| Rotation contrast | yes, as rotation | unknown | rotation intervention, OFT/AntiPaSTO-style ablation | Better fit to parameterization than linear keep/drop. |
| `lm_head_read` | yes control | unlikely | activation steering and `P_lm dW` keep/drop | v10 says LoRA layers are not directly Yes/No readable. |
| `logits_null` or weak readout | maybe | unlikely | weak-readout steering and coherence/degradation check | Could hide information, but direct output behavior may be weak. |
| Global read | weak | unlikely | synthetic `dW'` control | Too broad. |
| Global write | maybe | weak | synthetic `dW'` and module ablation | Plausible capacity basis, not behavior-specific. |
| Write-not-read | yes | possible | `P_wnr dW` vs complement, synthetic `dW'` | Best old A-side recipe, but v9/v10 make it only suggestive. |
| QK merged circuit | not directly | possible for q/k modules | q/k keep/drop, attention-score intervention | Fits attention-routing story, not residual-write PCA. |
| OV write | yes | possible | attention-only module ablation | Natural attention write test. |
| MLP roundtrip | yes | possible | MLP-only module ablation | If positive, story shifts to feature-space steering. |
| Gate kernel | yes | possible | gate-conditioned MLP causal test | More realistic than raw MLP roundtrip. |
| Attention sink | speculative | unknown | BOS/sink attention routing ablation | Needs separate causal test. |
| LoRA rank | yes | yes | rank component keep/drop | Baseline parameterization. |
| DoRA magnitude/direction | yes | yes | magnitude vs direction ablation | Current behavioral gain over LoRA small. |
| DeLoRA direction/strength | yes | yes | λ vs normalized direction, rank groups | Best raw steerer; high priority. |
| PiSSA SVD | yes | yes | S-space quartiles and energy crops | Clean stable baseline; high priority. |
| OFT rotation | yes | yes | rotation-derived component vs residual | Medium priority. |
| IA3 gates | weak | yes for gates | attention gate vs MLP gate | Useful lower bound. |
| Weight-space task arithmetic | yes | yes | sign, addition, analogy rows across behaviors/adapters | Strong adoption signal, but future until multiple behavior diffs exist. |
| Shared adapter SVD | no | yes | shared keep/drop across families | Central planning-subspace ablation. |
| Per-adapter top/tail SVD | no | yes | own top/tail keep/drop | Distinguishes shared core vs basin divergence. |
| S-space crops | no | yes | crop/residual reconstruction and behavior | Central adapter-parameterization ablation. |
| Act projection/complement | no | tests carrier | v10 projection/complement | Already mostly negative for DeLoRA as trained-scale explanation. |

## Recommended additions to `fork_plan.md`

The current plan is mostly right. I would add three explicit sub-rows rather than a new broad experiment:

1. Under layer/module ablation, include read-side module groups: `q_proj_only`, `k_proj_only`, `v_proj_only`, `attention_qkv_only`, `up_proj_only`, `gate_proj_only`, `mlp_up_gate_only`, and `combined_read_only`, because several hypotheses are read-side and v10 residual-write-only cannot test them.
2. Under synthetic `dW'`, add a small fixed list: `write_not_downstream_read`, `gate_kernel`, `OV_write`, and `TaskDiff_signed_write`. These are the cleanest A-side constructive candidates.
3. Under future causal tests, add `attention_routing_basis`: compare QK score intervention vs OV write intervention using the same DD row keys. This is where merged K/Q and attention min/max/diff belong.
4. Under activation baselines, add `ReFT_r1` and `function_vector_head_patch` as stronger external baselines than PCA-only TaskDiff.
5. Under cross-adapter `dW`, add `task_arithmetic_sign_and_sum` once at least two behavior diffs exist.
6. Under projection/complement tests, add a metric variant: Euclidean projection vs softmax/Fisher-metric projection.

## Interpretation discipline

Use these claim templates to avoid overclaiming:

- If `keep_B` retains behavior and `drop_B` removes it: `B` is a causal carrier of the trained adapter behavior under this intervention family.
- If both `keep_B` and `drop_B` retain behavior: the basis is non-identifying or behavior is distributed/redundant.
- If `keep_B` fails but normmatched `keep_B` steers: `B` is a potent steering target, not the trained-scale carrier.
- If synthetic `dW'` steers without trained adapter deltas: the basis is a constructive method candidate, not evidence that the trained adapter used it.
- If activation steering beats weight steering on identical DD rows: weight steering is mechanistic-interest first, method baseline second.
- If an attention-weighted basis scores well: report the selected token identities before claiming attention routing, because min/max/diff attention weights can select formatting artifacts.