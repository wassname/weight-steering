# v6 hypothesis review

Reviewer: scout-mindset critique of [nbs/hypothesis_sweep_v6.py](nbs/hypothesis_sweep_v6.py) (934 lines).
Frame: project goal is to identify a low-rank subspace, derived without the trained
LoRA, that explains both the LoRA-induced activation difference (`R_act`) and the
LoRA weight delta (`R_w`), so we can steer the base model. A-side bases are built
from base weights and base activations only; B-side labels are LoRA-driven.

> Reviewer-of-reviewer note (wassname/copilot): the subagent's claimed [BUG] on
> `input_super_not_lm_read` was retracted after rechecking. `input_super[-1]` at
> [nbs/hypothesis_sweep_v6.py#L530](nbs/hypothesis_sweep_v6.py#L530) refers to the
> entry just appended at [nbs/hypothesis_sweep_v6.py#L521](nbs/hypothesis_sweep_v6.py#L521)
> *within the same per-layer loop iteration*, so it IS per-layer. Reclassified [OK].
> Same for `forbidden = ... input_super[-1], kv_super[-1] ...` at L528. No bugs found.

## Biggest concerns (top 5)

1. **`R_w` is dominated by `mlp.down_proj`.** `lora_weight_matrix` concatenates
   `o_proj` (d_model x d_model ~ 1.05M params) and `down_proj`
   (d_model x d_mlp ~ 1024 x 3072 ~ 3.1M params) along columns; the score uses
   `||basis.T @ M||_F^2 / ||M||_F^2`, which is Frobenius-weighted by tensor size
   ([nbs/hypothesis_sweep_v6.py#L617-L626](nbs/hypothesis_sweep_v6.py#L617-L626)).
   Whichever LoRA tensor has more parameters dominates `R_w`; `R_w ~ R_w(down_proj)`
   here. Fix: report `R_w` per residual-output tensor, then optionally a
   Frobenius-balanced sum.
2. **The "weight ceiling" is not a ceiling.** `taskdiff_basis_w` is the LoRA-fitted
   activation basis (`PCA(hs_diff_B_fit)`) scored on the weight axis
   ([nbs/hypothesis_sweep_v6.py#L729-L734](nbs/hypothesis_sweep_v6.py#L729-L734)).
   The natural weight ceiling is `dw_left_basis(layer)` which gives `R_w` ~
   d_model/PCS by construction. Reporting percentages relative to
   `taskdiff_basis_w` invites readers to misread small percentages as small
   absolute headroom. Caveat is in `v6_conclusion.md` but the column name
   `pct_w_taskdiff_basis` is misleading.
3. **`R_w` for read-side bases tests cross-space overlap, not "explains weight diff".**
   `mlp_up_read`, `mlp_gate_read`, `attn_qkv_read`, `kv_super`, `input_super`,
   `qk_circuit`, `lm_head_read`, `logits_null`, `input_super_not_lm_read` all live
   in the *read* (input) side of d_model. The LoRA delta tested by `R_w` lives in
   the *write* (output) side. Both are d_model so the math runs, but a high `R_w`
   for these means "read directions happen to coincide with write directions of the
   LoRA update", not "this primitive captures the LoRA write geometry". Should be
   flagged in the table or these rows should be excluded from the joint ranking.
4. **`principal_cos` is the *mean* of singular values, not a standard Grassmann
   metric** ([nbs/hypothesis_sweep_v6.py#L150-L152](nbs/hypothesis_sweep_v6.py#L150-L152)).
   Mean cos of principal angles penalizes mismatched ranks asymmetrically. Used
   only for the diagnostic `cos_with_dW` column so this is minor, but document it.
5. **Single LoRA seed - no subspace stability claim.** All A-side bases are
   deterministic given the FIT activations and base weights, but the entire
   conclusion conditions on one trained LoRA (`W_PATH`). A useful primitive must be
   stable across LoRA seeds; without that, "winner" rankings are anecdotes.

## What to fix first

1. Split `R_w` per tensor: report `R_w(o_proj)` and `R_w(down_proj)` separately.
   Add a column tagging whether each basis is a "write-side" hypothesis (where
   `R_w` is the natural fit) or "read-side" (where `R_w` measures cross-alignment).
2. Replace the misleading `pct_w_taskdiff_basis` with `R_w / R_w(dw_left_basis)`,
   which is a true ceiling-relative number.
3. Train >=3 LoRA seeds; report median + IQR of `R_act`, `R_w`, and per-basis
   subspace overlap across seeds before any winner claim. Drop hypotheses whose
   top-PCS basis flips across seeds.
4. Add a baseline `randn_orthogonal` candidate at PCS=8 so the null line on the
   scatter plot is visible alongside actual hypotheses.
5. Add a unit test for `lora_weight_matrix` shape and key selection - currently
   silently filters by `W.shape[0] == d_model` ([nbs/hypothesis_sweep_v6.py#L617-L626](nbs/hypothesis_sweep_v6.py#L617-L626)).

---

## Scoring infrastructure (cross-cutting)

### `concentration_act` ([nbs/hypothesis_sweep_v6.py#L687-L696](nbs/hypothesis_sweep_v6.py#L687-L696))
Uses `hs_diff_B` (EVAL split) for both score and random null; A-side bases use
FIT - train/test split clean. Energy ratio normalized by `rank/d_model` matches
the random-orthogonal null in expectation. **[OK]**

### `concentration_w` ([nbs/hypothesis_sweep_v6.py#L698-L707](nbs/hypothesis_sweep_v6.py#L698-L707))
Frobenius-weighted blend of o_proj+down_proj LoRA deltas; down_proj dominates
because d_mlp >> d_model. Null is over random rank-r ortho bases of d_model.
The math is correct; the *aggregation* implicitly weights by tensor size, which
is not flagged in `v6_definitions.md` or `v6_conclusion.md`. **[METHOD]**

### `lora_weight_matrix` ([nbs/hypothesis_sweep_v6.py#L617-L626](nbs/hypothesis_sweep_v6.py#L617-L626))
Filters keys by `W.shape[0] == d_model`. Drops `q/k/v/up/gate` LoRA tensors
silently because they output to head_dim or d_mlp space. Intentional (basis
lives in d_model) and noted in the conclusion, but a bare `if W.shape[0] ==
d_model:` is a silent filter - should at least log dropped keys. **[METHOD]**

### `principal_cos` ([nbs/hypothesis_sweep_v6.py#L150-L152](nbs/hypothesis_sweep_v6.py#L150-L152))
Mean of singular values of `A.T @ B`. Only valid as a principal-angle cosine
when both inputs are orthonormal (they are, by `add()` check). Mean (not min,
not RMS) is unconventional but consistent. Used only as `cos_with_dW`
diagnostic. **[OK]**

### `act_null_stats` and `w_null_stats` ([nbs/hypothesis_sweep_v6.py#L633-L671](nbs/hypothesis_sweep_v6.py#L633-L671))
Per-(layer, rank) caches; rank-matched random ortho bases via QR of Gaussian.
Reproducible via `manual_seed`. `N_NULL=120` gives ~9% relative SE on null
mean - acceptable for z-scores in the 1-10 range, marginal beyond that. **[OK]**

### Train/test split ([nbs/hypothesis_sweep_v6.py#L57-L60](nbs/hypothesis_sweep_v6.py#L57-L60), [nbs/hypothesis_sweep_v6.py#L307-L311](nbs/hypothesis_sweep_v6.py#L307-L311))
`FIT = first half`, `EVAL = second half` of `SYCOPHANCY_TOPICS`. A-side captures
use FIT, scoring uses EVAL `hs_diff_B`. Ceiling `TaskDiff_lora_ceiling` uses
FIT for fitting, EVAL for scoring - proper held-out evaluation. **[OK]**

### Symmetric LoRA effect ([nbs/hypothesis_sweep_v6.py#L313-L315](nbs/hypothesis_sweep_v6.py#L313-L315))
`hs_diff_B = capture(alpha=+1) - capture(alpha=-1)`, central-difference
~ 2 * dh/da. Captures linearized behavior at alpha=0 with cancellation of
alpha-symmetric drift. **[OK]**

---

## W-axis (LoRA-free weight bases)

### `lm_head_read` ([nbs/hypothesis_sweep_v6.py#L411-L412](nbs/hypothesis_sweep_v6.py#L411-L412), [nbs/hypothesis_sweep_v6.py#L483](nbs/hypothesis_sweep_v6.py#L483))
Top right singular vectors of unembedding matrix. High `R_act` would mean LoRA
writes into the principal logit-readable subspace - meaningful claim. High
`R_w` is the read-vs-write cross-space concern (#3 above). **[METHOD]**

### `logits_null` ([nbs/hypothesis_sweep_v6.py#L412](nbs/hypothesis_sweep_v6.py#L412), [nbs/hypothesis_sweep_v6.py#L484](nbs/hypothesis_sweep_v6.py#L484))
Bottom right singular vectors of unembedding. Useful as a control: should score
near random-null on both axes if LoRA acts via the logit interface. **[OK]**

### `global_read` ([nbs/hypothesis_sweep_v6.py#L420-L421](nbs/hypothesis_sweep_v6.py#L420-L421), [nbs/hypothesis_sweep_v6.py#L485](nbs/hypothesis_sweep_v6.py#L485))
Top eigenspace of summed Gram matrix `Sum W^T W + lm_head^T lm_head` over all
read-side projections. Same per-layer basis is reused at every layer
([nbs/hypothesis_sweep_v6.py#L485](nbs/hypothesis_sweep_v6.py#L485)), hiding any
layer specificity. Same caveat as `lm_head_read` for `R_w` (read-side basis).
**[METHOD]**

### `global_write` ([nbs/hypothesis_sweep_v6.py#L425](nbs/hypothesis_sweep_v6.py#L425), [nbs/hypothesis_sweep_v6.py#L486](nbs/hypothesis_sweep_v6.py#L486))
Top left singular vectors of `[W_o | W_down]` concatenated across all layers.
Natural weight-side baseline. **[OK]**

### `global_write_not_global_read` ([nbs/hypothesis_sweep_v6.py#L487](nbs/hypothesis_sweep_v6.py#L487))
Global write subspace projected away from global read directions. Concept:
"causally isolated" write directions. **[OK]**

### `write` / `attn_write` / `mlp_write` ([nbs/hypothesis_sweep_v6.py#L489-L491](nbs/hypothesis_sweep_v6.py#L489-L491), [nbs/hypothesis_sweep_v6.py#L498-L500](nbs/hypothesis_sweep_v6.py#L498-L500))
Per-layer left singular vectors of write tensors. The native weight-side
hypothesis. v5 already showed `mlp_write` and `write` are the strongest
write-family candidates. **[OK]**

### `write_not_lm_head_read` / `write_not_global_read` / `write_not_downstream_read` ([nbs/hypothesis_sweep_v6.py#L492-L497](nbs/hypothesis_sweep_v6.py#L492-L497), [nbs/hypothesis_sweep_v6.py#L501-L503](nbs/hypothesis_sweep_v6.py#L501-L503))
Same write basis after subtracting various read subspaces. The "downstream"
version reverse-cumulates read grams from final layer back, which is the most
principled. **[OK]**

### `mlp_up_read` / `mlp_gate_read` / `attn_qkv_read` ([nbs/hypothesis_sweep_v6.py#L516-L518](nbs/hypothesis_sweep_v6.py#L516-L518), [nbs/hypothesis_sweep_v6.py#L533-L535](nbs/hypothesis_sweep_v6.py#L533-L535))
Right singular vectors of input projections. **Read-side bases - `R_w` is
cross-space.** For `attn_qkv_read`, GQA means k,v have fewer rows than q;
concatenation along dim=0 is fine but the right SVD is dominated by q.
**[METHOD]**

### `attn_ov_write` ([nbs/hypothesis_sweep_v6.py#L519](nbs/hypothesis_sweep_v6.py#L519), [nbs/hypothesis_sweep_v6.py#L536](nbs/hypothesis_sweep_v6.py#L536))
Left singular vectors of `W_o @ W_v` after row-expanding `W_v` (`v_for_o =
expand_rows_to(v, W_o.shape[1])`, [nbs/hypothesis_sweep_v6.py#L499](nbs/hypothesis_sweep_v6.py#L499)).
Standard OV-circuit basis. **[OK]**

### `mlp_roundtrip_write` ([nbs/hypothesis_sweep_v6.py#L520](nbs/hypothesis_sweep_v6.py#L520), [nbs/hypothesis_sweep_v6.py#L537](nbs/hypothesis_sweep_v6.py#L537))
Left singular vectors of `W_down @ W_up`, the *linear* MLP residual map
(ignoring SiLU gate). Reasonable but the actual MLP is gated. **[OK]**

### `qk_circuit` ([nbs/hypothesis_sweep_v6.py#L521](nbs/hypothesis_sweep_v6.py#L521), [nbs/hypothesis_sweep_v6.py#L538](nbs/hypothesis_sweep_v6.py#L538))
`left_svd_basis(q.T @ k_for_q)`. Mixes all heads into one d_model x d_model
matrix. Conceptually muddied because per-head QK circuits are usually distinct.
As a single-layer aggregate it may wash out head-specific structure sycophancy
likely uses. **[METHOD]**

### `input_super` / `kv_super` ([nbs/hypothesis_sweep_v6.py#L522-L523](nbs/hypothesis_sweep_v6.py#L522-L523), [nbs/hypothesis_sweep_v6.py#L539-L540](nbs/hypothesis_sweep_v6.py#L539-L540))
Right singular vectors of stacked input projections. Read-side, same `R_w`
caveat. **[METHOD]**

### `gate_kernel` ([nbs/hypothesis_sweep_v6.py#L501](nbs/hypothesis_sweep_v6.py#L501) for mean_gate, [nbs/hypothesis_sweep_v6.py#L524](nbs/hypothesis_sweep_v6.py#L524), [nbs/hypothesis_sweep_v6.py#L541](nbs/hypothesis_sweep_v6.py#L541))
`W_down @ diag(mean_silu_gate) @ W_up`. Mean gate averaged over FIT prompts -
data-dependent linearization at the empirical operating point. Couples the
basis to FIT data but uses base activations only, so A-side. **[OK]**

### `attention_sink` ([nbs/hypothesis_sweep_v6.py#L505-L514](nbs/hypothesis_sweep_v6.py#L505-L514), [nbs/hypothesis_sweep_v6.py#L525](nbs/hypothesis_sweep_v6.py#L525), [nbs/hypothesis_sweep_v6.py#L542](nbs/hypothesis_sweep_v6.py#L542))
PCA over per-head `W_o^h @ (W_v^h @ e_BOS)` sink vectors. GQA mapping is the
standard floor-grouping for Qwen3. **[OK]**

### `causally_isolated` ([nbs/hypothesis_sweep_v6.py#L527-L528](nbs/hypothesis_sweep_v6.py#L527-L528), [nbs/hypothesis_sweep_v6.py#L543](nbs/hypothesis_sweep_v6.py#L543))
`project_write_away(write_cols(layer), forbidden)` with forbidden = union of
input_super, kv_super, lm_head_read. Orientation correct: we want write
subspace minus read directions. The plan-merge note flags an earlier bug
returning a complement of the forbidden basis unrelated to write; current
version is correct. **[OK]**

### `input_super_not_lm_read` ([nbs/hypothesis_sweep_v6.py#L530](nbs/hypothesis_sweep_v6.py#L530), [nbs/hypothesis_sweep_v6.py#L544](nbs/hypothesis_sweep_v6.py#L544))
`project_away(input_super[-1], lm_read_broad)[:, :PCS]`. `input_super[-1]` is
the just-appended per-layer entry from the same loop iteration
([nbs/hypothesis_sweep_v6.py#L521](nbs/hypothesis_sweep_v6.py#L521)), so it IS
per-layer. **[OK]**

---

## A-axis (activation-based bases)

### `suppressed` / `amplified` / `added_features` ([nbs/hypothesis_sweep_v6.py#L477-L484](nbs/hypothesis_sweep_v6.py#L477-L484) helper, [nbs/hypothesis_sweep_v6.py#L545-L549](nbs/hypothesis_sweep_v6.py#L545-L549) registration)
PCA over functions of magnitude trajectories across layers. `suppressed` uses
`min(sum_relu(+d), sum_relu(-d))` - a quirky symmetric churn measure.
`amplified = relu(|h_last| - |h_first|)`, `added = sum_relu(|h_{l+1}|-|h_l|)`.
Conceptually obscure; would benefit from a literature pointer. **[METHOD]**

### `global_clean_resid_pca` / `global_persona_resid_pca` / `layer_clean_resid_pca` ([nbs/hypothesis_sweep_v6.py#L550-L552](nbs/hypothesis_sweep_v6.py#L550-L552))
Generic activation PCA - included as expected high-`R_act`, low-`R_w` baselines
that residualized `specific_concentration_act` controls for. v5 already showed
`layer_clean_resid_pca` was the raw winner. **[OK]**

### `TaskDiff_contrast` ([nbs/hypothesis_sweep_v6.py#L553](nbs/hypothesis_sweep_v6.py#L553))
PCA of persona+/persona- system-prompt activation difference. Natural A-side
proxy for the LoRA's behavioral axis. **[OK]**

### `attn_min/max/diff_taskdiff` and `attn_min_x_diffnorm_taskdiff` ([nbs/hypothesis_sweep_v6.py#L284-L302](nbs/hypothesis_sweep_v6.py#L284-L302), [nbs/hypothesis_sweep_v6.py#L554-L557](nbs/hypothesis_sweep_v6.py#L554-L557))
Token-level persona TaskDiff weighted by final-token attention statistics.
PCA after weighting samples by `sqrt(weight)` ([nbs/hypothesis_sweep_v6.py#L300](nbs/hypothesis_sweep_v6.py#L300))
- correct sqrt-trick. The aligned padding right-aligns hidden states
([nbs/hypothesis_sweep_v6.py#L264-L281](nbs/hypothesis_sweep_v6.py#L264-L281)),
which assumes both pos and neg have the same suffix - true since only the
system prompt differs. **[OK]**

### `up_proj_input_contrast` / `up_proj_output_written_contrast` ([nbs/hypothesis_sweep_v6.py#L558-L559](nbs/hypothesis_sweep_v6.py#L558-L559))
Persona contrast in `mlp.up_proj` input space, and after passing through
`W_up` then `W_down.T`. `act @ W_down.T` IS the residual write but skips the
SiLU-gate multiplication, so this measures "what `W_down` would write if there
were no gate". Acceptable proxy but should be named accordingly. **[METHOD]**

### `gate_active_written` ([nbs/hypothesis_sweep_v6.py#L502](nbs/hypothesis_sweep_v6.py#L502), [nbs/hypothesis_sweep_v6.py#L545](nbs/hypothesis_sweep_v6.py#L545), [nbs/hypothesis_sweep_v6.py#L560](nbs/hypothesis_sweep_v6.py#L560))
Same as up_written but with the SiLU gate applied:
`silu(W_gate h) * W_up h` then `@ W_down.T`. This IS the actual MLP residual
contribution per token. **[OK]**

### `chars_clusters` ([nbs/hypothesis_sweep_v6.py#L532](nbs/hypothesis_sweep_v6.py#L532), [nbs/hypothesis_sweep_v6.py#L561](nbs/hypothesis_sweep_v6.py#L561), [nbs/hypothesis_sweep_v6.py#L375-L389](nbs/hypothesis_sweep_v6.py#L375-L389))
K-means with deterministic largest-norm init, 8 iterations, then PCA of
centroid differences. With ~3 x ~15 = ~45 samples and 8 clusters, clusters
will be small and seed-fragile despite "deterministic" init. The PCA is over
`centroids - centroids.mean(0)` which is rank <= k_clusters - 1 = 7 < PCS=8 -
the basis can collapse to 7 dims silently. **[METHOD]**

### `churn` ([nbs/hypothesis_sweep_v6.py#L562](nbs/hypothesis_sweep_v6.py#L562))
PCA of `h_{l+1} - h_l`. Canonical "what changed at each layer" baseline. **[OK]**

### `rotation_contrast` ([nbs/hypothesis_sweep_v6.py#L470-L484](nbs/hypothesis_sweep_v6.py#L470-L484), [nbs/hypothesis_sweep_v6.py#L563](nbs/hypothesis_sweep_v6.py#L563))
Procrustes rotation between persona-/persona+ then SVD of the skew part. For
real antisymmetric matrices, singular values come in pairs (real eigenvalues
+/-i lambda), so taking the top k may split a pair and lose orthogonality of
the rotation axes. Re-orthonormalization after projection makes this harmless,
but axis ranking is unstable. **[METHOD]**

---

## Compound

### `qk_x_chars_clusters` ([nbs/hypothesis_sweep_v6.py#L564](nbs/hypothesis_sweep_v6.py#L564))
`intersect_basis(qk_circuit[layer], chars_clusters[layer])`. The function
([nbs/hypothesis_sweep_v6.py#L130-L134](nbs/hypothesis_sweep_v6.py#L130-L134))
returns `orthonormalize(A @ U[:, :k] + B @ Vh.T[:, :k])` - the **bisector**
("Bjorck-Golub principal vectors averaged"), not the strict intersection.
Strict intersection requires sigma ~ 1; bisector returns directions even when
sigma << 1. So a high `R_act` here can come from one of qk_circuit /
chars_clusters carrying signal alone. Should also report the principal angles
sigma. **[METHOD]**

### `WNR_union_TaskDiff` ([nbs/hypothesis_sweep_v6.py#L565](nbs/hypothesis_sweep_v6.py#L565))
`orthonormal_union(write_not_downstream_read, TaskDiff_contrast)` - rank up to
2*PCS=16 after orthonormalization. The null normalizes by rank, so head-to-head
with rank-8 baselines is fair in expectation. But `mean_conc_act` is biased
toward higher-rank bases when `hs_diff_B` energy is concentrated. Reporting
the joint score next to ranks 8 and 16 mixes hypothesis types. **[METHOD]**

---

## Ceiling

### `TaskDiff_lora_ceiling` ([nbs/hypothesis_sweep_v6.py#L566-L572](nbs/hypothesis_sweep_v6.py#L566-L572))
PCA of `hs_diff_B_fit` (FIT-half LoRA-induced activation diff), evaluated on
EVAL `hs_diff_B`. Train/test split honored. This is the activation ceiling.
**Critically, this is NOT a weight ceiling**, despite being used as one in the
`pct_w_taskdiff_basis` column ([nbs/hypothesis_sweep_v6.py#L734](nbs/hypothesis_sweep_v6.py#L734)).
The weight axis has no oracle ceiling computed; `dw_left_basis` would give
`R_w` ~ d_model/PCS = 128 trivially. **[CLAIM]** for the ceiling label on the
weight axis; **[OK]** for the activation ceiling.

---

## Verdict tally

- [OK]: ~17 hypotheses + main scoring infra
- [METHOD]: ~13 (most cluster around: `R_w` for read-side bases, `R_w` weighting,
  rank-confound for compound bases, k-means stability)
- [CLAIM]: 1 (weight ceiling mislabeling)
- [BUG]: 0 (subagent's flagged bug retracted on re-review)
- [MATH]: 0

Core sweep methodology is sound. Actionable items: split `R_w` per tensor, use
`dw_left_basis` as the true weight ceiling (or rename column to remove
"ceiling" framing), and run >=3 LoRA seeds before naming any winner.
