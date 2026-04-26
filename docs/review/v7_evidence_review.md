# v7 evidence cold-review

Reviewer: cold-eyes subagent (read evidence files only, no source).
File generated from subagent output; numeric verification reproduced from
[v7_summary_pct.tsv](../../out/sycophancy/lora/v7/v7_summary_pct.tsv).

## Verdicts

| Finding | Tag | Detail |
|---|---|---|
| A. Read vs write distinguishable in TSV | [OK]+[CONCERN] | `axis_kind` column tags every row. Conclusion claim "separate sub-table" is true only in the figure, not the TSV (single flat table). Mild over-claim. |
| B. dW_left_basis_ceiling = 100% on all three | [OK] | Row 1 of v7_summary_pct.tsv: pct_w_oracle_combined=100.0, oproj=100.0, downproj=100.0. No per-tensor normalization bug. |
| C. Top-5 intersection = {chars_clusters, attn_min_taskdiff} | [OK] | Reproduces from table. |
| D1. Single-seed overfitting | [CONCERN] | attn_min_taskdiff R_w lead is 1.30 vs cluster at 1.12-1.15, no error bars. Cannot rule out seed luck. |
| D2. Write-family R_w < null (z = -1 to -2) | [CONCERN, important] | `write` 0.954, `mlp_write` 0.950, `global_write` 0.946, `attn_write` 0.917, `mlp_roundtrip_write` 0.935 -- all BELOW random-rotation null. Two readings: (a) null mis-specified (top-SVD-of-weights bases live in a low-rank atypical region of the d_model sphere relative to LoRA's task-specific delta); (b) substantive: LoRA picks task-specific write directions, NOT generic high-singular-value ones. Either way, this is a real signal, not noise. v7 conclusion ignores it. |
| D3. dW_left ceiling R_w=16.39 not 128 | [OK] | Consistent with LoRA rank 32 (not d_model). Frobenius-balanced concentration of a rank-32 delta projected onto its own top-PCS=8 basis is bounded by ~rank/PCS, not d_model/PCS. |
| D4. chars_clusters rank-collapse inflates R_act ~14% | [CONCERN] | Acknowledged inline but TSV shows mean_rank=8 (one direction is noise). After ~14% deflation chars_clusters R_act ~ 10.4, still top-5 but fragile. attn_min_taskdiff R_w lead would not survive this artifact. |
| D5. attn_min_taskdiff topic-set leakage | [CONCERN, low] | LoRA training set and probe set share SYCOPHANCY_TOPICS. Basis-construction signal (base-model attention on persona-prompted probes) differs from training signal (chat completions), but worth checking whether splitting topics across train/probe changes the lead. |
| E. Conclusion calibration | [CLAIM] | Overclaims "best primitive = chars_clusters" (knife-edge: R_w_combined 1.155 vs TaskDiff_contrast 1.152). Underclaims by not surfacing D2. |

## BLUF

v7 evidence checks out mechanically: dW oracle row hits 100% on all three
pct_w columns, top-5 intersection {chars_clusters, attn_min_taskdiff}
reproduces, and the R_w_combined=16.39 ceiling is consistent with the LoRA
delta being rank-32 (not a normalization bug). Headline "best primitive =
chars_clusters" is on a knife-edge: R_w lead (1.155 vs 1.152 for
TaskDiff_contrast) is statistically tied and R_act is inflated by the
acknowledged k-1 rank collapse to 7. The most interesting unflagged signal
is that write-family bases (write, mlp_write, global_write, attn_write) all
sit at R_w_combined ~ 0.92-0.96 with z ~ -1 to -2 -- below random-rotation
null on the very axis they were designed for. That's either a null-spec
issue or a substantive finding that LoRA's delta lives in task-specific
write directions, not top-SVD write directions. Conclusion should say
which.

## Recommended next steps

1. Add a sentence in v7_conclusion.md flagging D2 as a real result needing
   interpretation.
2. Re-baseline against "random rank-8 basis sampled from the rank-32 LoRA
   delta column space" (i.e. null inside the LoRA-relevant slice). If
   write-family climbs back above 1 and chars_clusters/attn_min_taskdiff
   lose their lead, D2(a) wins. If write-family stays below, D2(b) wins.
3. v7b: multi-LoRA-seed, with stability filter on top-PCS principal angles.
4. Topic-split holdout for attn_min_taskdiff to rule out D5.
