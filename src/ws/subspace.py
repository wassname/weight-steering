"""SVD + AntiPaSTO subspace alignment for the diff vector w.

Per layer:
  W = U_out @ diag(S) @ U_in.T    # SVD of pretrained weight
  w_layer  = θ+_layer - θ-_layer  # the steering diff

Subspaces tested (see docs/AntiPaSTO_concepts/):
  - SVD top-k    : column span of U_out[:, :k] (or U_in[:, :k])
  - Suppressed   : PCA of layer-to-layer magnitude drops on a probe set
  - Write-not-read : col_span(W_o, W_down) ∩ orth(row_span(W_q,W_k,W_v,W_up_{L+1}))
  - Weak-readout : bottom-1% of unembedding SVD
  - Stenographic : task_diff ∩ suppressed

Metric per layer:
  energy_ratio = ||proj_subspace(w_layer)||² / ||w_layer||²
  null         = same projection applied to a random rank-r matrix scaled to ||w_layer||
  alignment    = energy_ratio / null  with bootstrap CI

A ratio > 1 (CI excluding 1) is real alignment.
"""
