"""Sycophancy eval: held-out QA prompts, sweep alpha, log P(sycophantic) - log P(honest).

For replication of paper §4 / Appendix E. Single behavior, single adapter, sweep
alpha ∈ [-2, -1, 0, 1, 2]. Output: polars table (alpha, mean_logratio, n).
"""
