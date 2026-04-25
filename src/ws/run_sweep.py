"""Phase 3 entrypoint: run replicate.py for each adapter in {lora, dora, pissa, delora}.

Final output: a polars table with columns
    (adapter, behavior, alpha, eval_logratio, subspace_alignment_ratio, train_time_s)
"""
