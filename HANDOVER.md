# Handover notes — 2026-04-27 ~21:30

## What just happened

Switched the entire weight-steering pipeline from sycophancy to honesty axis. Rationale in `RESEARCH_JOURNAL.md` (2026-04-27 entry) and `fork_plan.md` ("Resolved: train/eval axis switch").

Key change: old SYCOPHANCY_POS/NEG was 2-axis (sycophancy-vs-honesty mixed). New HONESTY_POS/NEG is 1-axis, axis-matched with eval (`daily_dilemmas-self-honesty` / `honesty_label`). Sycophancy becomes OOD transfer eval for later.

## Pueue queue state

- **230** — Running: `ws.run_sweep --behavior honesty`. Training 7 adapters (lora/dora/pissa/delora/oft/boft/ia3) on 1000 pairs. lora + dora DONE (~21:22, 21:26). pissa/delora/oft/boft/ia3 pending. ~5 adapters remaining at ~10min each = ~50min left.
- **231** — Queued after 230: T1 RepE activation baseline honesty
- **232** — Queued after 230: T3 prompt baseline honesty
- **233** — Queued after 230: T2 full DD benchmark honesty
- **234** — Queued after 230: T6 cross-adapter causal ablation honesty
- **235** — Queued after 230: T7 layer/module ablation honesty
- **236** — Queued after 230: T8 parameterization ablation honesty

230-236 are all queued and will run unattended. Check tomorrow with `pueue status`.

## Key files changed this session

- `src/ws/data.py` — honesty personas, `_load_suffixes`, behavior branches in `_topics`/`_build_specs`
- `src/ws/eval/activation_baseline.py` — honesty branch in `_fit_repe_directions` with suffix-based prompts
- `src/ws/eval/prompt_baseline.py` — dual `engineered_prompt_honest` + `engineered_prompt_dishonest`
- `evals/smoke.py` — `behavior` field added to SmokeCfg
- `data/branching_suffixes.json` — new file, 550 SSteer entries
- `fork_plan.md` — open-question section replaced with resolved decision
- `RESEARCH_JOURNAL.md` — 2026-04-27 axis-switch entry appended

## What still needs doing (after 231-236 finish)

1. **Task 28: Update README** — replace "first 100 dilemmas" and sycophancy table with honesty numbers from `out/honesty/{cross_adapter_full_dd,activation_baseline,prompt_baseline}/summary.csv`. Wait for all evals.
2. **Commit** — nothing committed yet. Files on `dev` branch, uncommitted. Commit message: "switch training/eval axis from sycophancy to honesty; add branching_suffixes.json".
3. **Task 23 close** — mark in_progress task 23 completed once 230 finishes cleanly (check `out/honesty/*/w.pt` all exist).
4. **T4 multiseed / T5 Gemma** — not started, re-scope to honesty axis when ready.

## Stale outputs to ignore

`out/honesty/{activation_baseline,cross_adapter_*,layer_module_ablation,parameterization_ablation,prompt_baseline}/` dirs exist from an aborted earlier run (timestamps 20:13-20:14, all empty). Tasks 231-236 will overwrite them.

`out/sycophancy/` — keep as historical record of old axis-mismatched results.

## Verification checklist (run tomorrow after 236 finishes)

```sh
# All adapters trained
ls out/honesty/{lora,dora,pissa,delora,oft,boft,ia3}/w.pt

# Eval summaries exist and have data
head -5 out/honesty/cross_adapter_full_dd/dilemmas_summary.csv
head -5 out/honesty/activation_baseline/summary.csv
head -5 out/honesty/prompt_baseline/summary.csv

# Sanity: idx_symmetric_diff=0 in prompt baseline
grep "idx_symmetric_diff" out/honesty/prompt_baseline/summary.csv
```
