# weight-steering recipes. Default model is Qwen3-0.6B for cheap iteration.

set shell := ["bash", "-cu"]

model := "Qwen/Qwen3-0.6B"
behavior := "sycophancy"
adapter := "lora"
out := "out"

SMOKE_MODEL := "katuni4ka/tiny-random-qwen3"
SMOKE_LOG := "out/smoke/smoke.log"

# Smoke: BEARTYPE=1 + tiny random model. Exercises full pipeline end-to-end.
# Catches dim/dtype errors via jaxtyping runtime checks. ~1 min on CPU.
smoke *ARGS:
    mkdir -p out/smoke && \
    BEARTYPE=1 uv run python evals/smoke.py \
        --model {{SMOKE_MODEL}} \
        {{ARGS}} \
        2>&1 | tee {{SMOKE_LOG}} | tail -200

# Generate +/- pair data for a behavior. Writes to out/data/{behavior}/.
data:
    uv run python -m ws.data --model {{model}} --behavior {{behavior}} --n-pairs 1000

# Train a single adapter (positive or negative). Pos/neg controls system prompt at gen time.
train sign="pos":
    uv run python -m ws.train --model {{model}} --behavior {{behavior}} \
        --adapter {{adapter}} --sign {{sign}} --out {{out}}

# Compute weight-space diff w = θ+ - θ- and persist.
diff:
    uv run python -m ws.diff --model {{model}} --behavior {{behavior}} \
        --adapter {{adapter}} --out {{out}}

# Phase 1 eval: sycophancy logratio sweep over alpha.
eval-syco:
    uv run python -m ws.eval.sycophancy --model {{model}} \
        --adapter {{adapter}} --out {{out}}

# Phase 4 eval: daily dilemmas Yes/No logratio.
eval-dilemmas:
    uv run python -m ws.eval.dilemmas --model {{model}} \
        --adapter {{adapter}} --out {{out}}

# Phase 2: project w onto SVD + AntiPaSTO subspaces, print alignment table.
subspace-align:
    uv run python -m ws.run_subspace --model {{model}} \
        --adapter {{adapter}} --out {{out}}

# Phase 3: full sweep over LoRA, DoRA, PiSSA-init LoRA, DeLoRA.
adapter-sweep:
    uv run python -m ws.run_sweep --model {{model}} --behavior {{behavior}} --out {{out}}

# Replicate: full phase-1 pipeline (data -> train pos -> train neg -> diff -> eval).
replicate:
    uv run python -m ws.replicate --model {{model}} --behavior {{behavior}} \
        --adapter {{adapter}} --n-pairs 1000

setup:
    uv sync

clean:
    rm -rf {{out}}
