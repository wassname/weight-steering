# weight-steering recipes. Default model is Qwen3-0.6B for cheap iteration.

model := "Qwen/Qwen3-0.6B"
behavior := "sycophancy"
adapter := "lora"
out := "out"

# Quick end-to-end check: tiny data, 1 epoch, qualitative gen at coeff +/-1.
smoke:
    uv run python -m scripts.replicate --model {{model}} --behavior {{behavior}} \
        --adapter {{adapter}} --n-pairs 32 --max-steps 20 --smoke

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
    uv run python -m scripts.subspace_align --model {{model}} \
        --adapter {{adapter}} --out {{out}}

# Phase 3: full sweep over LoRA, DoRA, PiSSA-init LoRA, DeLoRA.
adapter-sweep:
    uv run python -m scripts.adapter_sweep --model {{model}} --behavior {{behavior}} --out {{out}}

# Replicate: full phase-1 pipeline (data -> train pos -> train neg -> diff -> eval).
replicate:
    uv run python -m scripts.replicate --model {{model}} --behavior {{behavior}} \
        --adapter {{adapter}} --n-pairs 1000

setup:
    uv sync

clean:
    rm -rf {{out}}
