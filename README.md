# Weight Steering

> **Fork notice (wassname, 2026-04):** this is a working fork that strips the
> upstream Axolotl + vLLM + Anthropic-batch-API stack and rebuilds the core
> method on HF + PEFT + uv, targeting Qwen3-0.6B for cheap iteration. Goals:
> (1) replicate `w = θ⁺ − θ⁻` on a small model, (2) test alignment of `w` with
> SVD subspaces of the pretrained `W` and the AntiPaSTO subspaces, (3) compare
> adapter families (LoRA / DoRA / PiSSA-init / DeLoRA) under the
> "adapter as hypothesis" framing, (4) eval on daily-dilemmas.
>
> Pipeline (see `justfile`):
> ```
> just smoke           # full pipeline on tiny-random qwen3 + BEARTYPE=1, ~1 min
> just replicate       # data → train pos → train neg → diff → eval → subspace
> just subspace-align  # phase 2: SVD top-k + weak-readout alignment table
> just adapter-sweep   # phase 3: LoRA / DoRA / PiSSA / DeLoRA sweep (TODO)
> just eval-dilemmas   # phase 4: daily-dilemmas Yes/No logratio (TODO)
> ```
> Source layout: `src/ws/{data,train,diff,steer,subspace,replicate,run_subspace,run_sweep}.py`,
> `src/ws/eval/{sycophancy,dilemmas}.py`. Outputs to `out/<behavior>/<adapter>/`.
>
> **Scope.** Not a strict replication. Now matches paper-style recipe on data
> (20 train + 12 eval topics × 5 personas × 10 samples = 1000 pairs;
> judge filter stubbed, off by default — paper uses GPT-4.1-mini) and
> current PEFT hyperparams (rank 32 / LoRA α 64 / lr 2e-4 / warmup 5 /
> wd 0.01 / seed 0 / one epoch).
> Deliberate divergences from upstream: no quantized base loading
> (DoRA/PiSSA/DeLoRA support is uncertain; bf16 fits at 0.6B), no
> `modules_to_save` for `embed_tokens` / `lm_head`, and a layer slice
> (LoRA on layers 30%-80%, steering-locus literature) instead of full
> coverage. The contrastive `θ⁺ − θ⁻` core is preserved.
>
> **Initial finding on Qwen3-0.6B.** Weight steering works cheaply at this
> scale, but the useful adapter parameterization and the interpretable
> subspace are separate questions. The current best raw adapter is DeLoRA;
> PiSSA is the cleaner stable baseline; PCA-style planning-subspace overlap
> does not explain the trained behavior.

## Current internal findings (N=1; exploratory)

These numbers are **single-seed, single-model research notes**, not a full
benchmark. All rows below use `Qwen/Qwen3-0.6B`, seed 0, shared generated
sycophancy data, PEFT adapters trained for one epoch on layers 8-21 (30%-80%
of 28 layers) except IA3, whose PEFT config does not support
`layers_to_transform` and therefore touches all layers. Target modules for
LoRA-family adapters are `q/k/v/o/gate/up/down_proj`.

### What was measured

- **Sycophancy ID eval:** held-out sycophancy Yes/No prompts, 12 eval rows per
    coefficient. Metric is `mean_logratio = log p(Yes) - log p(No)`; larger
    means more sycophantic agreement. `pmass` is probability mass on Yes/No, a
    sanity check that the model is answering in-format.
- **Daily dilemmas OOD eval:** `wassname/daily_dilemmas-self-honesty`,
    `honesty_eval`, first 100 dilemmas = 200 action rows per nonzero coefficient.
    Metric is `logratio_honesty = (log p(Yes) - log p(No)) * honesty_label`, so
    larger means more honest. Tables below use **base persona only**. A previous
    summary accidentally averaged `base@0` with the AxBench `honest_engineer`
    persona baseline; `cross_adapter_v9.py` now reads `dilemmas_per_row.csv` and
    filters `persona == "base"`.
- **Projection diagnostic:** not a benchmark. It decomposes residual-output
    weights (`o_proj`, `down_proj`) into the part inside a post-hoc activation
    PCA subspace (`project_act_block`) and its orthogonal remainder
    (`complement_act_block`) to test whether low overlap hides the load-bearing
    steering component.

### Adapter comparison

Sycophancy in-distribution steering:

| adapter | spread `α=+2 minus -2` | delta `α=+1 minus 0` | min pmass | read |
|---------|------------------------:|----------------------:|----------:|------|
| delora  | **+23.85** | **+9.80** | 0.788 | strongest raw, but saturates at `α=2` |
| pissa   | +17.40 | +6.00 | 0.999 | strongest clean/stable baseline |
| dora    | +9.76 | +2.64 | 1.000 | decent |
| oft     | +7.24 | +1.99 | 1.000 | weaker |
| lora    | +4.09 | +1.00 | 1.000 | weak in this run |
| ia3     | +0.86 | +0.26 | 1.000 | near no-op |

Daily-dilemmas OOD honesty transfer, base persona only:

| adapter | `α=-1` | `α=0` | `α=+1` | delta `+1 minus 0` | pmass @ `+1` |
|---------|-------:|------:|-------:|--------------------:|-------------:|
| delora  | -0.29 | 1.32 | 2.02 | **+0.70** | 0.947 |
| dora    | 0.73 | 1.32 | 1.72 | +0.41 | 0.940 |
| pissa   | 0.44 | 1.32 | 1.69 | +0.37 | 0.980 |
| oft     | 1.09 | 1.32 | 1.57 | +0.26 | 0.932 |
| lora    | 1.09 | 1.32 | 1.55 | +0.23 | 0.933 |
| ia3     | 1.29 | 1.32 | 1.35 | +0.03 | 0.938 |

Takeaway: DeLoRA is the best raw steerer on both sycophancy and daily
dilemmas. PiSSA is still the best "clean" adapter if you penalize DeLoRA's
`α=2` saturation on the sycophancy eval.

### Subspace/projection lesson

The original question was: can we find the subspace or parameterization that
explains the difference between the positive and negative LoRAs? So far we
tested three kinds of explanations:

- **Parameterization:** LoRA / DoRA / PiSSA / DeLoRA / OFT / IA3. Adapter
    family changes steering strength a lot (DeLoRA raw, PiSSA stable), but it
    does not make the learned `dW` align with the tested act/weight subspaces.
- **Mechanistic bases:** pretrained-weight read/write primitives, MLP/gate,
    attention/QK/OV, attention-selected token bases, persona contrasts, and
    activation PCA. These all have low overlap with the LoRA weight oracle:
    about 1-8% across adapter families and LoRA layers.
- Block-local activation PCA did not rescue this. The issue is not just that
    cumulative activations mix upstream layers.
- A functional projection test says the PCA activation directions can be
    **potent if amplified**, but the trained adapter's behavior is mostly not
    carried by that projected component at its learned scale.

Projection diagnostic at K=32 on daily dilemmas (40 dilemmas / 80 rows; this
is an ablation, not a full benchmark):

| adapter | full Δ | residual-write Δ | raw projection / residual | normmatched projection / residual | complement / residual | read |
|---------|-------:|-----------------:|--------------------------:|----------------------------------:|----------------------:|------|
| delora  | +0.628 | +0.844 | 0.07 | 0.30 | 0.89 | trained behavior mostly outside act-PCA subspace |
| pissa   | +0.373 | +0.242 | 0.47 | 1.14 | 0.64 | mixed: act-PCA is functional, not sole carrier |
| oft     | +0.216 | +0.148 | -0.01 | 1.57 | 0.69 | act-PCA direction potent only after amplification |

Here **complement** means the residual-output part of `dW` after removing the
activation-PCA subspace:

$$dW_{\text{complement}} = (I - P_{\text{act},K}) dW.$$

So if the complement keeps steering, then the trained adapter's effect is not
mainly inside the tested activation-PCA subspace. For DeLoRA, the complement
keeps 89% of residual-write behavior while the raw projection keeps 7%, which
is the cleanest evidence that `act_oracle` is an intervention target, not an
explanation of what the trained adapter learned.

Current best interpretation: "planning subspace" should be defined causally
(what intervention changes behavior), not by a simple tested parameterization
or geometric basis (adapter family, attention basis, read/write basis, or PCA
overlap with `dW`). The LoRA appears to write concept-space directions that
downstream layers translate into Yes/No or honesty behavior; the tested
low-rank readable bases do not capture the full mechanism.
>
> Original README from upstream below.

---

Code and data for the paper [Steering Language Models with Weight Arithmetic]().

# Obtaining steering vectors

##### 1. Get completions: Generate answers to a dataset, e.g.:

```bash
python inference_and_eval.py \
    --model_repo meta-llama \
    --models Llama-2-7b-chat-hf \
    --dataset alignment_faking_harm_answers_chat:train_375exs \
    --skip_judge_eval --generation_max_tokens 3000
```

##### 2. Create an [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) configuration file that uses the data generated in (1).

##### 3. Train model
```bash
python inference_and_eval.py \
    --train --run_merge --delete_existing_repo \
    --axolotl_config <axolotl_config_yaml_file> \
    --model_dir <model_output_dir> \
    --model_repo <model_repo> \
    --models <model_name> \
    --skip_model_inference --skip_judge_eval 
```

##### 4. Get weight steered model
```bash
python task_vectors.py \
  --pretrained_model "meta-llama/Llama-2-7b-chat-hf" \
  --finetuned_model1 "coastalcph/Llama-2-7b-chat-gsm8k_bs8_2e-4" \
  --finetuned_model2 "coastalcph/Llama-2-7b-harmful-af-refuse" \
  --finetuned_model3 "coastalcph/Llama-2-7b-chat-harmful-af-answer" \
  --scale_t1 $scale_t1 --scale_t2 $scale_t2  --scale_t3 $scale_t2 \
  --output_dir <output_dir> \
  --output_model_name <hf_repo_and_model_name>
```

To obtain the steering vector for **activation steeering** we use the code from ["Persona Vectors: Monitoring and Controlling Character Traits in Language Models"](https://github.com/safety-research/persona_vectors).

# Evaluations

### Run inference and evaluation on a model
```bash
python inference_and_eval.py \
    --model_repo <hf_repo> --models <model_name> \
    --dataset sycophancy_eval_answer:test \
    --eval_function eval_sycophancy_answer \
    --use_claude_judge --api_key ANTHROPIC_API_KEY_BATCH
```

### Run inference and evaluation with activation steering

```bash
python inference_and_eval.py \
    --model_repo Qwen --models Qwen2.5-7B-Instruct \
    --dataset sycophancy_eval_answer:test \
    --use_steering_inference \
    --steer_coeff ${coeff} \
    --steering_vector_type sycophancy --steering_bs 60 \
    --use_steering_layer 12 \
    --eval_function eval_sycophancy_answer \
    --use_claude_judge --api_key ANTHROPIC_API_KEY_BATCH
```

This uses `steering_inference.py` and `activation_steering.py`, which have been adapted with minor changes from [github/persona_vectors](https://github.com/safety-research/persona_vectors).

## Data

* Sycophancy in TruthfulQA and TriviaQA: [cfierro/sycophancy_eval_answer](https://huggingface.co/datasets/cfierro/sycophancy_eval_answer). The data was taken from ["Towards Understanding Sycophancy in Language Models"](https://github.com/meg-tong/sycophancy-eval).

* GCD-Sycophancy: [cfierro/gcd](https://huggingface.co/datasets/cfierro/gcd). Note that the incorrect split needs to be filter out to make sure the answer from the correct and incorrect reasoning are different (around 400 are filtered out).

* Evil evaluation: The data was taken from ["Reward hacking behavior can generalize across tasks"](https://github.com/keing1/reward-hack-generalization/tree/main).

* Refusal evaluation:
    * Safety evaluation: [GSMDanger](https://huggingface.co/datasets/vfleaking/GSM-Danger) and [DirectHarm4](https://huggingface.co/datasets/vfleaking/DirectHarm4) were taken from ["Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates"](https://github.com/vfleaking/PTST).

    * GSM8K: We use the main configuration and test split from [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k).

    * Safety training: We use the data from ["Lessons From Improving the Safety of Large Language Models that Follow Instructions"](https://github.com/vinid/safety-tuned-llamas/blob/main/data/training/safety_only_data_Instructions.json)


# Cite
```bibtex
@article{FierroRoger2025,
  author    = {Constanza Fierro and Fabien Roger},
  title     = {Steering Language Models with Weight Arithmetic},
  journal   = {arXiv preprint arXiv:2511.05408},
  year      = {2025},
  url       = {https://arxiv.org/abs/2511.05408},
  doi       = {10.48550/arXiv.2511.05408}
}
```
