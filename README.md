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
