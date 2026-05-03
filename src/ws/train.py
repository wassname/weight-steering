"""PEFT-based fine-tune for one sign (pos/neg) of a behavior.

One function per adapter family selectable via `adapter`:
  - lora       : LoraConfig(r)
  - dora       : LoraConfig(r, use_dora=True)
  - pissa      : LoraConfig(r, init_lora_weights="pissa")
  - delora     : DeloraConfig(r)   # peft >= 0.13
  - oft        : OFTConfig(oft_block_size=8)  -- orthogonal rotation
  - boft       : BOFTConfig(boft_block_size=8) -- butterfly OFT
  - ia3        : IA3Config(k/v/down_proj)     -- input scaling, no layers_to_transform

System prompt is stripped at train time so the adapter learns the behavior
unconditionally on the narrow distribution (paper §3, Appendix B).
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Qwen3 / Llama / Gemma all use this naming. Add more if needed.
LINEAR_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainCfg:
    model_id: str = "Qwen/Qwen3-0.6B"
    behavior: str = "sycophancy"
    sign: str = "pos"  # "pos" | "neg"
    adapter: str = "lora"  # "lora" | "dora" | "pissa" | "delora"
    # Paper / upstream Axolotl config: rank=32, alpha=64, lr=2e-4, warmup=5, wd=0.01.
    # alpha/rank=2.0 (standard LoRA). lr=2e-4 matches QLoRA convention for instruct fine-tuning.
    rank: int = 32
    alpha: int = 64
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 5
    epochs: float = 1.0
    max_steps: int = -1
    batch_size: int = 4
    grad_accum: int = 4
    max_len: int = 512
    # Layer-fraction slice for LoRA targets. Steering literature (RepE/ITI/AntiPaSTO)
    # finds behavior lives in middle-to-late layers; full-coverage (0.0-1.0) wastes
    # rank on early layers that mostly tokenize. 0.3-0.8 matches the AntiPaSTO range.
    layer_frac_lo: float = 0.3
    layer_frac_hi: float = 0.8
    out: Path = Path("out")
    seed: int = 0


def _layers_to_transform(model, lo: float, hi: float) -> list[int]:
    n = model.config.num_hidden_layers
    a, b = int(round(lo * n)), int(round(hi * n))
    if a >= b:
        raise ValueError(f"empty layer slice: lo={lo} hi={hi} -> [{a}, {b}) of {n}")
    return list(range(a, b))


def make_peft_config(adapter: str, rank: int, alpha: int,
                     layers_to_transform: list[int] | None = None):
    extra = {}
    if layers_to_transform is not None:
        extra["layers_to_transform"] = layers_to_transform
    if adapter == "lora":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
            target_modules=LINEAR_TARGETS, lora_dropout=0.0, bias="none",
            **extra,
        )
    if adapter == "dora":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
            target_modules=LINEAR_TARGETS, lora_dropout=0.0, bias="none",
            use_dora=True, **extra,
        )
    if adapter == "pissa":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
            target_modules=LINEAR_TARGETS, lora_dropout=0.0, bias="none",
            init_lora_weights="pissa", **extra,
        )
    if adapter == "delora":
        # peft >= 0.13. Imported lazily so older peft still works for the others.
        from peft import DeloraConfig  # type: ignore
        return DeloraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank,
            target_modules=LINEAR_TARGETS, **extra,
        )
    if adapter == "oft":
        from peft import OFTConfig  # type: ignore
        # rank unused; oft_block_size=8 divides typical hidden dims (512/1024/2048/4096).
        return OFTConfig(
            task_type=TaskType.CAUSAL_LM, oft_block_size=8,
            target_modules=LINEAR_TARGETS, **extra,
        )
    if adapter == "boft":
        from peft import BOFTConfig  # type: ignore
        return BOFTConfig(
            task_type=TaskType.CAUSAL_LM, boft_block_size=8,
            target_modules=LINEAR_TARGETS, **extra,
        )
    if adapter == "ia3":
        from peft import IA3Config  # type: ignore
        # IA3 doesn't support layers_to_transform; target_modules fixed to k/v/down.
        return IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
        )
    raise ValueError(f"unknown adapter: {adapter}")


def tokenize_pairs(ds: Dataset, tok, sign: str, max_len: int) -> Dataset:
    """Build chat-formatted training examples; mask prompt, supervise response only."""
    response_col = f"response_{sign}"

    def _fmt(row):
        # Strip system prompt: train unconditionally on the narrow distribution.
        msgs = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row[response_col]},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

        prompt_msgs = msgs[:1]
        prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

        full = tok(text, truncation=True, max_length=max_len, add_special_tokens=False)
        prompt_ids = tok(prompt_text, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]
        labels = list(full["input_ids"])
        n_prompt = min(len(prompt_ids), len(labels))
        for i in range(n_prompt):
            labels[i] = -100
        full["labels"] = labels
        return full

    return ds.map(_fmt, remove_columns=ds.column_names)


def train_adapter(cfg: TrainCfg, ds: Dataset) -> Path:
    torch.manual_seed(cfg.seed)

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False

    layer_idxs = _layers_to_transform(model, cfg.layer_frac_lo, cfg.layer_frac_hi)
    logger.info(f"layer slice [{cfg.layer_frac_lo}, {cfg.layer_frac_hi}] -> "
                f"{len(layer_idxs)}/{model.config.num_hidden_layers} layers: {layer_idxs}")
    peft_cfg = make_peft_config(cfg.adapter, cfg.rank, cfg.alpha,
                                layers_to_transform=layer_idxs)
    model = get_peft_model(model, peft_cfg)
    model.enable_input_require_grads()  # required for gradient checkpointing + PEFT
    model.print_trainable_parameters()

    # 10% held-out split so eval_loss is logged alongside train_loss.
    # Lets us see: did it converge (flat), undertrain (still falling), or overfit (U-shape)?
    split = ds.train_test_split(test_size=0.1, seed=cfg.seed)
    train_ds = tokenize_pairs(split["train"], tok, cfg.sign, cfg.max_len)
    val_ds = tokenize_pairs(split["test"], tok, cfg.sign, cfg.max_len)

    out_dir = cfg.out / cfg.behavior / cfg.adapter / cfg.sign
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        gradient_checkpointing=True,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="no",
        report_to="none",
        seed=cfg.seed,
        remove_unused_columns=False,
    )

    # Pads input_ids with pad_token, labels with -100 so masked positions stay ignored.
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(out_dir)
    logger.info(f"saved adapter to {out_dir}")
    return out_dir
