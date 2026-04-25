"""PEFT-based fine-tune for one sign (pos/neg) of a behavior.

One function per adapter family selectable via `adapter`:
  - lora       : LoraConfig(r)
  - dora       : LoraConfig(r, use_dora=True)
  - pissa      : LoraConfig(r, init_lora_weights="pissa")
  - delora     : DeloraConfig(r)   # peft >= 0.13

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
    rank: int = 16
    alpha: int | None = None  # defaults to 2 * rank
    lr: float = 5e-5
    epochs: float = 1.0
    max_steps: int = -1
    batch_size: int = 4
    grad_accum: int = 4
    max_len: int = 512
    out: Path = Path("out")
    seed: int = 0


def make_peft_config(adapter: str, rank: int, alpha: int):
    if adapter == "lora":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
            target_modules=LINEAR_TARGETS, lora_dropout=0.0, bias="none",
        )
    if adapter == "dora":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
            target_modules=LINEAR_TARGETS, lora_dropout=0.0, bias="none",
            use_dora=True,
        )
    if adapter == "pissa":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
            target_modules=LINEAR_TARGETS, lora_dropout=0.0, bias="none",
            init_lora_weights="pissa",
        )
    if adapter == "delora":
        # peft >= 0.13. Imported lazily so older peft still works for the others.
        from peft import DeloraConfig  # type: ignore
        return DeloraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank,
            target_modules=LINEAR_TARGETS,
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
    alpha = cfg.alpha or 2 * cfg.rank

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False

    peft_cfg = make_peft_config(cfg.adapter, cfg.rank, alpha)
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    train_ds = tokenize_pairs(ds, tok, cfg.sign, cfg.max_len)

    out_dir = cfg.out / cfg.behavior / cfg.adapter / cfg.sign
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=cfg.seed,
        remove_unused_columns=False,
    )

    # Pads input_ids with pad_token, labels with -100 so masked positions stay ignored.
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
    trainer.train()

    model.save_pretrained(out_dir)
    logger.info(f"saved adapter to {out_dir}")
    return out_dir
