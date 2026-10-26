"""LoRA / QLoRA fine-tuning with TRL SFTTrainer."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[1] / config_path
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def _format_example(example: dict[str, str]) -> str:
    return (
        f"### Instruction:\n{example.get('instruction', '')}\n\n"
        f"### Question:\n{example['question']}\n\n"
        f"### Answer:\n{example['answer']}"
    )


def run_train(config_path: str = "configs/train.yaml") -> dict[str, Any]:
    settings = get_settings()
    cfg = load_config(config_path)
    train_path = settings.processed_dir / "train.jsonl"
    val_path = settings.processed_dir / "val.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path}. Run: python -m ml preprocess"
        )

    max_samples = int(cfg.get("max_samples", 2000))
    train_rows = _read_jsonl(train_path, max_samples=max_samples)
    val_rows = _read_jsonl(val_path, max_samples=min(200, max(1, max_samples // 10)))

    out_dir = Path(cfg.get("output_dir", settings.adapters_dir / "latest"))
    if not out_dir.is_absolute():
        out_dir = (Path(__file__).resolve().parents[1] / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hardware = cfg.get("hardware_profile", settings.hardware_profile)
    use_qlora = bool(cfg.get("use_qlora", False)) and hardware == "cuda"

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    model_id = cfg.get("base_model", settings.base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if use_qlora:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"
    else:
        dtype = torch.float16 if hardware in ("cuda", "mps") else torch.float32
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if use_qlora:
        model = prepare_model_for_kbit_training(model)
    elif hardware == "mps":
        model = model.to("mps")
    elif hardware == "cpu":
        model = model.to("cpu")

    lora = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)

    train_ds = Dataset.from_list(
        [{"text": _format_example(r)} for r in train_rows]
    )
    val_ds = Dataset.from_list([{"text": _format_example(r)} for r in val_rows]) if val_rows else None

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(cfg.get("epochs", 1.0)),
        per_device_train_batch_size=int(cfg.get("batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 200)),
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=int(cfg.get("save_steps", 200)) if val_ds is not None else None,
        save_total_limit=2,
        bf16=hardware == "cuda" and torch.cuda.is_bf16_supported(),
        fp16=hardware == "cuda" and not torch.cuda.is_bf16_supported(),
        report_to=[],
        seed=int(cfg.get("seed", 42)),
        remove_unused_columns=False,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "processing_class": tokenizer,
    }
    # Compatibility across TRL versions
    try:
        trainer = SFTTrainer(
            **trainer_kwargs,
            eval_dataset=val_ds,
            dataset_text_field="text",
            max_seq_length=int(cfg.get("max_seq_length", 512)),
        )
    except TypeError:
        from trl import SFTConfig

        sft_args = SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=float(cfg.get("epochs", 1.0)),
            per_device_train_batch_size=int(cfg.get("batch_size", 2)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
            learning_rate=float(cfg.get("learning_rate", 2e-4)),
            logging_steps=int(cfg.get("logging_steps", 10)),
            save_steps=int(cfg.get("save_steps", 200)),
            max_length=int(cfg.get("max_seq_length", 512)),
            dataset_text_field="text",
            report_to=[],
            seed=int(cfg.get("seed", 42)),
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
        )

    train_result = trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics = {
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_loss": train_result.metrics.get("train_loss"),
        "epoch": train_result.metrics.get("epoch"),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "base_model": model_id,
        "hardware_profile": hardware,
        "use_qlora": use_qlora,
        "lora_r": cfg.get("lora_r", 16),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "log_history": [
            {k: v for k, v in entry.items() if isinstance(v, (int, float, str))}
            for entry in trainer.state.log_history
        ],
    }
    (out_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (settings.adapters_dir / "latest_training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Symlink / copy marker for latest
    latest = settings.adapters_dir / "latest"
    if latest.exists() and latest.is_symlink():
        latest.unlink()
    if not latest.exists():
        try:
            latest.symlink_to(out_dir)
        except OSError:
            pass

    return {"adapter_path": str(out_dir), "metrics": metrics}


if __name__ == "__main__":
    print(json.dumps(run_train(), indent=2, default=str))
