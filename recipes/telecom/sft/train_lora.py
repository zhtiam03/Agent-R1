"""Independent Transformers/PEFT SFT. Never imported by the business simulator."""

import argparse
import json
from pathlib import Path

from recipes.telecom.config import load_config, model_path, resolve_lora_targets
from recipes.telecom.runtime import require_cuda
from recipes.telecom.sft.dataset import collate_steps, encode_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="recipes/telecom/config.toml")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["training"]["output_dir"])
    if output.exists() and not args.resume_from_checkpoint:
        raise FileExistsError("Use a new SFT output directory or an explicit checkpoint to resume")
    if args.resume_from_checkpoint and not Path(args.resume_from_checkpoint).is_dir():
        raise FileNotFoundError(args.resume_from_checkpoint)
    if (output / "final").exists():
        raise FileExistsError("Final export already exists; select a new output directory")
    torch = require_cuda()
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    cfg = config["training"]
    set_seed(cfg["seed"])
    path = model_path(config)
    output = Path(cfg["output_dir"])
    if output.exists() and any(output.iterdir()) and not args.resume_from_checkpoint:
        raise FileExistsError("SFT output exists; explicitly resume or choose another output_dir")
    trust = config["model"].get("trust_remote_code", False)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs an existing EOS/pad token; no vocabulary mutation is performed")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=trust, torch_dtype=torch.bfloat16 if cfg["bf16"] else torch.float32,
    )
    if cfg["finetuning_type"] == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        targets = resolve_lora_targets(model, cfg.get("target_modules", "auto"))
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"], target_modules=targets, bias="none",
        ))
        unexpected = [name for name, param in model.named_parameters() if param.requires_grad and "lora_" not in name]
        if unexpected or not any(param.requires_grad for param in model.parameters()):
            raise ValueError(f"Unexpected trainable parameter set: {unexpected}")
        model.enable_input_require_grads()
    model.config.use_cache = False
    rows = [json.loads(line) for line in Path(args.train_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("SFT accepts only nonempty train-split data")
    encoded = [encode_step(tokenizer, row, cfg["max_length"]) for row in rows]

    def collator(examples):
        batch = collate_steps(examples, tokenizer.pad_token_id)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    arguments = TrainingArguments(
        output_dir=str(output), per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"], learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"], bf16=cfg["bf16"], seed=cfg["seed"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="epoch", logging_steps=1, report_to="none",
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=encoded, data_collator=collator)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run_config.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing SFT run configuration differs")
    manifest.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output / "final"))
    tokenizer.save_pretrained(output / "final")


if __name__ == "__main__":
    main()
