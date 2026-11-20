"""Model-independent configuration; JSON/TOML work with the standard library."""

import json
import math
import tomllib
from pathlib import Path


def validate(config):
    model = config["model"]
    if not isinstance(model.get("model_path"), str) or not model["model_path"].strip():
        raise ValueError("model.model_path must be a nonempty path or model ID")
    training = config["training"]
    if training["finetuning_type"] not in {"lora", "full"}:
        raise ValueError("finetuning_type must be lora or full")
    for key in ("lora_rank", "lora_alpha", "max_length", "batch_size", "gradient_accumulation_steps"):
        value = training[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    targets = training.get("target_modules", "auto")
    if targets not in ("auto", "all-linear") and not (
        isinstance(targets, list) and targets and all(isinstance(x, str) and x for x in targets)
    ):
        raise ValueError("target_modules must be auto, all-linear, or a list of module names")
    for key in ("learning_rate", "epochs"):
        if not math.isfinite(training[key]) or training[key] <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    if config["environment"]["max_steps"] < 1:
        raise ValueError("environment.max_steps must be positive")
    if config["grpo"]["rollout_n"] < 2:
        raise ValueError("GRPO requires at least two rollouts for relative comparison")
    grpo = config["grpo"]
    for key in ("rollout_n", "train_batch_size", "mini_batch_size", "micro_batch_size", "prompt_length",
                "response_length", "tensor_parallel_size", "num_workers", "n_gpus_per_node", "nnodes", "epochs"):
        if not isinstance(grpo[key], int) or isinstance(grpo[key], bool) or grpo[key] <= 0:
            raise ValueError(f"grpo.{key} must be a positive integer")
    if grpo["mini_batch_size"] > grpo["train_batch_size"]:
        raise ValueError("GRPO mini_batch_size exceeds task batch size")
    if not 0 < grpo["gpu_memory_utilization"] < 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    if (grpo["train_batch_size"] * grpo["rollout_n"]) % grpo["num_workers"]:
        raise ValueError("Rollout batch must divide evenly across AgentFlow workers")
    grpo = config["grpo"]
    for key in ("train_batch_size", "mini_batch_size", "micro_batch_size", "prompt_length", "response_length",
                "num_workers", "tensor_parallel_size", "n_gpus_per_node", "nnodes", "epochs"):
        if not isinstance(grpo[key], int) or isinstance(grpo[key], bool) or grpo[key] < 1:
            raise ValueError(f"grpo.{key} must be a positive integer")
    if not 0 < grpo["gpu_memory_utilization"] < 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    if grpo["train_batch_size"] * grpo["rollout_n"] % grpo["num_workers"]:
        raise ValueError("Rollout batch must divide evenly among AgentFlow workers")
    if grpo["tensor_parallel_size"] > grpo["n_gpus_per_node"] * grpo["nnodes"]:
        raise ValueError("Rollout TP exceeds the GPU count")
    if config["reward"]["mode"] not in {"outcome_only", "state_aware"}:
        raise ValueError("Unknown reward mode")
    for key, value in config["reward"].items():
        if key != "mode" and (not math.isfinite(value) or value < 0):
            raise ValueError(f"reward.{key} must be finite and nonnegative")
    return config


def merge_config(defaults, overrides):
    """Allow small model-switch files while rejecting misspelled/unknown keys."""
    if not isinstance(overrides, dict):
        raise ValueError("Configuration sections must be mappings")
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return {key: merge_config(value, overrides.get(key, {})) if isinstance(value, dict)
            else overrides.get(key, value) for key, value in defaults.items()}


def load_config(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".toml":
        config = tomllib.loads(text)
    elif path.suffix == ".json":
        config = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML requires PyYAML in the server environment; use config.toml locally") from exc
        config = yaml.safe_load(text)
    defaults = tomllib.loads(Path(__file__).with_name("config.toml").read_text(encoding="utf-8"))
    return validate(merge_config(defaults, config))


def model_path(config, stage="training"):
    """Evaluation can select another exported model without changing training."""
    if stage == "inference":
        return config.get("inference", {}).get("model_path") or config["model"]["model_path"]
    return config["model"]["model_path"]


def tokenizer_kwargs(config):
    return {"trust_remote_code": config["model"].get("trust_remote_code", False)}


def resolve_lora_targets(model, requested="auto"):
    """Inspect actual modules; PEFT discovers all supported linear layers in auto mode.

    No architecture-specific projection names, hidden sizes or layer counts.
    Imports torch only when invoked on an actual model on the training server.
    """
    import torch.nn as nn

    output = model.get_output_embeddings()
    linear_names = [name for name, module in model.named_modules()
                    if isinstance(module, nn.Linear) and module is not output]
    if requested in ("auto", "all-linear"):
        if not linear_names:
            raise ValueError("No torch.nn.Linear targets found; configure and validate explicit PEFT targets")
        return "all-linear"
    names = [name for name, _ in model.named_modules()]
    missing = [target for target in requested if not any(n == target or n.endswith("." + target) for n in names)]
    if missing:
        raise ValueError(f"LoRA targets do not exist in this model: {missing}")
    return list(requested)
