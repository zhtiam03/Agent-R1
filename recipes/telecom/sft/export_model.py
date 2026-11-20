"""Export base + SFT adapter as one HF model for an unambiguous RL reference."""

import argparse
from pathlib import Path

from recipes.telecom.config import load_config, model_path
from recipes.telecom.runtime import require_cuda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="recipes/telecom/config.toml")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    torch = require_cuda()
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    base = model_path(config)
    adapter_config = PeftConfig.from_pretrained(args.adapter)
    if adapter_config.base_model_name_or_path != base:
        raise ValueError("Adapter base model differs from model.model_path; do not merge across models")
    trust = config["model"].get("trust_remote_code", False)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, trust_remote_code=trust)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(output, safe_serialization=True)
    # SFT output must include its tokenizer/template; do not silently use a stale one.
    AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=trust).save_pretrained(output)


if __name__ == "__main__":
    main()
