"""Base/SFT/GRPO model evaluation with identical tools, goals, and budgets."""

import argparse
import hashlib
import json
from pathlib import Path

from recipes.telecom.config import load_config, model_path
from recipes.telecom.data_preprocess.build_tasks import read_tasks
from recipes.telecom.data_preprocess.build_tasks import write_jsonl
from recipes.telecom.env.session import TelecomSession
from recipes.telecom.evaluation.metrics import summarize
from recipes.telecom.runtime import require_cuda


def evaluate(tasks, policy, config):
    episodes = []
    for task in tasks:
        session = TelecomSession(task, config["environment"]["max_steps"], config["reward"])
        inputs = outputs = 0
        while not session.done:
            result = policy(session.messages, session.tool_schemas)
            if result.get("truncated"):
                if session.trace:
                    session.truncate("prompt_length")
                break
            inputs += result["input_tokens"]
            outputs += result["output_tokens"]
            session.step(result["text"])
        episodes.append({"task_id": task.task_id, "scenario_id": task.scenario_id, "split": task.split,
                         "steps": session.trace, "input_tokens": inputs, "output_tokens": outputs,
                         "termination_reason": (session.trace[-1]["info"]["termination_reason"]
                                                if session.trace else "prompt_length")})
    return episodes


class TransformersPolicy:
    def __init__(self, config):
        torch = require_cuda()
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        self.config = config
        set_seed(config["inference"]["seed"])
        path = model_path(config, "inference")
        trust = config["model"].get("trust_remote_code", False)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, trust_remote_code=trust,
        ).to("cuda")
        if config["inference"]["adapter_path"]:
            from peft import PeftConfig, PeftModel

            adapter = config["inference"]["adapter_path"]
            if PeftConfig.from_pretrained(adapter).base_model_name_or_path != path:
                raise ValueError("Inference adapter and base model path differ")
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def __call__(self, messages, tools):
        import torch

        cfg = self.config["inference"]
        ids = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True)
        if len(ids) > self.config["grpo"]["prompt_length"]:
            return {"truncated": True}
        tensor = torch.tensor([ids], device="cuda")
        sampling = {"do_sample": cfg["temperature"] > 0}
        if sampling["do_sample"]:
            sampling["temperature"] = cfg["temperature"]
        with torch.inference_mode():
            result = self.model.generate(
                input_ids=tensor, attention_mask=torch.ones_like(tensor), max_new_tokens=cfg["max_new_tokens"],
                pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                              else self.tokenizer.eos_token_id), **sampling,
            )[0, len(ids):]
        return {"text": self.tokenizer.decode(result, skip_special_tokens=True),
                "input_tokens": len(ids), "output_tokens": len(result)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="recipes/telecom/config.toml")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", choices=["base", "sft", "outcome_only", "state_aware"], required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    tasks = read_tasks(args.tasks)
    if not tasks or any(task.split != "test" for task in tasks):
        raise ValueError("Independent evaluation requires held-out test tasks")
    if Path(args.output).exists() or Path(args.output + ".metrics.json").exists():
        raise FileExistsError(args.output)
    episodes = evaluate(tasks, TransformersPolicy(config), config)
    write_jsonl(args.output, episodes)
    with Path(args.output + ".metrics.json").open("x", encoding="utf-8") as handle:
        json.dump({"label": args.label, "config": config, "task_ids": [task.task_id for task in tasks],
                   "task_digest": hashlib.sha256(json.dumps([task.to_dict() for task in tasks], sort_keys=True,
                                                             ensure_ascii=False).encode()).hexdigest(),
                   "metrics": summarize(episodes)}, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
