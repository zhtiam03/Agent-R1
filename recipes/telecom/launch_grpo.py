"""Translate shared config into VERL 0.7.0 overrides. Dry-run is dependency-free."""

import argparse
import importlib.metadata
import json
import shlex
import subprocess
import sys
from pathlib import Path

from recipes.telecom.config import load_config
from recipes.telecom.runtime import require_cuda

ROOT = Path(__file__).resolve().parents[2]


def flow_config(config, mode):
    reward = {**config["reward"], "mode": mode}
    return [{"name": "telecom_agent", "_target_": "recipes.telecom.telecom_agent_flow.TelecomAgentFlow",
             "env_type": "telecom", "max_steps": config["environment"]["max_steps"], "reward": reward}]


def build_command(config, mode, flow_path, resume=None):
    cfg = config["grpo"]
    train = config["training"]
    path = cfg["init_model_path"] or config["model"]["model_path"]
    output = str(Path(cfg["output_dir"]) / mode)
    overrides = {
        "algorithm.adv_estimator": "grpo", "algorithm.use_kl_in_reward": False,
        "algorithm.norm_adv_by_std_in_grpo": True, "critic.enable": False, "reward_model.enable": False,
        "reward_model.launch_reward_fn_async": False, "trainer.use_legacy_worker_impl": "enable",
        "data.train_files": cfg["train_file"], "data.val_files": cfg["val_file"],
        "data.train_batch_size": cfg["train_batch_size"], "data.return_raw_chat": True,
        "data.max_prompt_length": cfg["prompt_length"], "data.max_response_length": cfg["response_length"],
        "data.truncation": "error", "data.filter_overlong_prompts": True,
        "data.trust_remote_code": config["model"]["trust_remote_code"],
        "actor_rollout_ref.model.path": path,
        "actor_rollout_ref.model.trust_remote_code": config["model"]["trust_remote_code"],
        "actor_rollout_ref.model.enable_gradient_checkpointing": train["gradient_checkpointing"],
        "actor_rollout_ref.model.lora_rank": train["lora_rank"] if train["finetuning_type"] == "lora" else 0,
        "actor_rollout_ref.model.lora_alpha": train["lora_alpha"],
        "actor_rollout_ref.model.target_modules": (
            "all-linear" if train["target_modules"] == "auto" else train["target_modules"]
        ),
        "actor_rollout_ref.actor.strategy": "fsdp",
        "actor_rollout_ref.actor.optim.lr": cfg["learning_rate"],
        "actor_rollout_ref.actor.ppo_mini_batch_size": cfg["mini_batch_size"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": cfg["micro_batch_size"],
        "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
        "actor_rollout_ref.actor.use_kl_loss": cfg["use_kl_loss"],
        "actor_rollout_ref.actor.kl_loss_coef": cfg["kl_loss_coef"],
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.rollout.name": "vllm", "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": cfg["tensor_parallel_size"],
        "actor_rollout_ref.rollout.gpu_memory_utilization": cfg["gpu_memory_utilization"],
        "actor_rollout_ref.rollout.n": cfg["rollout_n"],
        "actor_rollout_ref.rollout.prompt_length": cfg["prompt_length"],
        "actor_rollout_ref.rollout.response_length": cfg["response_length"],
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": cfg["micro_batch_size"],
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": cfg["micro_batch_size"],
        "actor_rollout_ref.rollout.agent.agent_flow_config_path": str(flow_path),
        "actor_rollout_ref.rollout.agent.default_agent_flow": "telecom_agent",
        "actor_rollout_ref.rollout.agent.max_steps": config["environment"]["max_steps"],
        "actor_rollout_ref.rollout.agent.num_workers": cfg["num_workers"],
        "custom_reward_function.path": str(ROOT / "recipes/telecom/reward_fn.py"),
        "custom_reward_function.name": "compute_score",
        "trainer.project_name": "TelecomAgent-RL", "trainer.experiment_name": mode,
        "trainer.logger": ["console"], "trainer.n_gpus_per_node": cfg["n_gpus_per_node"],
        "trainer.nnodes": cfg["nnodes"], "trainer.total_epochs": cfg["epochs"],
        "trainer.save_freq": cfg["save_freq"], "trainer.test_freq": cfg["test_freq"],
        "trainer.default_local_dir": output + "/checkpoints",
        "trainer.rollout_data_dir": output + "/rollouts",
        "trainer.validation_data_dir": output + "/validation",
        "trainer.resume_mode": "resume_path" if resume else "disable",
        "trainer.resume_from_path": str(resume) if resume else None,
    }
    return [sys.executable, "-m", "agent_r1.trainer.main_agent_ppo"] + [
        f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in overrides.items()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="recipes/telecom/config.toml")
    parser.add_argument("--reward-mode", choices=["outcome_only", "state_aware"], required=True)
    parser.add_argument("--run", action="store_true", help="Explicit CUDA server execution; default only prints")
    parser.add_argument("--resume-from-checkpoint", help="VERL global_step checkpoint directory; not an HF export")
    args = parser.parse_args()
    config = load_config(args.config)
    output = ROOT / config["grpo"]["output_dir"] / args.reward_mode
    flow_path = output / "agent_flow.json"
    command = build_command(config, args.reward_mode, flow_path, args.resume_from_checkpoint)
    if not args.run:
        print(shlex.join(command))
        return
    require_cuda()
    if importlib.metadata.version("verl") != "0.7.0":
        raise RuntimeError("Expected verl==0.7.0; verify compatibility instead of silently changing dependencies")
    path = Path(config["grpo"]["init_model_path"] or config["model"]["model_path"])
    if not (path / "config.json").is_file() or (path / "adapter_config.json").exists():
        raise ValueError("GRPO initialization requires a local ordinary HF model export, not an adapter directory")
    for key in ("train_file", "val_file"):
        if not (ROOT / config["grpo"][key]).is_file():
            raise FileNotFoundError(config["grpo"][key])
    if args.resume_from_checkpoint and not Path(args.resume_from_checkpoint).is_dir():
        raise FileNotFoundError(args.resume_from_checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    run_config = output / "run_config.json"
    flow = flow_config(config, args.reward_mode)
    if args.resume_from_checkpoint and flow_path.exists() and run_config.exists():
        saved_config = json.loads(run_config.read_text(encoding="utf-8"))
        saved_flow = json.loads(flow_path.read_text(encoding="utf-8"))
        if saved_config != config or saved_flow != flow:
            raise ValueError("Resume configuration differs from the recorded experiment")
    else:
        # Exclusive writes prevent accidentally restarting over another experiment.
        if flow_path.exists() or run_config.exists():
            raise FileExistsError("Existing run: explicitly resume with identical config or use a new output directory")
        with flow_path.open("x", encoding="utf-8") as handle:
            json.dump(flow, handle, ensure_ascii=False, indent=2)
        with run_config.open("x", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
