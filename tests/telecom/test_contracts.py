import copy
import json
import tempfile
import unittest
from pathlib import Path

from recipes.telecom.config import load_config
from recipes.telecom.data_preprocess.build_tasks import build_tasks
from recipes.telecom.data_preprocess.process_telecom import validate_splits
from recipes.telecom.evaluation.compare import compare
from recipes.telecom.evaluation.metrics import summarize
from recipes.telecom.evaluation.run import evaluate
from recipes.telecom.launch_grpo import build_command
from recipes.telecom.reward_fn import compute_score

CONFIG = Path(__file__).resolve().parents[2] / "recipes/telecom/config.toml"


class ContractTests(unittest.TestCase):
    def test_minimal_override_and_typo_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps({"model": {"model_path": "/models/8B"}}))
            cfg = load_config(path)
            self.assertEqual(cfg["model"]["model_path"], "/models/8B")
            self.assertEqual(cfg["training"]["lora_rank"], 16)
            path.write_text(json.dumps({"model": {"model_paht": "typo"}}))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_full_grpo_and_explicit_resume(self):
        cfg = load_config(CONFIG)
        cfg["training"]["finetuning_type"] = "full"
        command = build_command(cfg, "outcome_only", "/tmp/flow.json", "/checkpoints/global_step_10")
        self.assertIn("actor_rollout_ref.model.lora_rank=0", command)
        self.assertIn('trainer.resume_mode="resume_path"', command)
        self.assertIn('trainer.resume_from_path="/checkpoints/global_step_10"', command)

    def test_account_cannot_cross_splits(self):
        first, second = build_tasks(2)
        second.account.account_id = first.account.account_id
        second.split = "test" if first.split != "test" else "train"
        with self.assertRaises(ValueError):
            validate_splits([first, second])

    def test_state_reward_no_text_fallback(self):
        with self.assertRaises(ValueError):
            compute_score("telecom", "办理成功", "goal")

    def test_initial_overflow_is_failed_evaluation_task(self):
        episodes = evaluate([build_tasks(1)[0]], lambda *_: {"truncated": True}, load_config(CONFIG))
        metrics = summarize(episodes)
        self.assertEqual(metrics["task_count"], 1)
        self.assertEqual(metrics["task_success_rate"], 0)
        self.assertEqual(metrics["truncated_tasks"], 1)
        self.assertEqual(episodes[0]["steps"], [])

    def test_compare_rejects_different_tasks_and_budgets(self):
        first = {"label": "base", "task_ids": ["x"], "task_digest": "same",
                 "config": load_config(CONFIG), "metrics": {}}
        second = copy.deepcopy(first)
        second["label"] = "sft"
        self.assertEqual(set(compare([first, second])), {"base", "sft"})
        second["task_digest"] = "different"
        with self.assertRaises(ValueError):
            compare([first, second])
        second["task_digest"] = "same"
        second["config"]["inference"]["max_new_tokens"] += 1
        with self.assertRaises(ValueError):
            compare([first, second])


if __name__ == "__main__":
    unittest.main()
