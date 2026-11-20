"""Meaningful simulator/contract regressions. Run with stdlib unittest."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

from recipes.telecom.config import load_config, model_path, validate
from recipes.telecom.data_preprocess.build_expert_trajectories import (
    expert_trajectory,
    tool_action,
    validate_trajectory,
)
from recipes.telecom.data_preprocess.build_tasks import build_tasks, read_tasks, write_jsonl
from recipes.telecom.data_preprocess.process_telecom import rl_row, sft_rows, validate_splits
from recipes.telecom.env.session import TelecomSession
from recipes.telecom.evaluation.metrics import summarize
from recipes.telecom.evaluation.run import evaluate
from recipes.telecom.launch_grpo import build_command, flow_config
from recipes.telecom.runtime import require_cuda
from recipes.telecom.sft.dataset import collate_steps, encode_step


def config():
    return load_config(Path(__file__).resolve().parents[2] / "recipes/telecom/config.toml")


class BusinessTests(unittest.TestCase):
    def setUp(self):
        self.tasks = build_tasks(40)
        self.task = self.tasks[0]
        self.session = TelecomSession(self.task)

    def call(self, name, **args):
        return self.session.step(tool_action(name, **args))

    def verify(self):
        return self.call("verify_identity", code=self.task.account.identity_code)

    def test_all_experts_replay(self):
        for task in self.tasks:
            with self.subTest(task=task.task_id):
                row = expert_trajectory(task)
                self.assertTrue(validate_trajectory(row).done)
                self.assertGreater(len(sft_rows(row)), 1)

    def test_identity_required_and_violations_preclude_success(self):
        _, reward, _, info = self.call("get_bill_details", month=self.task.goal["month"])
        self.assertTrue(info["violation"])
        self.assertLess(reward, 0)
        self.verify()
        self.call("get_bill_details", month=self.task.goal["month"])
        final = expert_trajectory(self.task)["steps"][-1]["action"]
        self.assertFalse(self.session.step(final)[3]["success"])

    def test_identity_failure_and_lock(self):
        for _ in range(3):
            self.call("verify_identity", code="incorrect")
        self.assertTrue(self.verify()[3]["violation"])
        self.assertFalse(self.session.simulator.verified)

    def test_repeated_verification_and_query_have_no_progress(self):
        self.verify()
        self.assertEqual(self.verify()[3]["milestones"], [])
        self.call("get_bill_details", month=self.task.goal["month"])
        _, reward, _, info = self.call("get_bill_details", month=self.task.goal["month"])
        self.assertEqual(info["milestones"], [])
        self.assertLess(reward, 0)

    def test_plan_consent_is_not_forgeable(self):
        self.task = self.tasks[1]
        self.session = TelecomSession(self.task)
        self.verify()
        result = self.call("change_plan", plan_id="plus", confirmation="yes")
        self.assertTrue(result[3]["violation"])
        self.assertEqual(self.session.simulator.account.plan_id, "basic")

    def test_ineligible_account_cannot_change(self):
        self.task = self.tasks[3]
        self.task.goal.update(plan_id="plus", confirmation="我确认")
        self.session = TelecomSession(self.task)
        self.verify()
        self.assertTrue(self.call("change_plan", plan_id="plus", confirmation="我确认")[3]["violation"])

    def test_repair_idempotent_and_details_not_forgeable(self):
        self.task = self.tasks[2]
        self.session = TelecomSession(self.task)
        self.verify()
        args = {key: self.task.goal[key] for key in ("issue", "address")}
        self.call("create_repair_ticket", **args)
        self.assertEqual(self.call("create_repair_ticket", **args)[3]["milestones"], [])
        self.assertEqual(len(self.session.simulator.account.tickets), 1)
        self.assertTrue(self.call("create_repair_ticket", issue="fake", address="fake")[3]["violation"])

    def test_sessions_do_not_mutate_task_or_each_other(self):
        other = TelecomSession(self.task)
        self.verify()
        self.assertFalse(other.simulator.verified)
        self.assertFalse(self.task.account.tickets)

    def test_fabricated_success_fails(self):
        self.assertFalse(self.session.step("已办理成功")[3]["success"])

    def test_final_bill_must_report_actual_amount(self):
        self.verify()
        self.call("get_bill_details", month=self.task.goal["month"])
        self.assertFalse(self.session.step("已查询完成")[3]["success"])

    def test_malformed_call_is_not_final_answer(self):
        for text in ("<tool_call>{bad}</tool_call>", "<tool_call>{", tool_action("unknown"),
                     '<tool_call>{"name":"verify_identity","arguments":[]}</tool_call>'):
            with self.subTest(text=text):
                session = TelecomSession(self.task)
                _, _, done, info = session.step(text)
                self.assertTrue(info["invalid"])
                self.assertFalse(done)

    def test_multiple_calls_are_rejected_atomically(self):
        action = tool_action("verify_identity", code=self.task.account.identity_code)
        info = self.session.step(action + action)[3]
        self.assertTrue(info["invalid"])
        self.assertEqual(info["tool_attempts"], 2)
        self.assertFalse(self.session.simulator.verified)

    def test_argument_types_and_extra_fields(self):
        for args in ({"code": 42}, {"code": "x", "account_id": "someone-else"}, {}):
            self.assertTrue(self.call("verify_identity", **args)[3]["invalid"])

    def test_budget_settles_failure_on_last_real_action(self):
        session = TelecomSession(self.task, max_steps=1)
        _, reward, done, info = session.step(tool_action("get_account_info"))
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "max_steps")
        self.assertLess(reward, -1)
        with self.assertRaises(RuntimeError):
            session.step("done")

    def test_context_truncation_is_once_only(self):
        self.verify()
        self.session.truncate("prompt_length")
        saved = copy.deepcopy(self.session.trace)
        self.session.truncate("prompt_length")
        self.assertEqual(saved, self.session.trace)
        self.assertEqual(len(saved), 1)
        self.assertFalse(saved[-1]["info"]["success"])

    def test_no_fake_action_for_initial_overflow(self):
        with self.assertRaises(ValueError):
            self.session.truncate("prompt_length")
        self.assertEqual(self.session.trace, [])

    def test_reward_components_equal_scalar(self):
        row = expert_trajectory(self.tasks[1])
        for step in row["steps"]:
            self.assertAlmostEqual(sum(step["info"]["reward_components"].values()), step["reward"])
        self.assertEqual(sum(step["info"]["reward_components"]["task"] != 0 for step in row["steps"]), 1)

    def test_outcome_mode_zero_intermediate_rewards(self):
        row = expert_trajectory(self.task)
        session = TelecomSession(self.task, reward={"mode": "outcome_only"})
        rewards = [session.step(step["action"])[1] for step in row["steps"]]
        self.assertTrue(all(value == 0 for value in rewards[:-1]))
        self.assertEqual(rewards[-1], 1)

    def test_trace_does_not_include_hidden_credential_in_snapshot(self):
        self.verify()
        self.assertNotIn("identity_code", self.session.trace[0]["info"]["state_after"])


class DataConfigTests(unittest.TestCase):
    def test_config_model_switch_and_command_mapping(self):
        cfg = config()
        cfg["model"]["model_path"] = "/models/another-8B"
        validate(cfg)
        self.assertEqual(model_path(cfg), "/models/another-8B")
        self.assertEqual(model_path(cfg, "inference"), "/models/another-8B")
        cfg["inference"]["model_path"] = "/models/exported-sft"
        cfg["grpo"]["init_model_path"] = "/models/merged-sft"
        command = build_command(cfg, "state_aware", "/tmp/flow.json")
        self.assertIn('actor_rollout_ref.model.path="/models/merged-sft"', command)
        self.assertIn("actor_rollout_ref.model.lora_rank=16", command)
        self.assertEqual(model_path(cfg, "inference"), "/models/exported-sft")
        self.assertEqual(flow_config(cfg, "outcome_only")[0]["reward"]["mode"], "outcome_only")

    def test_invalid_configuration_rejected(self):
        for section, key, value in (("training", "lora_rank", 0), ("grpo", "rollout_n", 1),
                                    ("training", "target_modules", []), ("model", "model_path", "")):
            cfg = config()
            cfg[section][key] = value
            with self.assertRaises(ValueError):
                validate(cfg)

    def test_splits_and_serialization(self):
        tasks = build_tasks(80)
        validate_splits(tasks)
        self.assertEqual({task.split for task in tasks}, {"train", "validation", "test"})
        self.assertEqual([task.to_dict() for task in tasks], [task.to_dict() for task in build_tasks(80)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            write_jsonl(path, (task.to_dict() for task in tasks))
            self.assertEqual(read_tasks(path), tasks)
            with self.assertRaises(FileExistsError):
                write_jsonl(path, [])

    def test_split_leakage_and_duplicates_rejected(self):
        task = build_tasks(1)[0]
        with self.assertRaises(ValueError):
            validate_splits([task, task])
        other = copy.deepcopy(task)
        other.task_id += "-paraphrase"
        other.split = "test" if task.split != "test" else "train"
        with self.assertRaises(ValueError):
            validate_splits([task, other])

    def test_rl_row_keeps_hidden_state_outside_prompt(self):
        task = build_tasks(1)[0]
        row = rl_row(task, 0)
        self.assertEqual(json.loads(row["env_kwargs"])["task"], task.to_dict())
        self.assertEqual(row["agent_name"], "telecom_agent")
        self.assertNotIn("goal", row["prompt"][0]["content"])

    def test_tampered_expert_rejected(self):
        row = expert_trajectory(build_tasks(1)[0])
        row["steps"][0]["reward"] += 10
        with self.assertRaises(ValueError):
            validate_trajectory(row)

    @unittest.skipUnless(sys.platform == "darwin", "Mac-specific guard")
    def test_mac_guard_does_not_import_torch(self):
        with self.assertRaisesRegex(RuntimeError, "GPU"):
            require_cuda()
        self.assertNotIn("torch", sys.modules)


class FakeTokenizer:
    def apply_chat_template(self, messages, tools, add_generation_prompt, tokenize):
        if add_generation_prompt:
            return [10, 11, 12]
        return [10, 11, 12, 20, 2]


class SFTAndEvaluationTests(unittest.TestCase):
    def test_prompt_and_padding_excluded_eos_retained(self):
        row = {"messages": [], "action": "answer"}
        encoded = encode_step(FakeTokenizer(), row, 10)
        self.assertEqual(encoded["labels"], [-100, -100, -100, 20, 2])
        shorter = {key: value[:-1] for key, value in encoded.items()}
        batch = collate_steps([encoded, shorter], 2)
        self.assertEqual(batch["labels"][0][-1], 2)
        self.assertEqual(batch["labels"][1][-1], -100)

    def test_no_silent_completion_truncation(self):
        with self.assertRaises(ValueError):
            encode_step(FakeTokenizer(), {"messages": [], "action": "x"}, 4)

    def test_changed_template_prefix_rejected(self):
        class BadTokenizer(FakeTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                return [99] if kwargs["add_generation_prompt"] else [1, 2]
        with self.assertRaises(ValueError):
            encode_step(BadTokenizer(), {"messages": [], "action": "x"}, 10)

    def test_metrics_independent_of_reward(self):
        task = build_tasks(1)[0]
        actions = iter(step["action"] for step in expert_trajectory(task)["steps"])
        def policy(messages, tools):
            return {"text": next(actions), "input_tokens": 10, "output_tokens": 5}
        episodes = evaluate([task], policy, config())
        metrics = summarize(episodes)
        self.assertEqual(metrics["task_success_rate"], 1)
        self.assertEqual(metrics["average_tool_calls"], 2)
        self.assertEqual(metrics["token_cost_per_task"], 45)
        for step in episodes[0]["steps"]:
            step["reward"] = -100
        self.assertEqual(summarize(episodes), metrics)

    def test_metrics_include_invalid_calls(self):
        actions = iter(["<tool_call>{bad}</tool_call>", "无法完成"])
        policy = lambda *_: {"text": next(actions), "input_tokens": 2, "output_tokens": 1}
        episodes = evaluate([build_tasks(1)[0]], policy, config())
        metrics = summarize(episodes)
        self.assertEqual(metrics["invalid_tool_call_rate"], 1)
        self.assertEqual(metrics["task_success_rate"], 0)


if __name__ == "__main__":
    unittest.main()
