"""Conversation and transition orchestration, usable without ML dependencies."""

import json
import re
from copy import deepcopy

from recipes.telecom.env.simulator import BusinessError, Simulator
from recipes.telecom.prompts import initial_messages
from recipes.telecom.reward_fn import DEFAULT_REWARD, score_transition
from recipes.telecom.tools import tool_schemas

CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


class TelecomSession:
    def __init__(self, task, max_steps=12, reward=None):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.simulator = Simulator(task)
        self.executor = self.simulator.execute
        self.messages = initial_messages(task)
        self.max_steps = max_steps
        self.reward_config = {**DEFAULT_REWARD, **(reward or {})}
        if self.reward_config["mode"] not in {"state_aware", "outcome_only"}:
            raise ValueError("Unknown reward mode")
        self.trace = []
        self.done = False
        self.violations = 0
        self.milestones = set()

    @property
    def tool_schemas(self):
        return tool_schemas()

    def snapshot(self):
        account = self.simulator.account
        return {"account_id": account.account_id, "status": account.status, "plan_id": account.plan_id,
                "verified": self.simulator.verified, "ticket_count": len(account.tickets),
                "events": sorted(self.simulator.events)}

    def step(self, action):
        if self.done:
            raise RuntimeError("Trajectory is already terminal")
        if not isinstance(action, str):
            raise TypeError("Action must be text")
        before = self.snapshot()
        prompt = deepcopy(self.messages)
        blocks = CALL.findall(action)
        has_marker = "<tool_call" in action or "</tool_call" in action
        event = {"done": False, "success": False, "invalid": False, "violation": False,
                 "tool_attempts": max(len(blocks), 1) if has_marker else 0, "milestones": [],
                 "compliant_mutation": False, "termination_reason": None}
        result = None
        name = None
        if has_marker:
            try:
                if len(blocks) != 1 or action.count("<tool_call>") != 1 or action.count("</tool_call>") != 1:
                    raise ValueError("exactly_one_complete_tool_call_required")
                payload = json.loads(blocks[0])
                if (not isinstance(payload, dict) or set(payload) != {"name", "arguments"}
                        or not isinstance(payload["name"], str)):
                    raise ValueError("invalid_tool_call_shape")
                name = payload["name"]
                result, milestones = self.executor(name, payload["arguments"])
                event["milestones"] = sorted(set(milestones) - self.milestones)
                self.milestones.update(event["milestones"])
                event["compliant_mutation"] = bool(set(event["milestones"]) & {"plan_changed", "repair_created"})
            except BusinessError as exc:
                event["violation"] = exc.violation
                result = {"error": str(exc)}
            except (ValueError, TypeError, KeyError) as exc:
                event["invalid"] = True
                result = {"error": str(exc)}
        elif not action.strip():
            event["invalid"] = True
            result = {"error": "empty_response"}
        else:
            event["done"] = True
            event["success"] = self.simulator.success(action) and self.violations == 0
            event["termination_reason"] = "completed" if event["success"] else "unsatisfied_final_answer"
        self.violations += int(event["violation"])
        if not event["done"] and len(self.trace) + 1 >= self.max_steps:
            event.update(done=True, success=False, termination_reason="max_steps")
        reward, parts = score_transition(event, self.reward_config)
        self.done = event["done"]
        self.messages.append({"role": "assistant", "content": action})
        if result is not None:
            text = json.dumps(result, ensure_ascii=False, sort_keys=True)
            self.messages.append({"role": "user", "content": f"<tool_response>\n{text}\n</tool_response>"})
        info = {**event, "step_reward": reward, "reward_components": parts,
                "tool_name": name, "state_before": before, "state_after": self.snapshot()}
        self.trace.append({"step_index": len(self.trace), "messages": prompt, "action": action,
                           "observation": deepcopy(self.messages), "reward": reward, "info": deepcopy(info)})
        return deepcopy(self.messages), reward, self.done, info

    def truncate(self, reason):
        """Settle a context-budget truncation on the last real action, exactly once."""
        if self.done:
            return
        if not self.trace:
            raise ValueError("Initial prompt exceeds context budget; no trainable action exists")
        last = self.trace[-1]
        last["info"].update(done=True, success=False, termination_reason=reason)
        reward, parts = score_transition(last["info"], self.reward_config)
        last["reward"] = reward
        last["info"].update(step_reward=reward, reward_components=parts)
        self.done = True
