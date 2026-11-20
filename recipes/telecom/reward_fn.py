"""Single reward authority, called once per transition by TelecomSession."""

DEFAULT_REWARD = {
    "mode": "state_aware", "task": 1.0, "failure": 1.0, "progress": 0.1,
    "compliance": 0.1, "tool_cost": 0.02, "invalid": 0.1, "violation": 1.0,
}


def score_transition(event, config):
    parts = {"task": 0.0, "progress": 0.0, "compliance": 0.0, "tool_cost": 0.0, "violation": 0.0, "invalid": 0.0}
    if event["done"]:
        parts["task"] = config["task"] if event["success"] else -config["failure"]
    if config["mode"] == "state_aware":
        parts["progress"] = config["progress"] * len(event.get("milestones", []))
        parts["compliance"] = config["compliance"] * int(event.get("compliant_mutation", False))
        parts["tool_cost"] = -config["tool_cost"] * event.get("tool_attempts", 0)
        parts["violation"] = -config["violation"] * int(event.get("violation", False))
        parts["invalid"] = -config["invalid"] * int(event.get("invalid", False))
    return sum(parts.values()), parts


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Compatibility callback: fail closed rather than invent text-only state rewards.

    Production rollout supplies numeric rewards. If this callback is reached,
    a reward manager must have explicitly preserved the transition breakdown.
    """
    info = (extra_info or {}).get("reward_extra_info", {})
    if "step_reward" not in info:
        raise ValueError("Telecom state reward requires step_reward metadata; text-only fallback is forbidden")
    return {"score": float(info["step_reward"])}
