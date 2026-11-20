"""Task-level metrics independent of reward magnitude."""


def summarize(episodes):
    if not episodes:
        raise ValueError("Cannot evaluate an empty task set")
    n = len(episodes)
    attempts = sum(step["info"]["tool_attempts"] for row in episodes for step in row["steps"])
    invalid = sum(step["info"]["tool_attempts"] for row in episodes for step in row["steps"] if step["info"]["invalid"])
    successes = sum(bool(row["steps"]) and row["steps"][-1]["info"]["success"] for row in episodes)
    compliant = sum(not any(step["info"]["violation"] for step in row["steps"]) for row in episodes)
    return {
        "task_count": n,
        "task_success_rate": successes / n,
        "business_compliance_rate": compliant / n,
        "invalid_tool_call_rate": invalid / attempts if attempts else 0.0,
        "average_tool_calls": attempts / n,
        "token_cost_per_task": sum(row["input_tokens"] + row["output_tokens"] for row in episodes) / n,
        "generated_tokens_per_task": sum(row["output_tokens"] for row in episodes) / n,
        "truncated_tasks": sum(row.get("termination_reason") in {"max_steps", "prompt_length"} for row in episodes),
    }
