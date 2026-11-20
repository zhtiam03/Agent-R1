"""Rule-based expert that interacts with the same simulator used for evaluation."""

import argparse
import json

from recipes.telecom.data_preprocess.build_tasks import read_tasks, write_jsonl
from recipes.telecom.env.session import TelecomSession


def tool_action(name, **arguments):
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "</tool_call>"


def expert_trajectory(task):
    session = TelecomSession(task)
    session.step(tool_action("verify_identity", code=task.account.identity_code))
    goal = task.goal
    if goal["type"] == "bill":
        session.step(tool_action("get_bill_details", month=goal["month"]))
        final = f"{goal['month']}账单金额为{task.account.bills[goal['month']]}元。"
    elif goal["type"] == "change_plan":
        session.step(tool_action("get_available_plans"))
        session.step(tool_action("change_plan", plan_id=goal["plan_id"], confirmation=goal["confirmation"]))
        final = f"已变更到{goal['plan_id']}套餐。"
    elif goal["type"] == "repair":
        session.step(tool_action("create_repair_ticket", issue=goal["issue"], address=goal["address"]))
        final = f"已创建报修工单{session.simulator.account.tickets[-1]['ticket_id']}。"
    else:
        session.step(tool_action("get_available_plans"))
        final = "账户当前不符合条件，无法办理套餐变更。"
    session.step(final)
    if not session.trace[-1]["info"]["success"]:
        raise ValueError(f"Expert did not satisfy task {task.task_id}")
    return {"task": task.to_dict(), "steps": session.trace, "source": "synthetic_rule_expert_v1"}


def validate_trajectory(row):
    from recipes.telecom.data_preprocess.schema import Task

    task = Task.from_dict(row["task"])
    session = TelecomSession(task)
    for saved in row["steps"]:
        session.step(saved["action"])
        if session.trace[-1] != saved:
            raise ValueError(f"Replay mismatch in {task.task_id}")
    if not session.done or not session.trace[-1]["info"]["success"]:
        raise ValueError("Expert trajectory must terminate successfully")
    return session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_jsonl(args.output, (expert_trajectory(task) for task in read_tasks(args.tasks)))


if __name__ == "__main__":
    main()
