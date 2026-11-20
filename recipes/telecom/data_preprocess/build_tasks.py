"""Generate small synthetic scenarios. Counts are requested, never claimed results."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from recipes.telecom.data_preprocess.schema import Account, Task


def scenario_split(scenario_id):
    bucket = int(hashlib.sha256(scenario_id.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def build_tasks(count=40, seed=42):
    if count < 1:
        raise ValueError("count must be positive")
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    tasks = []
    for i in range(count):
        scenario = f"synthetic-{seed}-{i}"
        code = str(rng.randint(100000, 999999))
        amount = rng.randint(40, 300)
        month = f"2026-{i % 12 + 1:02d}"
        kind = ("bill", "change_plan", "repair", "deny_change")[i % 4]
        account = Account(f"S{seed}-{i}", code, bills={month: amount})
        if kind == "bill":
            goal = {"type": kind, "month": month}
            request = f"请查询{month}账单，告诉我月份和金额。"
        elif kind == "change_plan":
            confirmation = "我确认变更到plus套餐"
            goal = {"type": kind, "plan_id": "plus", "confirmation": confirmation}
            request = f"请查询可选套餐并办理变更。{confirmation}。"
        elif kind == "repair":
            issue, address = "宽带无法连接", f"模拟路{i + 1}号"
            goal = {"type": kind, "issue": issue, "address": address}
            request = f"请报修：{issue}，地址：{address}。"
        else:
            account.balance = -amount
            goal = {"type": kind}
            request = "请检查我能否变更套餐，不符合条件时不要办理并说明无法办理。"
        prompt = f"当前账户为{account.account_id}。{request}我的验证码是{code}。"
        tasks.append(Task(scenario, scenario, scenario_split(scenario), prompt, account, goal))
    return tasks


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse accidental data replacement.
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_tasks(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Task.from_dict(json.loads(line)) for line in lines if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_jsonl(args.output, (task.to_dict() for task in build_tasks(args.count, args.seed)))


if __name__ == "__main__":
    main()
