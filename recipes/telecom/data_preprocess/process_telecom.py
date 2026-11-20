"""Validated split export: SFT JSONL and RL Parquet (server-only dependency)."""

import argparse
import json
from pathlib import Path

from recipes.telecom.data_preprocess.build_expert_trajectories import expert_trajectory, validate_trajectory
from recipes.telecom.data_preprocess.build_tasks import read_tasks, write_jsonl
from recipes.telecom.prompts import initial_messages


def validate_splits(tasks):
    ids, scenarios, accounts = set(), {}, {}
    for task in tasks:
        if task.task_id in ids:
            raise ValueError(f"Duplicate task_id: {task.task_id}")
        ids.add(task.task_id)
        if task.scenario_id in scenarios and scenarios[task.scenario_id] != task.split:
            raise ValueError("Scenario crosses dataset splits")
        scenarios[task.scenario_id] = task.split
        if task.account.account_id in accounts and accounts[task.account.account_id] != task.split:
            raise ValueError("Account crosses dataset splits")
        accounts[task.account.account_id] = task.split


def rl_row(task, index):
    return {"data_source": "telecom", "agent_name": "telecom_agent", "prompt": initial_messages(task),
            "reward_model": {"style": "rule", "ground_truth": json.dumps(task.goal, ensure_ascii=False)},
            "extra_info": {"index": index, "task_id": task.task_id, "split": task.split},
            "env_kwargs": json.dumps({"task": task.to_dict()}, ensure_ascii=False)}


def sft_rows(trajectory):
    validate_trajectory(trajectory)
    return [{"task_id": trajectory["task"]["task_id"], "split": trajectory["task"]["split"],
             "messages": step["messages"], "action": step["action"]} for step in trajectory["steps"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parquet", action="store_true", help="Requires pyarrow on the server; no auto-install")
    args = parser.parse_args()
    tasks = read_tasks(args.tasks)
    validate_splits(tasks)
    splits = ("train", "validation", "test")
    target = Path(args.output_dir)
    for split in splits:
        if not any(task.split == split for task in tasks):
            raise ValueError(f"No tasks in {split}; generate more scenarios or supply curated splits")
        suffixes = ("tasks.jsonl", "rl.jsonl", "expert.jsonl", "sft.jsonl") + (("parquet",) if args.parquet else ())
        if any((target / f"{split}.{suffix}").exists() for suffix in suffixes):
            raise FileExistsError(f"Refusing to overwrite existing {split} artifacts")
    if args.parquet:
        import pyarrow as pa
        import pyarrow.parquet as pq

    for split in ("train", "validation", "test"):
        selected = [task for task in tasks if task.split == split]
        if not selected:
            raise ValueError(f"No tasks in {split}; generate more scenarios or supply curated splits")
        rows = [rl_row(task, i) for i, task in enumerate(selected)]
        trajectories = [expert_trajectory(task) for task in selected]
        target = Path(args.output_dir)
        write_jsonl(target / f"{split}.tasks.jsonl", (task.to_dict() for task in selected))
        write_jsonl(target / f"{split}.rl.jsonl", rows)
        write_jsonl(target / f"{split}.expert.jsonl", trajectories)
        write_jsonl(target / f"{split}.sft.jsonl", (row for traj in trajectories for row in sft_rows(traj)))
        if args.parquet:
            path = target / f"{split}.parquet"
            if path.exists():
                raise FileExistsError(path)
            pq.write_table(pa.Table.from_pylist(rows), path)


if __name__ == "__main__":
    main()
