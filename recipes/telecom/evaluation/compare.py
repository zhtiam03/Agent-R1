"""Compare held-out runs; refuse mismatched task sets and evaluation budgets."""

import argparse
import json
from pathlib import Path


def compare(reports):
    labels = [row["label"] for row in reports]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate experiment labels")
    reference = reports[0]
    for row in reports:
        if row["task_digest"] != reference["task_digest"]:
            raise ValueError("Evaluation task content differs")
        if row["task_ids"] != reference["task_ids"]:
            raise ValueError("Evaluation task IDs/order differ")
        for key in ("environment",):
            if row["config"][key] != reference["config"][key]:
                raise ValueError(f"Evaluation {key} differs")
        for key in ("max_new_tokens", "temperature", "seed"):
            if row["config"]["inference"][key] != reference["config"]["inference"][key]:
                raise ValueError(f"Inference budget/sampling differs: {key}")
        if row["config"]["grpo"]["prompt_length"] != reference["config"]["grpo"]["prompt_length"]:
            raise ValueError("Prompt budget differs")
    return {row["label"]: row["metrics"] for row in reports}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", help="Paths to *.metrics.json")
    args = parser.parse_args()
    print(json.dumps(compare([json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
