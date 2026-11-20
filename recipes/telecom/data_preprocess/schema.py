"""Versioned, serializable synthetic task contract."""

from dataclasses import asdict, dataclass, field


@dataclass
class Account:
    account_id: str
    identity_code: str
    status: str = "active"
    plan_id: str = "basic"
    balance: int = 100
    bills: dict = field(default_factory=dict)
    tickets: list = field(default_factory=list)


@dataclass
class Task:
    task_id: str
    scenario_id: str
    split: str
    prompt: str
    account: Account
    goal: dict
    schema_version: int = 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["account"] = Account(**data["account"])
        task = cls(**data)
        if task.schema_version != 1 or task.split not in {"train", "validation", "test"}:
            raise ValueError("Unsupported task schema/split")
        if task.goal.get("type") not in {"bill", "change_plan", "repair", "deny_change"}:
            raise ValueError("Unknown goal type")
        return task
