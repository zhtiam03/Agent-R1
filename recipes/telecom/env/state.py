"""Synthetic business state. Never contains live customer data."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field


@dataclass
class Account:
    account_id: str
    identity_code: str
    plan_id: str = "basic"
    balance: int = 100
    data_remaining_mb: int = 1024
    status: str = "active"
    bills: dict = field(default_factory=dict)
    tickets: list = field(default_factory=list)


@dataclass
class Task:
    task_id: str
    scenario_id: str
    prompt: str
    goal: dict
    account: Account

    @classmethod
    def from_dict(cls, data):
        data = deepcopy(data)
        data["account"] = Account(**data["account"])
        task = cls(**data)
        if task.goal.get("type") not in {"bill", "change_plan", "repair", "deny_change"}:
            raise ValueError("Unknown task goal")
        if not task.account.account_id or not task.account.identity_code:
            raise ValueError("Account and identity code are required")
        if task.account.status not in {"active", "suspended"}:
            raise ValueError("Unknown account status")
        if task.goal["type"] == "bill" and task.goal.get("month") not in task.account.bills:
            raise ValueError("Bill goal references a missing month")
        return task

    def to_dict(self):
        return asdict(self)


def public_state(account, verified):
    state = asdict(account)
    state.pop("identity_code")
    state["verified"] = verified
    return state
