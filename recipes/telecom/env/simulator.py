"""Deterministic business rules; no Agent-R1, torch, YAML, or network imports."""

from copy import deepcopy

from recipes.telecom.tools import validate_call

PLANS = {
    "basic": {"name": "基础套餐", "monthly_fee": 39},
    "plus": {"name": "畅享套餐", "monthly_fee": 79},
}


class BusinessError(ValueError):
    def __init__(self, reason, violation=False):
        super().__init__(reason)
        self.violation = violation


class Simulator:
    def __init__(self, task):
        self.task = deepcopy(task)
        self.account = self.task.account
        self.verified = False
        self.verification_failures = 0
        self.events = set()

    def execute(self, name, args):
        validate_call(name, args)
        account = self.account
        if name == "get_account_info":
            return {"account_id": account.account_id, "status": account.status}, []
        if name == "verify_identity":
            if self.verification_failures >= 3:
                raise BusinessError("verification_locked", violation=True)
            if args["code"] != account.identity_code:
                self.verification_failures += 1
                raise BusinessError("verification_failed")
            first = not self.verified
            self.verified = True
            return {"verified": True}, ["verified"] if first else []
        if not self.verified:
            raise BusinessError("identity_required", violation=True)
        if name == "get_bill_details":
            if args["month"] not in account.bills:
                raise BusinessError("bill_not_found")
            key = "bill:" + args["month"]
            first = key not in self.events
            self.events.add(key)
            return {"month": args["month"], "amount": account.bills[args["month"]]}, [key] if first else []
        if name == "get_available_plans":
            eligible = account.status == "active" and account.balance >= 0
            self.events.add("plans_viewed")
            return {"plans": deepcopy(PLANS), "eligible": eligible}, []
        if name == "change_plan":
            if account.status != "active" or account.balance < 0:
                raise BusinessError("account_ineligible", violation=True)
            if args["plan_id"] not in PLANS:
                raise BusinessError("unknown_plan")
            # A model cannot forge confirmation: compare against hidden task consent.
            consent = self.task.goal.get("confirmation")
            if not consent or args["confirmation"] != consent or args["plan_id"] != self.task.goal.get("plan_id"):
                raise BusinessError("customer_consent_required", violation=True)
            changed = account.plan_id != args["plan_id"]
            account.plan_id = args["plan_id"]
            self.events.add("plan_changed")
            return {"plan_id": account.plan_id, "changed": changed}, ["plan_changed"] if changed else []
        if name == "create_repair_ticket":
            goal = self.task.goal
            if goal.get("type") != "repair" or args != {"issue": goal.get("issue"), "address": goal.get("address")}:
                raise BusinessError("customer_repair_details_required", violation=True)
            existing = next((t for t in account.tickets
                             if t["issue"] == args["issue"] and t["address"] == args["address"]), None)
            if existing:
                return deepcopy(existing), []
            ticket = {"ticket_id": f"T-{account.account_id}-{len(account.tickets) + 1}", **args, "status": "open"}
            account.tickets.append(ticket)
            self.events.add("repair_created")
            return deepcopy(ticket), ["repair_created"]
        raise ValueError("unknown_tool")

    def success(self, final_text):
        goal = self.task.goal
        if goal["type"] == "bill":
            return ("bill:" + goal["month"] in self.events and goal["month"] in final_text
                    and str(self.account.bills[goal["month"]]) in final_text)
        if goal["type"] == "change_plan":
            return ("plan_changed" in self.events and self.account.plan_id == goal["plan_id"]
                    and goal["plan_id"] in final_text)
        if goal["type"] == "repair":
            return "repair_created" in self.events and any(t["ticket_id"] in final_text for t in self.account.tickets)
        return (
            "plans_viewed" in self.events and (self.account.status != "active" or self.account.balance < 0)
            and "plan_changed" not in self.events and any(word in final_text for word in ("无法", "不能", "不符合"))
        )
