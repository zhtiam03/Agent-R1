"""Dependency-free public tool specifications (execution is owned by the simulator)."""

TOOLS = {
    "get_account_info": ("查询当前模拟账户的基础信息。", {}),
    "verify_identity": ("使用用户提供的验证码验证当前账户。", {"code": "用户提供的验证码"}),
    "get_bill_details": ("查询指定月份账单，需要身份验证。", {"month": "账期，例如2026-01"}),
    "get_available_plans": ("查询可选套餐及当前账户的办理资格。", {}),
    "change_plan": ("变更套餐，需要验证身份及用户确认。", {"plan_id": "套餐ID", "confirmation": "用户的确认原文"}),
    "create_repair_ticket": (
        "创建故障工单，需要身份验证。",
        {"issue": "故障描述", "address": "服务地址"},
    ),
}


def tool_schemas():
    return [
        {"type": "function", "function": {"name": name, "description": description, "parameters": {
            "type": "object",
            "properties": {key: {"type": "string", "description": text} for key, text in params.items()},
            "required": list(params), "additionalProperties": False,
        }}}
        for name, (description, params) in TOOLS.items()
    ]


def validate_call(name, arguments):
    if name not in TOOLS:
        raise ValueError("unknown_tool")
    if not isinstance(arguments, dict):
        raise ValueError("arguments_must_be_object")
    required = TOOLS[name][1]
    if set(arguments) != set(required):
        raise ValueError("incorrect_arguments")
    if any(not isinstance(v, str) or not v.strip() for v in arguments.values()):
        raise ValueError("arguments_must_be_nonempty_strings")
