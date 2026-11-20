"""Agent-R1 boundary. Import only in a configured Agent-R1 server environment."""

from agent_r1.env.base import Action, AgentEnv, Observation
from agent_r1.tool import BaseTool, ToolResponse

from recipes.telecom.data_preprocess.schema import Task
from recipes.telecom.env.session import TelecomSession
from recipes.telecom.tools import TOOLS


class TelecomTool(BaseTool):
    def __init__(self):
        description, params = TOOLS[self.name]
        self.description = description
        self.parameters = {"type": "object", "properties": {
            key: {"type": "string", "description": value} for key, value in params.items()
        }, "required": list(params)}
        super().__init__()

    def call(self, args, simulator):
        """Synchronous business operation; also used by the async tool interface."""
        return simulator.execute(self.name, args)

    async def execute(self, args, **kwargs):
        result, milestones = self.call(args, kwargs["simulator"])
        import json

        return ToolResponse(text=json.dumps(result, ensure_ascii=False)), None, {"milestones": milestones}


# Register every declared tool; instances never hold global mutable account state.
for _name in TOOLS:
    _cls = type("Telecom_" + _name, (TelecomTool,), {"name": _name, "__module__": __name__})
    BaseTool.register(_name)(_cls)


@AgentEnv.register("telecom")
class TelecomEnv(AgentEnv):
    def __init__(self, max_steps=12, reward=None, **kwargs):
        self.max_steps = max_steps
        self.reward = reward
        self.tools = {name: BaseTool.from_name(name) for name in TOOLS}

    def reset(self, **kwargs):
        import json

        values = kwargs.get("env_kwargs", {})
        if isinstance(values, str):
            values = json.loads(values)
        self.session = TelecomSession(Task.from_dict(values["task"]), self.max_steps, self.reward)
        self.session.executor = lambda name, args: self.tools[name].call(args, self.session.simulator)
        return Observation(messages=self.session.messages.copy())

    @property
    def tool_schemas(self):
        return self.session.tool_schemas

    async def step(self, action: Action):
        messages, reward, done, info = self.session.step(action.text)
        return Observation(messages=messages), reward, done, info
