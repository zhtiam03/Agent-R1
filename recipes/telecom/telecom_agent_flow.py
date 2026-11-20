"""Step-native rollout adapter preserving business rewards and audit metadata."""

from time import perf_counter
from uuid import uuid4

from agent_r1.agent_flow.agent_env_loop import AgentEnvLoop
from agent_r1.agent_flow.agent_flow import AgentFlowOutput, AgentFlowStep, register
from agent_r1.env.base import Action

from recipes.telecom import adapters as _adapters  # noqa: F401: registration per Ray worker


@register("telecom_agent")
class TelecomAgentFlow(AgentEnvLoop):
    async def run(self, sampling_params, **kwargs):
        env = self._create_env(**kwargs)
        obs = env.reset(**kwargs)
        raw_steps = []
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0}
        for _ in range(self.max_steps):
            prompt_ids = await self._obs_to_prompt(obs, tools=env.tool_schemas)
            if len(prompt_ids) > self.prompt_length:
                env.session.truncate("prompt_length")
                break
            start = perf_counter()
            output = await self.server_manager.generate(
                request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=sampling_params,
            )
            metrics["generate_sequences"] += perf_counter() - start
            response_ids = output.token_ids[:self.response_length]
            if not response_ids:
                raise ValueError("Empty generation cannot form a trainable AgentFlowStep")
            text = self.tokenizer.decode(response_ids, skip_special_tokens=self.skip_special_tokens)
            start = perf_counter()
            obs, reward, done, info = await env.step(Action(text=text, token_ids=response_ids))
            metrics["tool_calls"] += perf_counter() - start
            raw_steps.append(AgentFlowStep(
                prompt_ids=prompt_ids, response_ids=response_ids, reward_score=reward,
                response_logprobs=output.log_probs[:self.response_length] if output.log_probs else None,
            ))
            if done:
                break
        if not env.session.done:
            env.session.truncate("max_steps")
        steps = []
        # Postprocess after termination settlement so the final reward is never stale.
        for step, transition in zip(raw_steps, env.session.trace, strict=True):
            step.reward_score = transition["reward"]
            info = transition["info"]
            step.extra_fields = {"telecom_transition": transition, "reward_extra_info": {
                "step_reward": transition["reward"], "task_success": info["success"],
                "violation": info["violation"], "invalid_tool_call": info["invalid"],
                "tool_attempts": info["tool_attempts"], "termination_reason": info["termination_reason"],
            }}
            steps.append(await self._postprocess(step, **kwargs))
        return AgentFlowOutput(steps=steps, metrics=metrics)
