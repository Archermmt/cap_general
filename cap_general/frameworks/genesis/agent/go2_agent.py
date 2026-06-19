"""Genesis GO2 locomotion agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from cap_general.core.agent import BaseAgent, BaseAgentConfig


@dataclass
class Go2AgentConfig(BaseAgentConfig):
    """Configuration for Go2Agent."""

    env: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_go2"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy: str = "runner"
    horizon: int = 1000


@BaseAgent.register()
class Go2Agent(BaseAgent):
    """Agent that evaluates Genesis GO2 locomotion policies."""

    name = "Genesis GO2 Agent"
    config_cls = Go2AgentConfig

    def __init__(self, config: Go2AgentConfig):
        self._policy_name = config.policy
        self.horizon = int(config.horizon)
        super().__init__(config=config)

    @classmethod
    def agent_type(cls) -> str:
        return "go2_agent"

    def _execute_rules(self) -> str:
        return (
            "The Genesis GO2 agent evaluates locomotion policies in an env-controlled scene. "
            "Use walk_forward(max_steps=...) to make GO2 walk forward. Do not create "
            "Genesis scenes or robots in generated code."
        )

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {"walk_forward": self.walk_forward}

    def walk_forward(self, max_steps: int | None = None) -> dict[str, Any]:
        """Make GO2 walk forward for up to max_steps policy steps."""
        steps = int(max_steps or self.horizon)
        env = self._env.example_env
        if env is None:
            return {"steps": 0, "obs": self._env.get_observation(self._record_dir / self.step_dir), "mock": True}

        obs = self._env.policy_obs
        print("[TMINFO] reset obs " + str(obs), flush=True)
        for step_idx in range(steps):
            action = self._run_policy(self._policy_name, env=env, obs=obs)
            print(f"[TMINFO] {step_idx}/{steps} th action {action}", flush=True)
            obs, _reward, terminated, truncated, _info = self._env.step(action)
            if terminated or truncated:
                break
            obs = self._env.policy_obs
        return {
            "steps": step_idx + 1 if steps > 0 else 0,
            "obs": self._env.get_observation(self._record_dir / self.step_dir),
        }
