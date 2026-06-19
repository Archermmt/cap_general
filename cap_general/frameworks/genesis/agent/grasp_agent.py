"""Genesis grasp manipulation agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from cap_general.core.agent import BaseAgent, BaseAgentConfig


@dataclass
class GraspAgentConfig(BaseAgentConfig):
    """Configuration for GraspAgent."""

    env: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_grasp"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    rl_policy: str = "runner"
    bc_policy: str = "bc"
    stage: str = "rl"
    horizon: int = 1000
    run_demo_after_episode: bool = True


@BaseAgent.register()
class GraspAgent(BaseAgent):
    """Agent that evaluates Genesis Franka grasp policies."""

    name = "Genesis Grasp Agent"
    config_cls = GraspAgentConfig

    def __init__(self, config: GraspAgentConfig):
        self._rl_policy_name = config.rl_policy
        self._bc_policy_name = config.bc_policy
        self._stage = config.stage
        self.horizon = int(config.horizon)
        self._run_demo_after_episode = bool(config.run_demo_after_episode)
        super().__init__(config=config)

    @classmethod
    def agent_type(cls) -> str:
        return "grasp_agent"

    def _execute_rules(self) -> str:
        return (
            "The Genesis grasp agent evaluates RL or BC policies in an env-controlled "
            "Franka grasp scene. Use grasp_episode(stage='rl'|'bc', max_steps=...). "
            "Do not create Genesis scenes, robots, cameras, or policies in generated code."
        )

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {"grasp_episode": self.grasp_episode, "grasp_and_lift_demo": self.grasp_and_lift_demo}

    def grasp_episode(self, stage: str | None = None, max_steps: int | None = None) -> dict[str, Any]:
        """Run one Genesis grasp episode with an RL or BC policy."""
        current_stage = stage or self._stage
        steps = int(max_steps or self.horizon)
        env = self._env.example_env
        if env is None:
            return {
                "steps": 0,
                "stage": current_stage,
                "obs": self._env.get_observation(self._record_dir / self.step_dir),
                "mock": True,
            }

        obs = self._env.policy_obs
        for _ in range(steps):
            if current_stage == "rl":
                action = self._run_policy(self._rl_policy_name, env=env, obs=obs)
            elif current_stage == "bc":
                rgb_obs = self._env.get_stereo_rgb_images(normalize=True).float()
                ee_pose = env.robot.ee_pose.float()
                action = self._run_policy(self._bc_policy_name, env=env, rgb_obs=rgb_obs, ee_pose=ee_pose)
            else:
                raise ValueError(f"Unknown grasp stage: {current_stage!r}")
            obs, _reward, terminated, truncated, _info = self._env.step(action)
            if terminated or truncated:
                break
            obs = self._env.policy_obs

        demo_ran = self.grasp_and_lift_demo() if self._run_demo_after_episode else False
        return {"stage": current_stage, "demo_ran": demo_ran}

    def grasp_and_lift_demo(self) -> bool:
        """Run the scripted grasp-and-lift demo from the underlying env."""
        return bool(self._env.grasp_and_lift_demo())
