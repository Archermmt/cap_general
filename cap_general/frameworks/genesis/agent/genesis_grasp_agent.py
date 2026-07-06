"""Genesis grasp manipulation agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cap_general.frameworks.genesis.agent.genesis_base_agent import GenesisBaseAgent, GenesisBaseAgentConfig


@dataclass
class GenesisGraspAgentConfig(GenesisBaseAgentConfig):
    """Configuration for GenesisGraspAgent."""

    robot: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_grasp"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    rl_policy: str = "runner"
    bc_policy: str = "bc"
    stage: str = "bc"
    max_steps: int = 100
    run_demo_after_episode: bool = True


@GenesisBaseAgent.register()
class GenesisGraspAgent(GenesisBaseAgent):
    """Agent that evaluates Genesis Franka grasp policies."""

    agent_type = "genesis_grasp"
    config_cls = GenesisGraspAgentConfig
    train_best_metric = "mean_episode_rew_keypoints"

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {
            "grasp_episode": self.grasp_episode,
            "release_grasp": self.release_grasp,
        }

    def grasp_episode(self, stage: str | None = None, max_steps: int | None = None) -> dict[str, Any]:
        """Run one Genesis grasp episode with an RL or BC policy.

        Args:
            stage: ``'rl'`` for the RL runner policy, ``'bc'`` for the behavior-cloning policy.
                Defaults to the agent's configured stage.
            max_steps: Maximum simulation steps (default: agent max_steps).
        """
        current_stage = stage or self._config.stage
        steps = int(max_steps or self._config.max_steps)
        obs = self._robot.policy_obs
        for _ in range(steps):
            if current_stage == "rl":
                action = self._run_policy(self._config.rl_policy, inputs={"obs": obs})
            elif current_stage == "bc":
                rgb_obs = self._robot.get_stereo_rgb_images(normalize=True).float()
                ee_pose = self._robot.robot.ee_pose.float()
                action = self._run_policy(
                    self._config.bc_policy,
                    inputs={"env": self._robot, "rgb_obs": rgb_obs, "ee_pose": ee_pose},
                )
            else:
                raise ValueError(f"Unknown grasp stage: {current_stage!r}")
            obs, _reward, terminated, truncated, _info = self._robot.step(action)
            if terminated or truncated:
                break
            obs = self._robot.policy_obs

        if self._config.run_demo_after_episode:
            self._robot.grasp_and_lift_demo()
        return {"stage": current_stage}

    def release_grasp(self) -> dict[str, Any]:
        """Open the gripper and return to reset position after a grasp episode."""
        self._robot.release_grasp()
        return {"ok": True}
