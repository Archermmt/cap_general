"""LIBERO VLA agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from cap_general.core.agent import BaseAgent, BaseAgentConfig
from cap_general.frameworks.libero.robot.libero_robot import _binarize_gripper_open, build_example_from_obs


@dataclass
class LiberoAgentConfig(BaseAgentConfig):
    """Configuration for LiberoAgent."""

    robot: dict[str, Any] = field(default_factory=lambda: {"type": "libero_robot"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)


@BaseAgent.register()
class LiberoAgent(BaseAgent):
    """Agent that runs LIBERO subtasks with a configured VLA policy."""

    name = "LIBERO VLA Agent"
    config_cls = LiberoAgentConfig

    @classmethod
    def agent_type(cls) -> str:
        return "libero"

    def _execute_rules(self) -> str:
        """Return valid rules for execute for the loaded LIBERO suite."""
        task_suite = getattr(self._robot, "_task_suite", None)
        if task_suite is None:
            return ""

        suite_name = getattr(self._robot, "_task_suite_name", "unknown")
        tasks = [task_suite.get_task(i).language for i in range(task_suite.get_num_tasks())]
        task_list = "\n".join(f"  - {task}" for task in tasks)
        return (
            f"Task suite: {suite_name}\n"
            "Available task descriptions (pass verbatim to libero_vla_episode):\n"
            f"{task_list}\n"
            "IMPORTANT: libero_vla_episode(task=...) only accepts the exact strings "
            "listed above. Map the user's goal to one or more of these descriptions "
            "when decomposing subtasks."
        )

    def _options_doc(self, method_name: str) -> str:
        """Return LIBERO-specific options docs for supported methods."""
        if method_name != "reset":
            return super()._options_doc(method_name)
        base_doc = super()._options_doc(method_name)
        extra_doc = (
            "episode_idx: LIBERO initial state index to load when reset_level is "
            "1 or 2. Defaults to 0."
        )
        return f"{base_doc}\n{extra_doc}" if base_doc else extra_doc

    def functions(self) -> dict[str, Callable[..., Any]]:
        """Return LIBERO functions exposed to generated code."""
        return {"libero_vla_episode": self.libero_vla_episode}

    def libero_vla_episode(self, task: str, max_steps: int = 300, policy_name: str = "starvla") -> bool:
        """Run a full LIBERO episode using the configured VLA policy.

        Args:
            task: Task description string. It must match one task language in the
                current LIBERO suite.
            max_steps: Maximum number of robot control steps.
            vla_policy: Name of the configured VLA policy to run.

        Returns:
            True when the LIBERO success predicate is reached.
        """
        self._robot.set_task_goal(task)
        obs, done = self._robot.last_obs, False
        for step_idx in range(max_steps):
            example = build_example_from_obs(obs, task)
            response = self._run_policy(policy_name, example=example, step=step_idx)
            raw = response.get("raw_action", response)
            action = np.concatenate(
                [
                    np.asarray(raw["world_vector"], dtype=np.float32).reshape(-1)[:3],
                    np.asarray(raw["rotation_delta"], dtype=np.float32).reshape(-1)[:3],
                    _binarize_gripper_open(raw["open_gripper"]),
                ]
            )
            obs, _reward, terminated, truncated, _info = self._robot.step(action.tolist())
            done = bool(terminated or truncated)
            if done:
                break
        return done
