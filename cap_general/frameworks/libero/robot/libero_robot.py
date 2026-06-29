"""LIBERO environment controller."""

from __future__ import annotations

import logging
import math
import os
import sys
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np

from cap_general.core.robot import BaseRobot, BaseRobotConfig, ResetLevel

_DEFAULT_RESOLUTION = 256
_DEFAULT_IMAGE_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def build_example_from_obs(raw_obs: dict, task_description: str) -> dict:
    """Build the example dict expected by LocalStarVLA.step() from a LIBERO obs."""
    img = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {"image": [img, wrist], "lang": task_description}


def _quat2axisangle(quat: Any) -> Any:
    """Convert a xyzw quaternion to an axis-angle vector."""
    import numpy as np

    quat_arr = np.array(quat, dtype=np.float64)
    if quat_arr[3] > 1.0:
        quat_arr[3] = 1.0
    elif quat_arr[3] < -1.0:
        quat_arr[3] = -1.0

    den = math.sqrt(1.0 - quat_arr[3] * quat_arr[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat_arr[:3] * 2.0 * math.acos(quat_arr[3])) / den


def _binarize_gripper_open(val: Any) -> Any:
    """Convert a LIBERO gripper value to StarVLA-style open/close signal."""
    import numpy as np

    v = float(np.asarray(val).reshape(-1)[0])
    return np.asarray([1.0 - 2.0 * (v > 0.5)], dtype=np.float32)


def _coerce_env_reset_level(value: Any) -> ResetLevel:
    """Map agent-level reset scopes onto environment-level reset scopes."""
    raw_level = ResetLevel.AGENT if value is None else value
    level_value = int(raw_level)
    if level_value <= int(ResetLevel.ROBOT):
        return ResetLevel.ROBOT
    return ResetLevel.AGENT


def _is_libero_missing_datasets_warning(line: str) -> bool:
    return line.startswith("[Warning]: datasets path ") and line.endswith(" does not exist!")


def _call_without_libero_dataset_warning(func, *args, **kwargs):
    buffer = StringIO()
    with redirect_stdout(buffer):
        result = func(*args, **kwargs)
    for line in buffer.getvalue().splitlines():
        if not _is_libero_missing_datasets_warning(line):
            print(line)
    return result


@contextmanager
def _libero_init_state_torch_load_compat():
    """Allow trusted LIBERO init-state files to load on PyTorch 2.6+."""
    try:
        import torch
    except ImportError:
        yield
        return

    original_load = torch.load

    def _load_with_legacy_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _load_with_legacy_default
    try:
        yield
    finally:
        torch.load = original_load


@dataclass
class LiberoRobotConfig(BaseRobotConfig):
    """Configuration for LiberoRobot."""

    task_suite_name: str = "libero_goal"
    task_id: int = 0
    seed: int = 7
    resolution: int = _DEFAULT_RESOLUTION
    libero_home: str | None = None
    image_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_IMAGE_KEYS))
    reset_settle_steps: int = 10


@BaseRobot.register()
class LiberoRobot(BaseRobot):
    """Gymnasium-style wrapper around LIBERO OffScreenRenderEnv."""

    name = "LIBERO Robot"
    config_cls = LiberoRobotConfig
    dummy_action = _DUMMY_ACTION

    def __init__(
        self,
        config: LiberoRobotConfig,
        logger: logging.Logger,
    ):
        super().__init__(config=config, logger=logger)
        self._task_suite_name = config.task_suite_name
        self._task_id = int(config.task_id)
        self._resolution = int(config.resolution)
        self._libero_home = self._resolve_libero_home(config.libero_home)
        self._reset_settle_steps = int(config.reset_settle_steps)
        self._task_env = None
        self._task_suite = None
        self._task = None
        self._initial_states = []
        self._last_reward = 0.0
        self.task_description = ""

        self._init_libero_robot()

    @classmethod
    def robot_type(cls) -> str:
        return "libero_robot"

    @staticmethod
    def _resolve_libero_home(configured_home: str | None) -> str:
        return configured_home or os.environ.get("LIBERO_HOME")

    def _init_libero_robot(self) -> None:
        if self._libero_home not in sys.path:
            sys.path.insert(0, self._libero_home)
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path(self._libero_home) / "libero"))

        try:
            from gymnasium import spaces
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as exc:
            robosuite_hint = ""
            try:
                import robosuite

                robosuite_hint = (
                    f" Detected robosuite=={getattr(robosuite, '__version__', 'unknown')}; "
                    "LIBERO expects robosuite==1.4.0."
                )
            except ImportError:
                robosuite_hint = " robosuite is not importable."
            missing_module = getattr(exc, "name", None)
            missing_hint = f" Missing module: {missing_module!r}." if missing_module else ""
            raise ImportError(
                "LiberoRobot requires gymnasium and LIBERO to be importable. "
                "Set libero_home or LIBERO_HOME to the LIBERO repository root."
                f"{missing_hint}{robosuite_hint} Install the LIBERO extra with: "
                'pip install -e ".[libero]".'
            ) from exc

        benchmark_dict = benchmark.get_benchmark_dict()
        if self._task_suite_name not in benchmark_dict:
            known = ", ".join(sorted(benchmark_dict))
            raise KeyError(f"Unknown LIBERO task suite {self._task_suite_name!r}. Known: {known}")

        self._task_suite = benchmark_dict[self._task_suite_name]()
        self._set_task_metadata(self._task_id)

        bddl_path = _call_without_libero_dataset_warning(get_libero_path, "bddl_files")
        task_bddl_file = Path(bddl_path) / self._task.problem_folder / self._task.bddl_file
        self._task_env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_heights=self._resolution,
            camera_widths=self._resolution,
        )
        self._task_env.seed(self._config.seed)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype="float32")
        img_space = spaces.Box(
            low=0,
            high=255,
            shape=(self._resolution, self._resolution, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                "agentview_image": img_space,
                "robot0_eye_in_hand_image": img_space,
                "robot0_eef_pos": spaces.Box(-float("inf"), float("inf"), (3,), dtype="float32"),
                "robot0_eef_quat": spaces.Box(-1.0, 1.0, (4,), dtype="float32"),
                "robot0_gripper_qpos": spaces.Box(-1.0, 1.0, (2,), dtype="float32"),
            }
        )

    def _set_task_metadata(self, task_id: int) -> None:
        self._task_id = int(task_id)
        self._task = self._task_suite.get_task(self._task_id)
        with _libero_init_state_torch_load_compat():
            self._initial_states = _call_without_libero_dataset_warning(
                self._task_suite.get_task_init_states,
                self._task_id,
            )
        self.task_description = self._task.language

    def _reset(
        self,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the LIBERO episode and return the initial observation."""
        options = options or {}
        reset_level = _coerce_env_reset_level(options.get("reset_level"))
        info = {
            "seed": self._config.seed,
            "options": options,
            "task_description": self.task_description,
            "task_suite_name": self._task_suite_name,
            "task_id": self._task_id,
        }
        if reset_level is ResetLevel.ROBOT:
            return self._reset_robot(), info

        episode_idx = int(options.get("episode_idx", 0))
        self._task_env.reset()
        obs = self._task_env.set_init_state(self._initial_states[episode_idx])
        for _ in range(self._reset_settle_steps):
            obs, _, done, _ = self._task_env.step(_DUMMY_ACTION)
            if done:
                break
        return obs, info

    def _reset_robot(self) -> dict[str, Any]:
        """Restore robot pose without changing object placements."""
        problem = self._task_env.env
        sim = problem.sim
        for robot in problem.robots:
            robot.reset(deterministic=True)
            sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
            if robot.has_gripper:
                sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = robot.gripper.init_qpos
                sim.data.qvel[robot._ref_gripper_joint_vel_indexes] = 0.0
        sim.forward()
        problem.timestep = 0
        problem.done = False
        return self._task_env.env._get_observations()

    def _step(self, action: Any) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        """Apply a 7D LIBERO action and return a Gymnasium step tuple."""
        libero_action = self._coerce_action(action)
        obs, reward, done, info = self._task_env.step(libero_action)
        self._last_reward = float(reward)
        return obs, 0.0, bool(done), False, info

    def compute_reward(self) -> SupportsFloat:
        """Return the most recent LIBERO reward."""
        return self._last_reward

    def _coerce_action(self, action: Any) -> list[float]:
        if action is None:
            return list(self.dummy_action)
        if isinstance(action, dict):
            if "action" in action:
                return self._coerce_action(action["action"])
            if "raw_action" in action:
                return self._coerce_action(action["raw_action"])
            if {"world_vector", "rotation_delta", "open_gripper"}.issubset(action):
                return self._flatten_action_parts(
                    action["world_vector"],
                    action["rotation_delta"],
                    action["open_gripper"],
                )
        if hasattr(action, "tolist"):
            return list(action.tolist())
        return list(action)

    @staticmethod
    def _flatten_action_parts(
        world_vector: Any,
        rotation_delta: Any,
        open_gripper: Any,
    ) -> list[float]:
        import numpy as np

        return [
            *np.asarray(world_vector, dtype=float).reshape(-1)[:3].tolist(),
            *np.asarray(rotation_delta, dtype=float).reshape(-1)[:3].tolist(),
            *np.asarray(open_gripper, dtype=float).reshape(-1)[:1].tolist(),
        ]

    def _normalize_states(self) -> dict[str, Any]:
        if not isinstance(self._last_obs, dict):
            return {}
        state_keys = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_joint_pos",
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
            "robot0_joint_vel",
        ]
        states = {key: self._last_obs[key] for key in state_keys if key in self._last_obs}
        if "robot0_eef_quat" in states:
            states["robot0_eef_axis_angle"] = _quat2axisangle(states["robot0_eef_quat"])
        if "robot0_gripper_qpos" in states:
            states["robot0_gripper_open"] = _binarize_gripper_open(states["robot0_gripper_qpos"])
        return states

    def _record_frame(self, obs: Any) -> None:
        if isinstance(obs, dict) and "agentview_image" in obs:
            obs = {**obs, "agentview_image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])}
        super()._record_frame(obs)

    def get_observation(self, folder) -> dict:
        orig = self._last_obs
        if isinstance(orig, dict) and "agentview_image" in orig:
            self._last_obs = {**orig, "agentview_image": np.ascontiguousarray(orig["agentview_image"][::-1, ::-1])}
        try:
            return super().get_observation(folder)
        finally:
            self._last_obs = orig

    def set_task_goal(self, task: int | str) -> None:
        """Hot-swap LIBERO success predicates without resetting physics state."""
        from libero.libero import get_libero_path
        from libero.libero.envs import bddl_utils as BDDLUtils

        task_id = self._task_id_from_language(task) if isinstance(task, str) else int(task)
        new_task = self._task_suite.get_task(task_id)
        bddl_path = _call_without_libero_dataset_warning(get_libero_path, "bddl_files")
        new_bddl = Path(bddl_path) / new_task.problem_folder / new_task.bddl_file
        new_parsed = BDDLUtils.robosuite_parse_problem(str(new_bddl))

        problem = self._task_env.env
        problem.parsed_problem["goal_state"] = new_parsed["goal_state"]
        problem.parsed_problem["obj_of_interest"] = new_parsed["obj_of_interest"]
        problem.parsed_problem["language_instruction"] = new_parsed["language_instruction"]
        problem.obj_of_interest = new_parsed["obj_of_interest"]
        problem.timestep = 0
        problem.done = False

        self._set_task_metadata(task_id)

    def _task_id_from_language(self, language: str) -> int:
        target = language.strip().lower()
        for i in range(self._task_suite.get_num_tasks()):
            if self._task_suite.get_task(i).language.strip().lower() == target:
                return i
        raise ValueError(f"Task language {language!r} not found in suite {self._task_suite_name!r}")

    def close(self) -> None:
        """Close the wrapped LIBERO environment when supported."""
        if self._task_env is not None and hasattr(self._task_env, "close"):
            self._task_env.close()

    @property
    def num_initial_states(self) -> int:
        return len(self._initial_states)
