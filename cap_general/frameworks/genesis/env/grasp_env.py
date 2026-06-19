"""CAP wrapper for Genesis Franka grasp evaluation."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cap_general.core.env import BaseEnv, BaseEnvConfig
from cap_general.frameworks.genesis.utils import load_module_from_file


@dataclass
class GraspEnvConfig(BaseEnvConfig):
    """Configuration for the Genesis grasp manipulation example."""

    example_root: str | Path = "/Users/tongmeng/Desktop/codes/genesis-world/examples/manipulation"
    log_dir: str | Path = "logs/grasp_rl"
    backend: str = "cpu"
    stage: str = "rl"
    show_viewer: bool = False
    num_envs: int = 1
    box_fixed: bool = False
    visualize_camera: bool = False
    record_video: dict[str, str] | None = None
    max_episode_steps: int | None = 1_000_000


@BaseEnv.register()
class GraspEnv(BaseEnv):
    """Genesis grasp manipulation eval environment."""

    name = "Genesis Grasp Env"
    config_cls = GraspEnvConfig

    def __init__(self, config: GraspEnvConfig, logger: logging.Logger | None = None):
        super().__init__(config=config, logger=logger)
        self._config = config
        self._example_env = None
        self._last_policy_obs = None
        self._last_reward = 0.0
        self._last_done = False
        self._mock_reason: str | None = None

    @classmethod
    def env_type(cls) -> str:
        return "genesis_grasp"

    @property
    def example_env(self) -> Any:
        """Return the underlying genesis-world GraspEnv."""
        self._ensure_example_env()
        return self._example_env

    @property
    def policy_obs(self) -> Any:
        """Return the latest policy observation."""
        return self._last_policy_obs

    def _reset(self, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._ensure_example_env()
        if self._example_env is None:
            obs = self._mock_observation()
            return obs, {"mock": True, "reason": self._mock_reason}
        self._last_policy_obs = self._example_env.reset()
        self._last_reward = 0.0
        self._last_done = False
        return self._build_observation(), {"mock": False, "options": options or {}}

    def _step(self, action: Any = None) -> tuple[dict[str, Any], bool, bool, dict[str, Any]]:
        self._ensure_example_env()
        if self._example_env is None:
            return self._mock_observation(), False, False, {"mock": True}

        if action is None:
            action = self._zero_action()
        obs, reward, done, info = self._example_env.step(action)
        self._last_policy_obs = obs
        self._last_reward = float(reward.mean().item()) if hasattr(reward, "mean") else float(reward)
        self._last_done = bool(done.any().item()) if hasattr(done, "any") else bool(done)
        return self._build_observation(), self._last_done, False, info

    def compute_reward(self) -> float:
        return self._last_reward

    def get_observation(self, folder: str | Path) -> dict[str, Any]:
        if self._example_env is None:
            return self._mock_observation()
        return self._build_observation()

    def get_stereo_rgb_images(self, normalize: bool = True) -> Any:
        """Return stereo RGB images from the underlying GraspEnv."""
        self._ensure_example_env()
        if self._example_env is None:
            return None
        return self._example_env.get_stereo_rgb_images(normalize=normalize)

    def grasp_and_lift_demo(self) -> bool:
        """Run the demo lift sequence from the underlying GraspEnv."""
        self._ensure_example_env()
        if self._example_env is None:
            return False
        self._example_env.grasp_and_lift_demo()
        return True

    def _ensure_example_env(self) -> None:
        if self._example_env is not None or self._mock_reason is not None:
            return
        try:
            import genesis as gs
        except ImportError as exc:
            self._mock_reason = f"genesis import failed: {exc}"
            self.logger.warning("Genesis grasp env running in mock mode: %s", self._mock_reason)
            return

        try:
            backend = getattr(gs, self._config.backend)
            try:
                gs.init(backend=backend)
            except Exception as exc:
                message = str(exc)
                if "already" not in message.lower() and "initialized" not in message.lower():
                    raise
            example_root = Path(self._config.example_root).expanduser()
            module = load_module_from_file("cap_general_genesis_grasp_env", example_root / "grasp_env.py")
            env_cfg, reward_cfg, robot_cfg, _rl_train_cfg, _bc_train_cfg = self._load_cfgs()
            env_cfg = dict(env_cfg)
            env_cfg["num_envs"] = self._config.num_envs
            env_cfg["box_fixed"] = self._config.box_fixed
            env_cfg["visualize_camera"] = self._config.visualize_camera
            if self._config.max_episode_steps is not None:
                env_cfg["episode_length_s"] = float(self._config.max_episode_steps) * float(env_cfg["ctrl_dt"])
            if self._config.record_video:
                env_cfg["record_video"] = self._config.record_video
            self._example_env = module.GraspEnv(
                env_cfg=env_cfg,
                reward_cfg=dict(reward_cfg),
                robot_cfg=robot_cfg,
                show_viewer=self._config.show_viewer,
            )
        except Exception as exc:  # pragma: no cover - depends on Genesis runtime
            self._mock_reason = str(exc)
            self.logger.warning("Genesis grasp env running in mock mode: %s", exc)

    def _load_cfgs(self):
        with (Path(self._config.log_dir).expanduser() / "cfgs.pkl").open("rb") as file:
            return pickle.load(file)

    def _zero_action(self):
        import genesis as gs
        import torch

        return torch.zeros(
            (self._example_env.num_envs, self._example_env.num_actions),
            dtype=gs.tc_float,
            device=gs.device,
        )

    def _build_observation(self) -> dict[str, Any]:
        env = self._example_env
        robot = getattr(env, "robot", None)
        return {
            "ee_pose": self._to_list(getattr(robot, "ee_pose", None)),
            "object_pos": self._to_list(env.object.get_pos()) if hasattr(env, "object") else None,
            "object_quat": self._to_list(env.object.get_quat()) if hasattr(env, "object") else None,
            "reward": self._last_reward,
            "done": self._last_done,
            "mock": False,
        }

    def _mock_observation(self) -> dict[str, Any]:
        return {
            "ee_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "object_pos": [0.0, 0.0, 0.0],
            "object_quat": [1.0, 0.0, 0.0, 0.0],
            "reward": self._last_reward,
            "done": self._last_done,
            "mock": True,
            "reason": self._mock_reason,
        }

    @staticmethod
    def _to_list(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value
