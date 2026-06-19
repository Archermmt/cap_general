"""CAP wrapper for Genesis GO2 locomotion evaluation."""

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
class Go2EnvConfig(BaseEnvConfig):
    """Configuration for the Genesis GO2 locomotion example."""

    example_root: str | Path = "/Users/tongmeng/Desktop/codes/genesis-world/examples/locomotion"
    log_dir: str | Path = "logs/go2-walking"
    backend: str = "cpu"
    show_viewer: bool = False
    num_envs: int = 1


@BaseEnv.register()
class Go2Env(BaseEnv):
    """Genesis GO2 locomotion eval environment."""

    name = "Genesis GO2 Env"
    config_cls = Go2EnvConfig

    def __init__(self, config: Go2EnvConfig, logger: logging.Logger | None = None):
        super().__init__(config=config, logger=logger)
        self._config = config
        self._example_env = None
        self._last_policy_obs = None
        self._last_reward = 0.0
        self._last_done = False
        self._mock_reason: str | None = None

    @classmethod
    def env_type(cls) -> str:
        return "genesis_go2"

    @property
    def example_env(self) -> Any:
        """Return the underlying genesis-world Go2Env."""
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

    def _ensure_example_env(self) -> None:
        if self._example_env is not None or self._mock_reason is not None:
            return
        try:
            import genesis as gs
        except ImportError as exc:
            self._mock_reason = f"genesis import failed: {exc}"
            self.logger.warning("Genesis GO2 env running in mock mode: %s", self._mock_reason)
            return

        try:
            backend = getattr(gs, self._config.backend)
            gs.init(backend=backend)
            example_root = Path(self._config.example_root).expanduser()
            module = load_module_from_file("cap_general_genesis_go2_env", example_root / "go2_env.py")
            env_cfg, obs_cfg, reward_cfg, command_cfg, _train_cfg = self._load_cfgs()
            env_cfg = dict(env_cfg)
            reward_cfg = dict(reward_cfg)
            reward_cfg["reward_scales"] = {}
            self._example_env = module.Go2Env(
                num_envs=self._config.num_envs,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                show_viewer=self._config.show_viewer,
            )
        except Exception as exc:  # pragma: no cover - depends on Genesis runtime
            self._mock_reason = str(exc)
            self.logger.warning("Genesis GO2 env running in mock mode: %s", exc)

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
        return {
            "base_pos": self._to_list(getattr(env, "base_pos", None)),
            "base_quat": self._to_list(getattr(env, "base_quat", None)),
            "commands": self._to_list(getattr(env, "commands", None)),
            "reward": self._last_reward,
            "done": self._last_done,
            "mock": False,
        }

    def _mock_observation(self) -> dict[str, Any]:
        return {
            "base_pos": [0.0, 0.0, 0.0],
            "base_quat": [1.0, 0.0, 0.0, 0.0],
            "commands": [],
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
