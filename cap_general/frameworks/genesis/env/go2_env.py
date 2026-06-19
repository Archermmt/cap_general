"""CAP wrapper for Genesis GO2 locomotion evaluation."""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass, field
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
    image_keys: list[str] = field(default_factory=lambda: ["body_camera_image"])
    camera_enabled: bool = True
    camera_res: tuple[int, int] = (320, 240)
    camera_fov: float = 70.0
    camera_pos: tuple[float, float, float] = (0.35, 0.0, 0.25)
    camera_lookat: tuple[float, float, float] = (1.0, 0.0, 0.15)
    camera_near: float = 0.05
    camera_far: float = 20.0
    turn_action_scale: float = 0.35
    max_episode_steps: int | None = 1_000_000


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
        self._body_camera = None
        self._body_camera_failed = False

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

    @property
    def dt(self) -> float:
        """Return the GO2 policy control period in seconds."""
        env = self._example_env
        return float(getattr(env, "dt", 1.0)) if env is not None else 1.0

    def set_walk_command(
        self,
        *,
        turn_angle: float = 0.0,
        steps: int | None = None,
        forward_speed: float | None = None,
    ) -> bool:
        """Set the GO2 forward walking command.

        ``turn_angle`` is the target yaw angle in radians over ``steps`` policy
        steps. It is converted to the yaw-rate command consumed by the
        genesis-world GO2 policy.
        """
        self._ensure_example_env()
        env = self._example_env
        if env is None:
            return False

        commands = getattr(env, "commands", None)
        if commands is None:
            return False

        if forward_speed is not None:
            commands[:, 0] = float(forward_speed)
        commands[:, 1] = 0.0
        if steps and steps > 0:
            dt = float(getattr(env, "dt", 1.0))
            commands[:, 2] = float(turn_angle) / max(float(steps) * dt, 1e-6)
        else:
            commands[:, 2] = 0.0

        if hasattr(env, "_update_observation"):
            env._update_observation()
        if hasattr(env, "get_observations"):
            self._last_policy_obs = env.get_observations()
        return True

    def stop_command(self) -> bool:
        """Set all GO2 velocity commands to zero."""
        self._ensure_example_env()
        env = self._example_env
        if env is None:
            return False

        commands = getattr(env, "commands", None)
        if commands is None:
            return False
        commands.zero_()

        if hasattr(env, "_update_observation"):
            env._update_observation()
        if hasattr(env, "get_observations"):
            self._last_policy_obs = env.get_observations()
        return True

    def apply_turn_to_action(self, action: Any, turn_angle: float) -> Any:
        """Bias policy action so GO2 walks while turning smoothly."""
        if not turn_angle:
            return action

        env = self._example_env
        env_cfg = getattr(env, "env_cfg", {}) if env is not None else {}
        joint_names = list(env_cfg.get("joint_names", []))
        if not joint_names:
            return action

        try:
            import torch
        except ImportError:
            return action

        turn_value = (float(turn_angle) + math.pi) % (2.0 * math.pi) - math.pi
        turn = torch.as_tensor(turn_value, dtype=action.dtype, device=action.device)
        steering = torch.clamp(turn / torch.pi, -1.0, 1.0) * float(self._config.turn_action_scale)
        if torch.allclose(steering, torch.zeros_like(steering)):
            return action

        adjusted = action.clone()
        bias_by_joint = {
            "FR_hip_joint": steering,
            "RR_hip_joint": steering,
            "FL_hip_joint": -steering,
            "RL_hip_joint": -steering,
        }
        for joint_name, bias in bias_by_joint.items():
            if joint_name in joint_names:
                adjusted[:, joint_names.index(joint_name)] += bias
        return adjusted

    def _reset(self, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._ensure_example_env()
        if self._example_env is None:
            obs = self._mock_observation()
            return obs, {"mock": True, "reason": self._mock_reason}
        if hasattr(self._example_env, "get_observations"):
            self._last_policy_obs = self._example_env.get_observations()
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
        self._last_obs = self._mock_observation() if self._example_env is None else self._build_observation()
        return super().get_observation(folder)

    def _normalize_states(self) -> dict:
        if self._last_obs is None or not isinstance(self._last_obs, dict):
            return {}
        return {
            key: value
            for key, value in self._last_obs.items()
            if key not in set(self._image_keys)
        }

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
            if self._config.max_episode_steps is not None:
                env_cfg["episode_length_s"] = float(self._config.max_episode_steps) * 0.02
            reward_cfg = dict(reward_cfg)
            reward_cfg["reward_scales"] = {}
            self._example_env = self._build_example_env_with_camera(
                module=module,
                num_envs=self._config.num_envs,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                show_viewer=self._config.show_viewer,
            )
            if hasattr(self._example_env, "get_observations"):
                self._last_policy_obs = self._example_env.get_observations()
        except Exception as exc:  # pragma: no cover - depends on Genesis runtime
            self._mock_reason = str(exc)
            self.logger.warning("Genesis GO2 env running in mock mode: %s", exc)

    def _build_example_env_with_camera(self, module: Any, **kwargs: Any) -> Any:
        if not self._config.camera_enabled:
            return module.Go2Env(**kwargs)

        original_scene_cls = module.gs.Scene
        camera_holder: dict[str, Any] = {}

        def scene_factory(*scene_args: Any, **scene_kwargs: Any) -> Any:
            scene = original_scene_cls(*scene_args, **scene_kwargs)
            original_build = scene.build

            def build_with_body_camera(*build_args: Any, **build_kwargs: Any) -> Any:
                self._add_body_camera(scene, camera_holder)
                return original_build(*build_args, **build_kwargs)

            scene.build = build_with_body_camera
            return scene

        module.gs.Scene = scene_factory
        try:
            example_env = module.Go2Env(**kwargs)
        finally:
            module.gs.Scene = original_scene_cls
        self._body_camera = camera_holder.get("camera")
        return example_env

    def _add_body_camera(self, scene: Any, camera_holder: dict[str, Any]) -> None:
        if camera_holder.get("camera") is not None:
            return
        try:
            from genesis.utils import geom as gu

            robot = scene.entities[-1]
            camera = scene.add_camera(
                res=tuple(self._config.camera_res),
                pos=tuple(self._config.camera_pos),
                lookat=tuple(self._config.camera_lookat),
                up=(0.0, 0.0, 1.0),
                fov=float(self._config.camera_fov),
                near=float(self._config.camera_near),
                far=float(self._config.camera_far),
                GUI=False,
                debug=True,
            )
            base_link = getattr(robot, "base_link", None) or robot.links[0]
            offset_T = gu.pos_lookat_up_to_T(
                np.asarray(self._config.camera_pos, dtype=np.float32),
                np.asarray(self._config.camera_lookat, dtype=np.float32),
                np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
            )
            camera.attach(base_link, offset_T)
            camera_holder["camera"] = camera
        except Exception as exc:  # pragma: no cover - depends on Genesis renderer/runtime
            self.logger.warning("Failed to add GO2 body camera: %s", exc)

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
        obs = {
            "base_pos": self._to_list(getattr(env, "base_pos", None)),
            "base_quat": self._to_list(getattr(env, "base_quat", None)),
            "commands": self._to_list(getattr(env, "commands", None)),
            "reward": self._last_reward,
            "done": self._last_done,
            "mock": False,
        }
        body_camera_image = self._read_body_camera_image()
        if body_camera_image is not None:
            obs["body_camera_image"] = body_camera_image
        return obs

    def _mock_observation(self) -> dict[str, Any]:
        return {
            "base_pos": [0.0, 0.0, 0.0],
            "base_quat": [1.0, 0.0, 0.0, 0.0],
            "commands": [],
            "body_camera_image": None,
            "reward": self._last_reward,
            "done": self._last_done,
            "mock": True,
            "reason": self._mock_reason,
        }

    def _read_body_camera_image(self) -> Any | None:
        if self._body_camera is None or self._body_camera_failed:
            return None
        try:
            self._body_camera.move_to_attach()
            rgb = self._body_camera.render(rgb=True, force_render=True)[0]
            if getattr(rgb, "ndim", 0) > 3:
                rgb = rgb[0]
            return self._to_image_array(rgb)
        except Exception as exc:  # pragma: no cover - depends on Genesis renderer/runtime
            self._body_camera_failed = True
            self.logger.warning("Disabled GO2 body camera after read failure: %s", exc)
            return None

    @staticmethod
    def _to_list(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    @staticmethod
    def _to_image_array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.dtype != np.uint8:
            if array.size and float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array
