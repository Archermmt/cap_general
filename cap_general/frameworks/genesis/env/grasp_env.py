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
    camera_res: tuple[int, int] = (320, 240)
    camera_fov: float = 90.0
    camera_pos: tuple[float, float, float] = (0.0, 0.0, 0.10)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.45)
    camera_up: tuple[float, float, float] = (1.0, 0.0, 0.0)
    camera_near: float = 0.02
    camera_far: float = 5.0
    record_video: dict[str, str] | None = None
    max_episode_steps: int | None = 1_000_000


@BaseEnv.register()
class GraspEnv(BaseEnv):
    """Genesis grasp manipulation eval environment."""

    name = "Genesis Grasp Env"
    config_cls = GraspEnvConfig

    def __init__(self, config: GraspEnvConfig, logger: logging.Logger | None = None):
        if config.visualize_camera and "hand_camera_image" not in config.image_keys:
            config.image_keys = [*config.image_keys, "hand_camera_image"]
        super().__init__(config=config, logger=logger)
        self._config = config
        self._example_env = None
        self._last_policy_obs = None
        self._last_reward = 0.0
        self._last_done = False
        self._mock_reason: str | None = None
        self._hand_camera = None
        self._hand_camera_failed = False

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
        self._last_obs = self._mock_observation() if self._example_env is None else self._build_observation()
        return super().get_observation(folder)

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
            env_cfg["visualize_camera"] = False
            if self._config.max_episode_steps is not None:
                env_cfg["episode_length_s"] = float(self._config.max_episode_steps) * float(env_cfg["ctrl_dt"])
            if self._config.record_video:
                env_cfg["record_video"] = self._config.record_video
            self._example_env = self._build_example_env_with_camera(
                module=module,
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

    def _build_example_env_with_camera(self, module: Any, **kwargs: Any) -> Any:
        if not self._config.visualize_camera:
            return module.GraspEnv(**kwargs)

        original_scene_cls = module.gs.Scene
        camera_holder: dict[str, Any] = {}

        def scene_factory(*scene_args: Any, **scene_kwargs: Any) -> Any:
            scene = original_scene_cls(*scene_args, **scene_kwargs)
            original_build = scene.build

            def build_with_hand_camera(*build_args: Any, **build_kwargs: Any) -> Any:
                self._add_hand_camera(scene, camera_holder)
                return original_build(*build_args, **build_kwargs)

            scene.build = build_with_hand_camera
            return scene

        module.gs.Scene = scene_factory
        try:
            example_env = module.GraspEnv(**kwargs)
        finally:
            module.gs.Scene = original_scene_cls
        self._hand_camera = camera_holder.get("camera")
        return example_env

    def _add_hand_camera(self, scene: Any, camera_holder: dict[str, Any]) -> None:
        if camera_holder.get("camera") is not None:
            return
        try:
            from genesis.utils import geom as gu

            hand_link = self._find_hand_link(scene)
            if hand_link is None:
                raise RuntimeError("could not find Franka hand link for camera attachment")

            camera = scene.add_camera(
                res=tuple(self._config.camera_res),
                pos=tuple(self._config.camera_pos),
                lookat=tuple(self._config.camera_lookat),
                up=tuple(self._config.camera_up),
                fov=float(self._config.camera_fov),
                near=float(self._config.camera_near),
                far=float(self._config.camera_far),
                GUI=False,
                debug=True,
            )
            offset_T = gu.pos_lookat_up_to_T(
                np.asarray(self._config.camera_pos, dtype=np.float32),
                np.asarray(self._config.camera_lookat, dtype=np.float32),
                np.asarray(self._config.camera_up, dtype=np.float32),
            )
            camera.attach(hand_link, offset_T)
            camera_holder["camera"] = camera
        except Exception as exc:  # pragma: no cover - depends on Genesis renderer/runtime
            self.logger.warning("Failed to add Genesis grasp hand camera: %s", exc)

    @staticmethod
    def _find_hand_link(scene: Any) -> Any | None:
        for entity in getattr(scene, "entities", []):
            if not hasattr(entity, "get_link"):
                continue
            for link_name in ("hand", "panda_hand", "franka_hand"):
                try:
                    return entity.get_link(link_name)
                except Exception:
                    continue
        return None

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
        obs = {
            "ee_pose": self._to_list(getattr(robot, "ee_pose", None)),
            "object_pos": self._to_list(env.object.get_pos()) if hasattr(env, "object") else None,
            "object_quat": self._to_list(env.object.get_quat()) if hasattr(env, "object") else None,
            "reward": self._last_reward,
            "done": self._last_done,
            "mock": False,
        }
        hand_camera_image = self._read_hand_camera_image()
        if hand_camera_image is not None:
            obs["hand_camera_image"] = hand_camera_image
        return obs

    def _mock_observation(self) -> dict[str, Any]:
        return {
            "ee_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "object_pos": [0.0, 0.0, 0.0],
            "object_quat": [1.0, 0.0, 0.0, 0.0],
            "hand_camera_image": None,
            "reward": self._last_reward,
            "done": self._last_done,
            "mock": True,
            "reason": self._mock_reason,
        }

    def _normalize_states(self) -> dict:
        if self._last_obs is None or not isinstance(self._last_obs, dict):
            return {}
        return {
            key: value
            for key, value in self._last_obs.items()
            if key not in set(self._image_keys)
        }

    def _read_hand_camera_image(self) -> Any | None:
        if self._hand_camera is None or self._hand_camera_failed:
            return None
        try:
            self._hand_camera.move_to_attach()
            rgb = self._hand_camera.render(rgb=True, force_render=True)[0]
            if getattr(rgb, "ndim", 0) > 3:
                rgb = rgb[0]
            return self._to_image_array(rgb)
        except Exception as exc:  # pragma: no cover - depends on Genesis renderer/runtime
            self._hand_camera_failed = True
            self.logger.warning("Disabled Genesis grasp hand camera after read failure: %s", exc)
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
