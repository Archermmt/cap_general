"""CAP wrapper for Genesis Franka grasp evaluation."""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from cap_general.core.robot import BaseRobot, BaseRobotConfig
from cap_general.frameworks.genesis.utils import step_scene


def _load_genesis_deps():
    global BatchRendererCameraOptions, Camera, RasterizerCameraOptions, TensorDict, _ENABLE_MADRONA
    global gs, torch, transform_by_trans_quat, transform_quat_by_quat, xyz_to_quat

    import genesis as gs
    import torch
    from genesis.options.sensors import BatchRendererCameraOptions, RasterizerCameraOptions
    from genesis.utils.geom import transform_by_trans_quat, transform_quat_by_quat, xyz_to_quat
    from genesis.vis.camera import Camera
    from tensordict import TensorDict

    try:
        import gs_madrona  # noqa: F401

        _ENABLE_MADRONA = True
    except ImportError:
        _ENABLE_MADRONA = False
    return gs


@dataclass
class GraspRobotConfig(BaseRobotConfig):
    """Configuration for the Genesis grasp manipulation example."""

    example_root: str | Path = "/Users/archer/Desktop/codes/genesis-world/examples/manipulation"
    log_dir: str | Path = "logs/grasp_rl"
    stage: str = "rl"
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
    robot_pos: tuple[float, float, float] | None = None
    object_pos_offset: tuple[float, float, float] | None = None


@BaseRobot.register()
class GraspRobot(BaseRobot):
    """Genesis grasp manipulation eval environment."""

    name = "Genesis Grasp Robot"
    config_cls = GraspRobotConfig

    def __init__(self, config: GraspRobotConfig, logger: logging.Logger):
        if config.visualize_camera and "hand_camera_image" not in config.image_keys:
            config.image_keys = [*config.image_keys, "hand_camera_image"]
        super().__init__(config=config, logger=logger)
        self._config = config
        self._gs_scene = None
        self._example_env = None
        self._last_policy_obs = None
        self._last_reward = 0.0
        self._last_done = False
        self._mock_reason: str | None = None
        self._hand_camera = None
        self._hand_camera_failed = False
        self._train_episode_length_s: float | None = None
        self._eval_episode_length_s: float | None = None

    @classmethod
    def robot_type(cls) -> str:
        return "genesis_grasp"

    def bind_scene(self, scene: Any | None) -> None:
        if scene is None:
            raise RuntimeError("GraspRobot requires a Genesis scene")
        self._gs_scene = scene

    def post_build(self) -> None:
        if self._example_env is not None and getattr(self._example_env, "_deferred_build", False):
            self._example_env._post_build()
            self._last_policy_obs = self._example_env.get_observations()

    @property
    def example_env(self) -> Any:
        """Return the underlying genesis-world GraspRobot."""
        self._ensure_example_env()
        return self._example_env

    @property
    def policy_obs(self) -> Any:
        """Return the latest policy observation."""
        if self._last_policy_obs is None and self._example_env is not None:
            if not getattr(self._example_env, "_deferred_build", False) and hasattr(
                self._example_env, "get_observations"
            ):
                self._last_policy_obs = self._example_env.get_observations()
        return self._last_policy_obs

    def get_observations(self) -> Any:
        """Return raw vector observations used by training runners."""
        env = self._require_example_env()
        return env.get_observations()

    @property
    def num_envs(self) -> int:
        return int(self._require_example_env().num_envs)

    @property
    def num_actions(self) -> int:
        return int(self._require_example_env().num_actions)

    @property
    def device(self) -> Any:
        return self._require_example_env().device

    @property
    def cfg(self) -> dict[str, Any]:
        return self._require_example_env().cfg

    @property
    def max_episode_length(self) -> int:
        return int(self._require_example_env().max_episode_length)

    @property
    def episode_length_buf(self) -> Any:
        return self._require_example_env().episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: Any) -> None:
        self._require_example_env().episode_length_buf = value

    @property
    def image_height(self) -> int:
        return int(self._require_example_env().image_height)

    @property
    def image_width(self) -> int:
        return int(self._require_example_env().image_width)

    @property
    def robot(self) -> Any:
        return self._require_example_env().robot

    @property
    def object(self) -> Any:
        return self._require_example_env().object

    def _reset(self, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._ensure_example_env()
        if self._training:
            del options
            return self._require_example_env().reset()
        if self._example_env is None:
            obs = self._mock_observation()
            return obs, {"mock": True, "reason": self._mock_reason}
        if getattr(self._example_env, "_deferred_build", False):
            obs = self._mock_observation()
            return obs, {"mock": False, "pending_build": True, "options": options or {}}
        self._last_policy_obs = self._example_env.reset()
        self._last_reward = 0.0
        self._last_done = False
        return self._build_observation(), {"mock": False, "options": options or {}}

    def _step(self, action: Any = None) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._ensure_example_env()
        if self._training:
            return self._require_example_env().step(action)
        if self._example_env is None:
            return self._mock_observation(), 0.0, False, False, {"mock": True}
        if getattr(self._example_env, "_deferred_build", False):
            return self._mock_observation(), 0.0, False, False, {"pending_build": True}

        if action is None:
            action = self._zero_action()
        obs, reward, done, info = self._example_env.step(action)
        self._last_policy_obs = obs
        self._last_reward = float(reward.mean().item()) if hasattr(reward, "mean") else float(reward)
        self._last_done = bool(done.any().item()) if hasattr(done, "any") else bool(done)
        return self._build_observation(), 0.0, self._last_done, False, info

    def _on_train(self) -> None:
        env = self._require_example_env()
        self._set_episode_length(self._train_episode_length_s)
        self._last_policy_obs = env.reset()

    def _on_eval(self) -> None:
        if self._example_env is not None and not getattr(self._example_env, "_deferred_build", False):
            self._set_episode_length(self._eval_episode_length_s)
            self._last_policy_obs = self._example_env.get_observations()

    def _set_episode_length(self, episode_length_s: float | None) -> None:
        if episode_length_s is None or self._example_env is None:
            return
        self._example_env.env_cfg["episode_length_s"] = episode_length_s
        self._example_env.max_episode_length = math.ceil(episode_length_s / self._example_env.ctrl_dt)

    def _require_example_env(self) -> Any:
        self._ensure_example_env()
        if self._example_env is None:
            raise RuntimeError("Grasp env is in mock mode and cannot be used for training")
        if getattr(self._example_env, "_deferred_build", False):
            raise RuntimeError("Grasp env must be built before training")
        return self._example_env

    def compute_reward(self) -> float:
        return self._last_reward

    def get_stereo_rgb_images(self, normalize: bool = True) -> Any:
        """Return stereo RGB images from the underlying GraspRobot."""
        self._ensure_example_env()
        if self._example_env is None:
            return None
        return self._example_env.get_stereo_rgb_images(normalize=normalize)

    def grasp_and_lift_demo(self) -> bool:
        """Run the demo lift sequence from the underlying GraspRobot."""
        self._ensure_example_env()
        if self._example_env is None:
            return False
        self._example_env.grasp_and_lift_demo()
        return True

    def _ensure_example_env(self) -> None:
        if self._example_env is not None or self._mock_reason is not None:
            return
        try:
            gs = _load_genesis_deps()
        except ImportError as exc:
            self._mock_reason = f"genesis import failed: {exc}"
            self.logger.warning("Genesis grasp env running in mock mode: %s", self._mock_reason)
            return

        try:
            env_cfg, reward_cfg, robot_cfg, _rl_train_cfg, _bc_train_cfg = self._load_cfgs()
            env_cfg = dict(env_cfg)
            self._train_episode_length_s = float(env_cfg["episode_length_s"])
            env_cfg["num_envs"] = self._config.num_envs
            env_cfg["box_fixed"] = self._config.box_fixed
            env_cfg["visualize_camera"] = False
            if self._config.max_episode_steps is not None:
                env_cfg["episode_length_s"] = float(self._config.max_episode_steps) * float(env_cfg["ctrl_dt"])
            self._eval_episode_length_s = float(env_cfg["episode_length_s"])
            if self._config.record_video:
                env_cfg["record_video"] = self._config.record_video
            if self._config.robot_pos is not None:
                env_cfg["robot_pos"] = list(self._config.robot_pos)
            if self._config.object_pos_offset is not None:
                env_cfg["object_pos_offset"] = list(self._config.object_pos_offset)
            env_cfg["_scene"] = self._gs_scene
            self._example_env = self._build_example_env_with_camera(
                env_cfg=env_cfg,
                reward_cfg=dict(reward_cfg),
                robot_cfg=robot_cfg,
            )
        except Exception as exc:  # pragma: no cover - depends on Genesis runtime
            self._mock_reason = str(exc)
            self.logger.warning("Genesis grasp env running in mock mode: %s", exc)

    def _load_cfgs(self):
        with (Path(self._config.log_dir).expanduser() / "cfgs.pkl").open("rb") as file:
            return pickle.load(file)

    def _build_example_env_with_camera(self, **kwargs: Any) -> Any:
        env_cfg = dict(kwargs["env_cfg"])
        if not self._config.visualize_camera:
            kwargs["env_cfg"] = env_cfg
            return _GenesisGraspCoreRobot(**kwargs)

        camera_holder: dict[str, Any] = {}
        env_cfg["_before_scene_build"] = lambda scene: self._add_hand_camera(scene, camera_holder)
        kwargs["env_cfg"] = env_cfg
        example_env = _GenesisGraspCoreRobot(**kwargs)
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

    def _step_observation(self) -> dict[str, Any]:
        return {
            "reward": self._last_reward,
            "done": self._last_done,
            "mock": False,
        }

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
        return {key: value for key, value in self._last_obs.items() if key not in set(self._image_keys)}

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


# Embedded genesis-world grasp env implementation.
try:
    import gs_madrona

    _ENABLE_MADRONA = True
except ImportError:
    _ENABLE_MADRONA = False


class _GenesisGraspCoreRobot:
    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
    ) -> None:
        self.num_envs = env_cfg["num_envs"]
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        # configs
        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)
        self.scene_offset = torch.tensor(
            env_cfg.get("object_pos_offset", env_cfg.get("robot_pos", (0.0, 0.0, 0.0))),
            device=self.device,
            dtype=gs.tc_float,
        )
        self._deferred_build = False

        # camera config
        self.image_width = env_cfg["image_resolution"][0]
        self.image_height = env_cfg["image_resolution"][1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)

        # == setup scene ==
        self.scene = env_cfg["_scene"]

        # == add robot ==
        robot_cfg_with_pos = dict(robot_cfg)
        if "robot_pos" in env_cfg:
            robot_cfg_with_pos["robot_pos"] = env_cfg["robot_pos"]
        self.robot = Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            args=robot_cfg_with_pos,
            device=gs.device,
        )

        # == add object ==
        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=env_cfg["box_size"],
                fixed=env_cfg.get("box_fixed", True),
                batch_fixed_verts=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.0, 0.0),
                ),
            ),
        )

        # == stereo camera sensors (lazy rendering — zero cost until read()) ==
        if _ENABLE_MADRONA and gs.backend == gs.cuda:
            CameraOptions = BatchRendererCameraOptions
            cam_kwargs = dict(use_rasterizer=True)
        else:
            CameraOptions = RasterizerCameraOptions
            cam_kwargs = {}

        self.left_cam = self.scene.add_sensor(
            CameraOptions(
                res=(self.image_width, self.image_height),
                pos=(1.25, 0.3, 0.3),
                lookat=(0.0, 0.0, 0.0),
                fov=60,
                **cam_kwargs,
            )
        )
        self.right_cam = self.scene.add_sensor(
            CameraOptions(
                res=(self.image_width, self.image_height),
                pos=(1.25, -0.3, 0.3),
                lookat=(0.0, 0.0, 0.0),
                fov=60,
                **cam_kwargs,
            )
        )

        # == camera data readers ==
        def _read_scene_cam(cam):
            rgb = cam.render(rgb=True)[0]
            if rgb.ndim == 4:
                rgb = rgb[0]
            return rgb[..., :3]

        def _read_sensor_cam(cam):
            return cam.read(envs_idx=0).rgb

        # == set up video recording (must be before build) ==

        record_video = env_cfg.get("record_video", {})
        for cam_name, filename in record_video.items():
            cam = getattr(self, cam_name)
            reader = _read_scene_cam if isinstance(cam, Camera) else _read_sensor_cam
            self.scene.start_recording(
                data_func=partial(reader, cam),
                rec_options=gs.recorders.VideoFile(filename=filename),
            )

        # build
        before_scene_build = env_cfg.get("_before_scene_build")
        if before_scene_build is not None:
            before_scene_build(self.scene)
        self._deferred_build = True

    def _post_build(self) -> None:
        self._deferred_build = False
        # set pd gains (must be called after scene.build)
        self.robot.set_pd_gains()

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.keypoints_offset = self.get_keypoint_offsets(batch_size=self.num_envs, device=self.device, unit_length=0.5)
        # == init buffers ==
        self._init_buffers()
        self.reset()

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.reset_buf = torch.ones(self.num_envs, dtype=gs.tc_bool, device=gs.device)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=gs.device, dtype=gs.tc_float)
        self.extras = dict()

    def _reset_idx(self, envs_idx=None) -> None:
        """Reset specified environments.

        Parameters
        ----------
        envs_idx : torch.Tensor or None
            Boolean mask of shape (num_envs,) for selective reset, or None for full reset.
        """
        # Reset robot
        self.robot.reset(envs_idx)

        # Generate random object state for all envs
        random_x = torch.rand(self.num_envs, device=self.device) * 0.4 + 0.2
        random_y = (torch.rand(self.num_envs, device=self.device) - 0.5) * 0.5
        random_z = torch.full((self.num_envs,), 0.025, device=self.device)
        random_pos = torch.stack([random_x, random_y, random_z], dim=-1)
        random_pos = random_pos + self.scene_offset.reshape(1, 3)

        q_downward = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
        random_yaw = (torch.rand(self.num_envs, device=self.device) * 2 * math.pi - math.pi) * 0.25
        q_yaw = torch.stack(
            [
                torch.cos(random_yaw / 2),
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                torch.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        goal_yaw = transform_quat_by_quat(q_yaw, q_downward)
        goal_pose = torch.cat([random_pos, goal_yaw], dim=-1)

        # Reset object — set_pos/set_quat with skip_forward, then FK runs once for everything
        if envs_idx is None:
            self.goal_pose.copy_(goal_pose)
            self.object.set_pos(random_pos, skip_forward=True)
            self.object.set_quat(goal_yaw, skip_forward=False)
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            torch.where(envs_idx[:, None], goal_pose, self.goal_pose, out=self.goal_pose)
            self.object.set_pos(random_pos, envs_idx=envs_idx, skip_forward=True)
            self.object.set_quat(goal_yaw, envs_idx=envs_idx, skip_forward=False)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # Invalidate camera caches after state change
        self.left_cam._stale = True
        self.right_cam._stale = True

        # Fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

    def reset(self) -> TensorDict:
        self._reset_idx()
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # apply action
        actions = self.rescale_action(actions)
        self.robot.apply_action(actions, open_gripper=True)
        step_scene(self.scene)

        # update time
        self.episode_length_buf += 1

        # check termination (bool mask)
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()

        # timeout for value bootstrapping (only true timeouts, not NaN errors)
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        # compute reward before reset (reflects terminal state)
        reward = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # soft-reset envs that need it
        self._reset_idx(self.reset_buf)

        return self.get_observations(), reward, self.reset_buf, self.extras

    def get_observations(self) -> TensorDict:
        # Current end-effector pose
        finger_pos, finger_quat = (
            self.robot.center_finger_pose[:, :3],
            self.robot.center_finger_pose[:, 3:7],
        )
        obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()
        obs_components = [
            finger_pos - obj_pos,  # 3D position difference
            finger_quat,  # current orientation (w, x, y, z)
            obj_pos,  # goal position
            obj_quat,  # goal orientation (w, x, y, z)
        ]
        self.obs_buf = torch.cat(obs_components, dim=-1)
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scales

    def get_stereo_rgb_images(self, normalize: bool = True) -> torch.Tensor:
        rgb_left = self.left_cam.read().rgb  # (B, H, W, 3)
        rgb_right = self.right_cam.read().rgb  # (B, H, W, 3)

        # Convert to (B, 3, H, W) float
        rgb_left = rgb_left.permute(0, 3, 1, 2).float()
        rgb_right = rgb_right.permute(0, 3, 1, 2).float()

        if normalize:
            rgb_left = rgb_left / 255.0
            rgb_right = rgb_right / 255.0

        # Concatenate left and right rgb images along channel dimension: [B, 6, H, W]
        return torch.cat([rgb_left, rgb_right], dim=1)

    # ------------ begin reward functions----------------
    def _reward_keypoints(self) -> torch.Tensor:
        keypoints_offset = self.keypoints_offset
        # there is a offset between the finger tip and the finger base frame
        finger_tip_z_offset = torch.tensor(
            [0.0, 0.0, -0.06],
            device=self.device,
            dtype=gs.tc_float,
        ).repeat(self.num_envs, 1)
        finger_pos_keypoints = self._to_world_frame(
            self.robot.center_finger_pose[:, :3] + finger_tip_z_offset,
            self.robot.center_finger_pose[:, 3:7],
            keypoints_offset,
        )
        object_pos_keypoints = self._to_world_frame(self.object.get_pos(), self.object.get_quat(), keypoints_offset)
        dist = torch.norm(finger_pos_keypoints - object_pos_keypoints, p=2, dim=-1).sum(-1)
        return torch.exp(-dist)

    # ------------ end reward functions----------------

    @staticmethod
    def _to_world_frame(
        position: torch.Tensor,  # [B, 3]
        quaternion: torch.Tensor,  # [B, 4]
        keypoints_offset: torch.Tensor,  # [B, K, 3]
    ) -> torch.Tensor:
        return transform_by_trans_quat(keypoints_offset, position.unsqueeze(1), quaternion.unsqueeze(1))

    @staticmethod
    def get_keypoint_offsets(batch_size: int, device: str, unit_length: float = 0.5) -> torch.Tensor:
        """
        Get uniformly-spaced keypoints along a line of unit length, centered at body center.
        """
        keypoint_offsets = (
            torch.tensor(
                [
                    [0, 0, 0],  # origin
                    [-1.0, 0, 0],  # x-negative
                    [1.0, 0, 0],  # x-positive
                    [0, -1.0, 0],  # y-negative
                    [0, 1.0, 0],  # y-positive
                    [0, 0, -1.0],  # z-negative
                    [0, 0, 1.0],  # z-positive
                ],
                device=device,
                dtype=torch.float32,
            )
            * unit_length
        )
        return keypoint_offsets[None].repeat((batch_size, 1, 1))

    def grasp_and_lift_demo(self) -> None:
        total_steps = 500
        goal_pose = self.robot.ee_pose.clone()
        # lift pose (above the object)
        lift_height = 0.3
        lift_pose = goal_pose.clone()
        lift_pose[:, 2] += lift_height
        # final pose (above the table)
        final_pose = goal_pose.clone()
        final_pose[:, 0] = 0.3 + self.scene_offset[0]
        final_pose[:, 1] = self.scene_offset[1]
        final_pose[:, 2] = 0.4
        # reset pose (home pose)
        reset_pose = torch.tensor([0.2, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        reset_pose[:, :3] += self.scene_offset.reshape(1, 3)
        for i in range(total_steps):
            if i < total_steps / 4:  # grasping
                self.robot.go_to_goal(goal_pose, open_gripper=False)
            elif i < total_steps / 2:  # lifting
                self.robot.go_to_goal(lift_pose, open_gripper=False)
            elif i < total_steps * 3 / 4:  # final
                self.robot.go_to_goal(final_pose, open_gripper=False)
            else:  # reset
                self.robot.go_to_goal(reset_pose, open_gripper=True)
            step_scene(self.scene)


## ------------ robot ----------------
class Manipulator:
    def __init__(self, num_envs: int, scene: gs.Scene, args: dict, device: str = "cpu"):
        # == set members ==
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        # == Genesis configurations ==
        material: gs.materials.Rigid = gs.materials.Rigid()
        robot_pos = tuple(args.get("robot_pos", (0.0, 0.0, 0.0))) if args.get("robot_pos") else (0.0, 0.0, 0.0)
        morph: gs.morphs.MJCF = gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=robot_pos,
            quat=(1.0, 0.0, 0.0, 0.0),
        )
        self._robot_entity: gs.Entity = scene.add_entity(material=material, morph=morph)

        self._gripper_open_dof = 0.04
        self._gripper_close_dof = 0.00

        self._ik_method: Literal["gs_ik", "dls_ik"] = args["ik_method"]

        # == some buffer initialization ==
        self._init()

    def set_pd_gains(self):
        # set control gains
        # Note: the following values are tuned for achieving best behavior with Franka
        # Typically, each new robot would have a different set of parameters.
        # Sometimes high-quality URDF or XML file would also provide this and will be parsed.
        self._robot_entity.set_dofs_kp(
            torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self._robot_entity.set_dofs_kv(
            torch.tensor([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        self._robot_entity.set_dofs_force_range(
            torch.tensor([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            torch.tensor([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

    def _init(self):
        self._arm_dof_dim = self._robot_entity.n_dofs - 2  # total number of arm joints
        self._gripper_dim = 2  # number of gripper joints

        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
        self._fingers_dof = torch.arange(
            self._arm_dof_dim,
            self._arm_dof_dim + self._gripper_dim,
            device=self._device,
        )
        self._left_finger_dof = self._fingers_dof[0]
        self._right_finger_dof = self._fingers_dof[1]
        self._ee_link = self._robot_entity.get_link(self._args["ee_link_name"])
        self._left_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][0])
        self._right_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][1])
        self._default_joint_angles = self._args["default_arm_dof"]
        if self._args["default_gripper_dof"] is not None:
            self._default_joint_angles += self._args["default_gripper_dof"]
        self._init_qpos = torch.tensor(self._default_joint_angles, dtype=torch.float32, device=self._device)
        # On MPS/Metal, batched linear algebra is extremely slow due to per-element kernel dispatch.
        # Running the DLS solve on CPU is ~300x faster in that case.
        self._dls_solve_on_cpu = self._device == "mps" or str(self._device).startswith("mps")
        dls_lam_device = "cpu" if self._dls_solve_on_cpu else self._device
        self._dls_lambda_matrix = (0.01**2) * torch.eye(6, device=dls_lam_device)

    def reset(self, envs_idx=None, skip_forward=True):
        self._robot_entity.set_qpos(
            self._init_qpos,
            envs_idx=envs_idx,
            zero_velocity=True,
            skip_forward=skip_forward,
        )
        # In a shared scene, inactive manipulators are still advanced by
        # other agents' scene steps. Keep their PD target at the reset pose so
        # they do not collapse while waiting for their turn.
        try:
            self._robot_entity.control_dofs_position(position=self._init_qpos, envs_idx=envs_idx)
        except TypeError:
            self._robot_entity.control_dofs_position(position=self._init_qpos)

    def apply_action(self, action: torch.Tensor, open_gripper: bool) -> None:
        """Apply the action to the robot."""
        if self._ik_method == "gs_ik":
            q_pos = self._gs_ik(action)
        elif self._ik_method == "dls_ik":
            q_pos = self._dls_ik(action)
        else:
            raise ValueError(f"Invalid control mode: {self._ik_method}")
        # set gripper to open
        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof
        self._robot_entity.control_dofs_position(position=q_pos)

    def _gs_ik(self, action: torch.Tensor) -> torch.Tensor:
        """
        Genesis inverse kinematics
        """
        delta_position = action[:, :3]
        delta_orientation = action[:, 3:6]

        # compute target pose
        target_position = delta_position + self._ee_link.get_pos()
        quat_rel = xyz_to_quat(delta_orientation, rpy=True, degrees=False)
        target_orientation = transform_quat_by_quat(quat_rel, self._ee_link.get_quat())
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=target_position,
            quat=target_orientation,
            dofs_idx_local=self._arm_dof_idx,
        )
        return q_pos

    def _dls_ik(self, action: torch.Tensor) -> torch.Tensor:
        """
        Damped least squares inverse kinematics.

        Solves (J @ J^T + lambda^2 * I) @ y = dx, then dq = J^T @ y.
        """
        delta_pose = action[:, :6]
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)
        if self._dls_solve_on_cpu:
            jacobian = jacobian.cpu()
            delta_pose = delta_pose.cpu()
        A = torch.baddbmm(self._dls_lambda_matrix, jacobian, jacobian.mT)
        y = torch.linalg.solve(A, delta_pose)
        delta_joint_pos = (jacobian.mT @ y.unsqueeze(-1)).squeeze(-1)
        if self._dls_solve_on_cpu:
            delta_joint_pos = delta_joint_pos.to(self._device)
        return self._robot_entity.get_qpos() + delta_joint_pos

    def go_to_goal(self, goal_pose: torch.Tensor, open_gripper: bool = True):
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=goal_pose[:, :3],
            quat=goal_pose[:, 3:7],
            dofs_idx_local=self._arm_dof_idx,
        )
        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof
        self._robot_entity.control_dofs_position(position=q_pos)

    @property
    def base_pos(self):
        return self._robot_entity.get_pos()

    @property
    def ee_pose(self) -> torch.Tensor:
        """
        The end-effector pose (the hand pose)
        """
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def left_finger_pose(self) -> torch.Tensor:
        pos, quat = self._left_finger_link.get_pos(), self._left_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def right_finger_pose(self) -> torch.Tensor:
        pos, quat = (
            self._right_finger_link.get_pos(),
            self._right_finger_link.get_quat(),
        )
        return torch.cat([pos, quat], dim=-1)

    @property
    def center_finger_pose(self) -> torch.Tensor:
        """
        The center finger pose is the average of the left and right finger poses.
        """
        left_finger_pose = self.left_finger_pose
        right_finger_pose = self.right_finger_pose
        center_finger_pos = (left_finger_pose[:, :3] + right_finger_pose[:, :3]) / 2
        center_finger_quat = left_finger_pose[:, 3:7]
        return torch.cat([center_finger_pos, center_finger_quat], dim=-1)
