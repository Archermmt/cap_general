"""CAP wrapper for Genesis Franka grasp evaluation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import numpy as np

from cap_general.core.robot import BaseRobot, BaseRobotConfig
from cap_general.core.utils import ResetLevel, tensor_to_image_array, tensor_to_list


@dataclass
class GenesisGraspRobotConfig(BaseRobotConfig):
    """Configuration for the Genesis grasp manipulation example."""

    env_cfg: dict[str, Any] = field(
        default_factory=lambda: {
            "num_actions": 6,
            "action_scales": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            "episode_length_s": 3.0,
            "ctrl_dt": 0.01,
            "box_size": [0.08, 0.03, 0.06],
            "image_resolution": [64, 64],
        }
    )
    reward_cfg: dict[str, Any] = field(default_factory=lambda: {"keypoints": 1.0})
    robot_cfg: dict[str, Any] = field(
        default_factory=lambda: {
            "ee_link_name": "hand",
            "gripper_link_names": ["left_finger", "right_finger"],
            "default_arm_dof": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            "default_gripper_dof": [0.04, 0.04],
            "ik_method": "dls_ik",
        }
    )
    stage: str = "rl"
    num_envs: int = 1
    box_fixed: bool = False
    visualize_camera: bool = False
    camera_res: tuple[int, int] = (320, 240)
    camera_fov: float = 65.0
    camera_pos: tuple[float, float, float] = (0.22, 0.0, 0.08)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.13)
    camera_up: tuple[float, float, float] = (0.0, 0.0, -1.0)
    camera_near: float = 0.02
    camera_far: float = 5.0
    record_video: dict[str, str] | None = None
    max_episode_steps: int | None = 1_000_000
    robot_pos: tuple[float, float, float] | None = None
    object_pos_offset: tuple[float, float, float] | None = None
    smooth_reset_steps: int = 125


@BaseRobot.register()
class GenesisGraspRobot(BaseRobot):
    """Genesis grasp manipulation eval environment."""

    robot_type = "genesis_grasp"
    config_cls = GenesisGraspRobotConfig

    def __init__(self, config: GenesisGraspRobotConfig, logger: logging.Logger):
        if config.visualize_camera and "hand_camera_image" not in config.image_keys:
            config.image_keys = [*config.image_keys, "hand_camera_image"]
        super().__init__(config=config, logger=logger)
        self._config = config

        # policy / episode state
        self._last_policy_obs = None
        self._last_reward = 0.0
        self._last_done = False

        # RL environment attributes (populated by init_genesis)
        self.num_envs: int = config.num_envs
        self.num_actions: int = 0
        self.device: Any = None
        self.cfg: dict[str, Any] = {}
        self.ctrl_dt: float = 0.0
        self.max_episode_length: int = 0
        self.episode_length_buf: Any = None
        self.image_height: int = 0
        self.image_width: int = 0
        self.robot: Any = None
        self.object: Any = None
        self.obs_buf: Any = None
        self.reset_buf: Any = None
        self.goal_pose: Any = None
        self.extras: dict = {}
        self.reward_functions: dict = {}
        self.episode_sums: dict = {}
        self.reward_scales: dict = {}
        self.action_scales: Any = None
        self.keypoints_offset: Any = None
        self.scene_offset: Any = None
        self._env_cfg: dict = {}

        # hand camera (set during init_genesis if visualize_camera)
        self._hand_camera: Any = None
        self._hand_camera_failed = False

        # episode length presets loaded from cfgs
        self._train_episode_length_s: float | None = None
        self._eval_episode_length_s: float | None = None

        self.left_cam: Any = None
        self.right_cam: Any = None

    def post_build(self, scene: Any) -> None:
        super().post_build(scene)
        import genesis as gs
        import torch

        # set pd gains (must be called after scene.build)
        self.robot.set_pd_gains()
        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.keypoints_offset = self.get_keypoint_offsets(batch_size=self.num_envs, device=self.device, unit_length=0.5)
        self._init_buffers()

    @property
    def policy_obs(self) -> Any:
        """Return the latest policy observation."""
        if self._last_policy_obs is None:
            self._last_policy_obs = self.get_observations()
        return self._last_policy_obs

    def get_observations(self) -> Any:
        """Return raw vector observations used by training runners."""
        return self._get_observations()

    def _reset(self, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        reset_level = ResetLevel((options or {}).get("reset_level", ResetLevel.AGENT))
        if not self._training and reset_level == ResetLevel.ROBOT:
            self._reset_robot_pose()
            obs = self._get_observations()
            self._last_policy_obs = obs
            self._last_reward = 0.0
            self._last_done = False
            return self._build_observation(), {"options": options or {}, "reset_level": "robot"}
        self._reset_idx()
        obs = self._get_observations()
        if self._training:
            del options
            return obs
        self._last_policy_obs = obs
        self._last_reward = 0.0
        self._last_done = False
        return self._build_observation(), {"options": options or {}}

    def _reset_robot_pose(self) -> None:
        """Reset only the manipulator pose; keep the object where it is."""
        smooth_steps = max(int(self._config.smooth_reset_steps), 0)
        self.robot.reset(smooth_steps=smooth_steps, step_callback=self._advance_reset_step)
        self.episode_length_buf.zero_()
        self.reset_buf.fill_(False)
        self.left_cam._stale = True
        self.right_cam._stale = True

    def _advance_reset_step(self) -> None:
        self._scene.step_scene()
        self._step_cnt += 1
        obs = self._build_observation()
        self._last_obs = obs
        self._record_frame(obs)

    def _step(self, action: Any = None) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        import genesis as gs
        import torch

        if action is None:
            action = torch.zeros(
                (self.num_envs, self.num_actions),
                dtype=gs.tc_float,
                device=gs.device,
            )
        actions = self.rescale_action(action)
        self.robot.apply_action(actions, open_gripper=True)
        self._scene.step_scene()

        self.episode_length_buf += 1
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self._scene.gs_scene.rigid_solver.get_error_envs_mask()
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        reward = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        self._reset_idx(self.reset_buf)
        obs, done, info = self._get_observations(), self.reset_buf, self.extras
        if self._training:
            return obs, reward, done, info
        self._last_policy_obs = obs
        self._last_reward = float(reward.mean().item()) if hasattr(reward, "mean") else float(reward)
        self._last_done = bool(done.any().item()) if hasattr(done, "any") else bool(done)
        return self._build_observation(), 0.0, self._last_done, False, info

    def _on_train(self) -> None:
        self._set_episode_length(self._train_episode_length_s)
        self._last_policy_obs = self._reset()

    def _on_eval(self) -> None:
        self._set_episode_length(self._eval_episode_length_s)
        self._last_policy_obs = self._get_observations()

    def _set_episode_length(self, episode_length_s: float | None) -> None:
        if episode_length_s is None:
            return
        self._env_cfg["episode_length_s"] = episode_length_s
        self.max_episode_length = math.ceil(episode_length_s / self.ctrl_dt)

    def compute_reward(self) -> float:
        return self._last_reward

    def get_stereo_rgb_images(self, normalize: bool = True) -> Any:
        """Return stereo RGB images from the underlying grasp environment."""
        import torch

        rgb_left = self.left_cam.read().rgb
        rgb_right = self.right_cam.read().rgb
        # Genesis may return different resolution than configured; resize to expected.
        expected_w, expected_h = self.image_width, self.image_height
        if rgb_left.shape[-2] != expected_h or rgb_left.shape[-1] != expected_w:
            import torchvision.transforms.functional as TF

            rgb_left = torch.from_numpy(rgb_left).permute(2, 0, 1).unsqueeze(0)
            rgb_right = torch.from_numpy(rgb_right).permute(2, 0, 1).unsqueeze(0)
            rgb_left = TF.resize(rgb_left, (expected_h, expected_w))
            rgb_right = TF.resize(rgb_right, (expected_h, expected_w))
            rgb_left = rgb_left.squeeze(0).permute(1, 2, 0)
            rgb_right = rgb_right.squeeze(0).permute(1, 2, 0)
        else:
            rgb_left = rgb_left.permute(0, 3, 1, 2).float()
            rgb_right = rgb_right.permute(0, 3, 1, 2).float()
        if normalize:
            rgb_left = rgb_left / 255.0
            rgb_right = rgb_right / 255.0
        return torch.cat([rgb_left, rgb_right], dim=1)

    def grasp_and_lift_demo(self) -> bool:
        """Grasp the object and lift it to the final position; gripper stays closed."""
        phase_steps = 125
        goal_pose = self.robot.ee_pose.clone()
        lift_pose = goal_pose.clone()
        lift_pose[:, 2] += 0.3
        final_pose = goal_pose.clone()
        final_pose[:, 0] = 0.3 + self.scene_offset[0]
        final_pose[:, 1] = self.scene_offset[1]
        final_pose[:, 2] = 0.4
        for target in (goal_pose, lift_pose, final_pose):
            for _ in range(phase_steps):
                self.robot.go_to_goal(target, open_gripper=False)
                self._scene.step_scene()
                self._step_cnt += 1
                obs = self._build_observation()
                self._last_obs = obs
                self._record_frame(obs)
        return True

    def release_grasp(self) -> bool:
        """Open the gripper and return to the reset position."""
        import torch

        phase_steps = 125
        reset_pose = torch.tensor([0.2, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        reset_pose[:, :3] += self.scene_offset.reshape(1, 3)
        for _ in range(phase_steps):
            self.robot.go_to_goal(reset_pose, open_gripper=True)
            self._scene.step_scene()
            self._step_cnt += 1
            obs = self._build_observation()
            self._last_obs = obs
            self._record_frame(obs)
        return True

    def init_genesis(self, gs_scene: Any) -> None:
        import genesis as gs
        import torch
        from genesis.vis.camera import Camera

        env_cfg = dict(self._config.env_cfg)
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
        reward_cfg = dict(self._config.reward_cfg)
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self._env_cfg = env_cfg
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)
        self.reward_scales = reward_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)
        self.scene_offset = torch.tensor(
            env_cfg.get("object_pos_offset", env_cfg.get("robot_pos", (0.0, 0.0, 0.0))),
            device=self.device,
            dtype=gs.tc_float,
        )

        self.image_width = env_cfg["image_resolution"][0]
        self.image_height = env_cfg["image_resolution"][1]

        robot_cfg_with_pos = dict(self._config.robot_cfg)
        if "robot_pos" in env_cfg:
            robot_cfg_with_pos["robot_pos"] = env_cfg["robot_pos"]
        self.robot = Manipulator(
            num_envs=self.num_envs,
            scene=gs_scene,
            args=robot_cfg_with_pos,
            device=gs.device,
        )

        self.object = gs_scene.add_entity(
            gs.morphs.Box(
                size=env_cfg["box_size"],
                fixed=env_cfg.get("box_fixed", True),
                batch_fixed_verts=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.0, 0.0)),
            ),
        )

        if _ENABLE_MADRONA and gs.backend == gs.cuda:
            from genesis.options.sensors import BatchRendererCameraOptions

            CameraOptions = BatchRendererCameraOptions
            cam_kwargs = dict(use_rasterizer=True)
        else:
            from genesis.options.sensors import RasterizerCameraOptions

            CameraOptions = RasterizerCameraOptions
            cam_kwargs = {}

        self.left_cam = gs_scene.add_sensor(
            CameraOptions(
                res=(self.image_width, self.image_height),
                pos=(1.25, 0.3, 0.3),
                lookat=(0.0, 0.0, 0.0),
                fov=60,
                **cam_kwargs,
            )
        )
        self.right_cam = gs_scene.add_sensor(
            CameraOptions(
                res=(self.image_width, self.image_height),
                pos=(1.25, -0.3, 0.3),
                lookat=(0.0, 0.0, 0.0),
                fov=60,
                **cam_kwargs,
            )
        )

        def _read_scene_cam(cam):
            rgb = cam.render(rgb=True)[0]
            if rgb.ndim == 4:
                rgb = rgb[0]
            return rgb[..., :3]

        def _read_sensor_cam(cam):
            return cam.read(envs_idx=0).rgb

        for cam_name, filename in (env_cfg.get("record_video") or {}).items():
            cam = getattr(self, cam_name)
            reader = _read_scene_cam if isinstance(cam, Camera) else _read_sensor_cam
            gs_scene.start_recording(
                data_func=partial(reader, cam),
                rec_options=gs.recorders.VideoFile(filename=filename),
            )

        if self._config.visualize_camera:
            self._add_hand_camera(gs_scene)

    def _add_hand_camera(self, scene: Any) -> None:
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
            self._hand_camera = camera
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

    def _build_observation(self) -> dict[str, Any]:
        obs = {
            "ee_pose": tensor_to_list(getattr(self.robot, "ee_pose", None)),
            "object_pos": tensor_to_list(self.object.get_pos()) if self.object is not None else None,
            "object_quat": tensor_to_list(self.object.get_quat()) if self.object is not None else None,
            "reward": self._last_reward,
            "done": self._last_done,
            "mock": False,
        }
        hand_camera_image = self._read_hand_camera_image()
        if hand_camera_image is not None:
            obs["hand_camera_image"] = hand_camera_image
        return obs

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
            return tensor_to_image_array(rgb)
        except Exception as exc:  # pragma: no cover - depends on Genesis renderer/runtime
            self._hand_camera_failed = True
            self.logger.warning("Disabled Genesis grasp hand camera after read failure: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # RL environment interface (called by training runners directly)
    # ------------------------------------------------------------------ #

    def _init_buffers(self) -> None:
        import genesis as gs
        import torch

        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.reset_buf = torch.ones(self.num_envs, dtype=gs.tc_bool, device=gs.device)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=gs.device, dtype=gs.tc_float)
        self.extras = {}

    def _reset_idx(self, envs_idx=None) -> None:
        import torch
        from genesis.utils.geom import transform_quat_by_quat

        # reset_buf may be an inference tensor leaked from training's inference_mode context;
        # clone to a regular tensor so in-place ops below work outside inference_mode.
        if self.reset_buf is not None and self.reset_buf.is_inference():
            self.reset_buf = self.reset_buf.clone()
        if envs_idx is not None and envs_idx.is_inference():
            envs_idx = envs_idx.clone()

        self.robot.reset(envs_idx)

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

        self.left_cam._stale = True
        self.right_cam._stale = True

        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self._env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

    def _get_observations(self) -> Any:
        import torch
        from tensordict import TensorDict

        finger_pos = self.robot.center_finger_pose[:, :3]
        finger_quat = self.robot.center_finger_pose[:, 3:7]
        obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()
        obs_components = [
            finger_pos - obj_pos,
            finger_quat,
            obj_pos,
            obj_quat,
        ]
        self.obs_buf = torch.cat(obs_components, dim=-1)
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def rescale_action(self, action: Any) -> Any:
        return action * self.action_scales

    # ------------ reward functions ----------------

    def _reward_keypoints(self) -> Any:
        import torch
        from genesis.utils.geom import transform_by_trans_quat

        keypoints_offset = self.keypoints_offset
        finger_tip_z_offset = torch.tensor([0.0, 0.0, -0.06], device=self.device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        finger_pos = self.robot.center_finger_pose[:, :3] + finger_tip_z_offset
        finger_quat = self.robot.center_finger_pose[:, 3:7]
        finger_pos_keypoints = transform_by_trans_quat(
            keypoints_offset, finger_pos.unsqueeze(1), finger_quat.unsqueeze(1)
        )
        obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()
        object_pos_keypoints = transform_by_trans_quat(keypoints_offset, obj_pos.unsqueeze(1), obj_quat.unsqueeze(1))
        dist = torch.norm(finger_pos_keypoints - object_pos_keypoints, p=2, dim=-1).sum(-1)
        return torch.exp(-dist)

    # ------------ static helpers ----------------

    @staticmethod
    def get_keypoint_offsets(batch_size: int, device: str, unit_length: float = 0.5) -> Any:
        import torch

        keypoint_offsets = (
            torch.tensor(
                [
                    [0, 0, 0],
                    [-1.0, 0, 0],
                    [1.0, 0, 0],
                    [0, -1.0, 0],
                    [0, 1.0, 0],
                    [0, 0, -1.0],
                    [0, 0, 1.0],
                ],
                device=device,
                dtype=torch.float32,
            )
            * unit_length
        )
        return keypoint_offsets[None].repeat((batch_size, 1, 1))


# Embedded genesis-world grasp env implementation.
try:
    import gs_madrona

    _ENABLE_MADRONA = True
except ImportError:
    _ENABLE_MADRONA = False


## ------------ robot ----------------
class Manipulator:
    def __init__(self, num_envs: int, scene: Any, args: dict, device: str = "cpu"):
        import genesis as gs
        import torch

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
        import torch

        # Note: the following values are tuned for achieving best behavior with Franka
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
        import torch

        self._arm_dof_dim = self._robot_entity.n_dofs - 2
        self._gripper_dim = 2

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

    def reset(self, envs_idx=None, skip_forward=True, smooth_steps: int = 0, step_callback=None):
        if smooth_steps > 0 and step_callback is not None:
            self._smooth_reset(envs_idx=envs_idx, steps=smooth_steps, step_callback=step_callback)
            return
        self._robot_entity.set_qpos(
            self._init_qpos,
            envs_idx=envs_idx,
            zero_velocity=True,
            skip_forward=skip_forward,
        )
        # In a shared scene, inactive manipulators are still advanced by
        # other controls' scene steps. Keep their PD target at the reset pose so
        # they do not collapse while waiting for their turn.
        try:
            self._robot_entity.control_dofs_position(position=self._init_qpos, envs_idx=envs_idx)
        except TypeError:
            self._robot_entity.control_dofs_position(position=self._init_qpos)

    def _smooth_reset(self, envs_idx=None, steps: int = 125, step_callback=None) -> None:
        import math
        import torch

        current_qpos = self._robot_entity.get_qpos()
        target_qpos = self._init_qpos.to(device=current_qpos.device, dtype=current_qpos.dtype)

        if current_qpos.ndim == 1:
            start_qpos = current_qpos
            target_qpos = target_qpos.reshape_as(start_qpos)
            selected_envs = envs_idx
        else:
            target_qpos = target_qpos.reshape(1, -1).expand_as(current_qpos)
            if envs_idx is None:
                start_qpos = current_qpos.clone()
                selected_envs = None
            else:
                selected_envs = envs_idx
                start_qpos = current_qpos[envs_idx].clone()
                target_qpos = target_qpos[envs_idx]

        total_steps = max(int(steps), 1)
        transition_steps = max(int(total_steps * 0.8), 1)
        for step in range(1, total_steps + 1):
            if step <= transition_steps:
                alpha = 0.5 - 0.5 * math.cos(math.pi * step / transition_steps)
            else:
                alpha = 1.0
            target = torch.lerp(start_qpos, target_qpos, alpha)
            try:
                self._robot_entity.control_dofs_position(position=target, envs_idx=selected_envs)
            except TypeError:
                self._robot_entity.control_dofs_position(position=target)
            step_callback()

        try:
            self._robot_entity.control_dofs_position(position=self._init_qpos, envs_idx=envs_idx)
        except TypeError:
            self._robot_entity.control_dofs_position(position=self._init_qpos)

    def apply_action(self, action: Any, open_gripper: bool) -> None:
        """Apply the action to the robot."""
        if self._ik_method == "gs_ik":
            q_pos = self._gs_ik(action)
        elif self._ik_method == "dls_ik":
            q_pos = self._dls_ik(action)
        else:
            raise ValueError(f"Invalid control mode: {self._ik_method}")
        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof
        self._robot_entity.control_dofs_position(position=q_pos)

    def _gs_ik(self, action: Any) -> Any:
        from genesis.utils.geom import transform_quat_by_quat, xyz_to_quat

        delta_position = action[:, :3]
        delta_orientation = action[:, 3:6]
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

    def _dls_ik(self, action: Any) -> Any:
        """Damped least squares inverse kinematics."""
        import torch

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

    def go_to_goal(self, goal_pose: Any, open_gripper: bool = True):
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
    def ee_pose(self) -> Any:
        import torch

        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def left_finger_pose(self) -> Any:
        import torch

        pos, quat = self._left_finger_link.get_pos(), self._left_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def right_finger_pose(self) -> Any:
        import torch

        pos, quat = self._right_finger_link.get_pos(), self._right_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def center_finger_pose(self) -> Any:
        import torch

        left_finger_pose = self.left_finger_pose
        right_finger_pose = self.right_finger_pose
        center_finger_pos = (left_finger_pose[:, :3] + right_finger_pose[:, :3]) / 2
        center_finger_quat = left_finger_pose[:, 3:7]
        return torch.cat([center_finger_pos, center_finger_quat], dim=-1)
