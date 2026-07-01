"""CAP wrapper for Genesis drone hover evaluation."""

from __future__ import annotations

import copy
import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cap_general.core.robot import BaseRobot, BaseRobotConfig
from cap_general.core.utils import tensor_to_image_array, tensor_to_list


@dataclass
class DroneHoverRobotConfig(BaseRobotConfig):
    """Configuration for the Genesis drone hover example."""

    example_root: str | Path = "/Users/archer/Desktop/codes/genesis-world/examples/drone"
    log_dir: str | Path = "logs/drone-hovering"
    num_envs: int = 1
    visualize_target: bool = True
    visualize_camera: bool = False
    camera_enabled: bool = True
    camera_res: tuple[int, int] = (320, 240)
    camera_fov: float = 90.0
    camera_pos: tuple[float, float, float] = (0.12, 0.0, -0.03)
    camera_lookat: tuple[float, float, float] = (1.0, 0.0, -0.03)
    camera_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    camera_near: float = 0.02
    camera_far: float = 20.0
    max_visualize_fps: int = 60
    max_episode_steps: int | None = 1_000_000
    auto_reset: bool = False
    base_init_pos: tuple[float, float, float] | None = None


def gs_rand_float(lower, upper, shape, device):
    import torch

    return (upper - lower) * torch.rand(size=shape, device=device) + lower


@BaseRobot.register()
class DroneHoverRobot(BaseRobot):
    """Genesis drone hover eval environment."""

    robot_type = "genesis_drone"
    config_cls = DroneHoverRobotConfig

    def __init__(self, config: DroneHoverRobotConfig, logger: logging.Logger):
        if config.camera_enabled and "camera_image" not in config.image_keys:
            config.image_keys = [*config.image_keys, "camera_image"]
        super().__init__(config=config, logger=logger)
        self._config = config

        self._last_policy_obs = None
        self._last_reward = 0.0
        self._last_done = False

        # RL environment attributes (populated by init_genesis)
        self.num_envs: int = config.num_envs
        self.num_actions: int = 0
        self.num_commands: int = 0
        self.cfg: dict[str, Any] = {}
        self.env_cfg: dict[str, Any] = {}
        self.obs_cfg: dict[str, Any] = {}
        self.reward_cfg: dict[str, Any] = {}
        self.command_cfg: dict[str, Any] = {}
        self.device: Any = None
        self.simulate_action_latency: bool = True
        self.dt: float = 0.01
        self.max_episode_length: int = 0
        self.obs_scales: dict[str, float] = {}
        self.reward_scales: dict[str, float] = {}
        self.lock_commands: bool = False

        # genesis entities
        self.drone: Any = None
        self.target: Any = None

        # buffers (set in post_build)
        self.rew_buf: Any = None
        self.reset_buf: Any = None
        self.episode_length_buf: Any = None
        self.commands: Any = None
        self.actions: Any = None
        self.last_actions: Any = None
        self.base_pos: Any = None
        self.base_quat: Any = None
        self.base_lin_vel: Any = None
        self.base_ang_vel: Any = None
        self.last_base_pos: Any = None
        self.base_euler: Any = None
        self.rel_pos: Any = None
        self.last_rel_pos: Any = None
        self.base_init_pos: Any = None
        self.base_init_quat: Any = None
        self.inv_base_init_quat: Any = None
        self.extras: dict = {}
        self.obs_buf: Any = None
        self.reward_functions: dict = {}
        self.episode_sums: dict = {}
        self._being_stepped: bool = False
        self._hover_rpm: Any = None
        self.crash_condition: Any = None

        # body camera
        self._body_camera: Any = None
        self._camera_failed = False

    def post_build(self, scene: Any) -> None:
        super().post_build(scene)
        import genesis as gs
        import torch

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.lock_commands = False

        self.extras = {}

        self._being_stepped = False
        self._hover_rpm = torch.ones(self.num_envs, 4, device=gs.device) * 14468.429183500699
        self._scene.register_pre_step_callback(self._pre_step_maintain_hover)

    @property
    def policy_obs(self) -> Any:
        """Return the latest policy observation."""
        if self._last_policy_obs is None:
            self._last_policy_obs = self._get_observations()
        return self._last_policy_obs

    def set_target_position(self, target_pos: Any) -> bool:
        """Set a fixed target position command for the drone."""
        target = self._target_tensor(target_pos)
        self.lock_commands = True
        self.commands.copy_(target)
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)
        self._update_observation()
        self._last_policy_obs = self._get_observations()
        return True

    def _target_tensor(self, target_pos: Any) -> Any:
        import torch

        target = torch.as_tensor(target_pos, dtype=self.commands.dtype, device=self.commands.device)
        if target.shape == (3,):
            target = target.reshape(1, 3).expand_as(self.commands)
        if target.shape != self.commands.shape:
            raise ValueError(
                f"target_pos must have shape (3,) or {tuple(self.commands.shape)}, got {tuple(target.shape)}"
            )
        return target

    def _reset(self, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        import torch

        self.lock_commands = False
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._update_observation()
        self._last_policy_obs = self._get_observations()
        self._last_reward = 0.0
        self._last_done = False
        return self._build_observation(), {"options": options or {}}

    def _step(self, action: Any = None) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        import genesis as gs
        import torch
        from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat

        if action is None:
            action = torch.zeros(
                (self.num_envs, self.num_actions),
                dtype=gs.tc_float,
                device=gs.device,
            )

        self._being_stepped = True
        self.actions = torch.clip(action, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]).detach()
        rpm = ((1 + self.actions * 0.8) * 14468.429183500699).detach()
        self.drone.set_propellers_rpm(rpm)
        if self.target is not None:
            all_envs_idx = torch.arange(self.num_envs, device=self.device)
            self.target.set_pos(self.commands.detach(), zero_velocity=True, envs_idx=all_envs_idx)
        self._scene.step_scene()

        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        envs_idx = self._at_target()
        if not self.lock_commands:
            self._resample_commands(envs_idx)

        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        if self.env_cfg.get("auto_reset", True):
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self._update_observation()
        self.last_actions[:] = self.actions[:]
        self._being_stepped = False
        obs, reward, done, info = self._get_observations(), self.rew_buf, self.reset_buf, self.extras
        self._last_policy_obs = obs
        self._last_reward = float(reward.mean().item()) if hasattr(reward, "mean") else float(reward)
        self._last_done = bool(done.any().item()) if hasattr(done, "any") else bool(done)
        return self._build_observation(), 0.0, self._last_done, False, info

    def compute_reward(self) -> float:
        return self._last_reward

    def get_observation(self, folder: str | Path) -> dict[str, Any]:
        self._last_obs = self._build_observation()
        return super().get_observation(folder)

    def _normalize_states(self) -> dict:
        if self._last_obs is None or not isinstance(self._last_obs, dict):
            return {}
        return {key: value for key, value in self._last_obs.items() if key not in set(self._image_keys)}

    def init_genesis(self, gs_scene: Any) -> None:
        import genesis as gs
        import torch
        from genesis.utils.geom import inv_quat

        env_cfg, obs_cfg, reward_cfg, command_cfg, _ = self._load_cfgs()
        env_cfg = dict(env_cfg)
        env_cfg["visualize_target"] = self._config.visualize_target
        env_cfg["visualize_camera"] = self._config.visualize_camera and not self._config.camera_enabled
        env_cfg["max_visualize_FPS"] = int(self._config.max_visualize_fps)
        env_cfg["auto_reset"] = bool(self._config.auto_reset)
        if self._config.max_episode_steps is not None:
            env_cfg["episode_length_s"] = float(self._config.max_episode_steps) * 0.01
        if self._config.base_init_pos is not None:
            env_cfg["base_init_pos"] = list(self._config.base_init_pos)
        reward_cfg = dict(reward_cfg)
        reward_cfg["reward_scales"] = {}
        self.num_envs = self._config.num_envs
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.cfg = env_cfg
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.device = gs.device
        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        # add target sphere
        if self.env_cfg["visualize_target"]:
            self.target = gs_scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5)),
                ),
            )
        else:
            self.target = None

        # add visualize camera
        if self.env_cfg["visualize_camera"]:
            gs_scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add drone entity
        self.drone = gs_scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        # add body camera (must happen before gs_scene.build())
        if self._config.camera_enabled:
            self._add_body_camera(gs_scene)

    def _load_cfgs(self):
        with (Path(self._config.log_dir).expanduser() / "cfgs.pkl").open("rb") as file:
            return pickle.load(file)

    def _add_body_camera(self, scene: Any) -> None:
        try:
            from genesis.utils import geom as gu

            drone = scene.entities[-1]
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
            base_link = getattr(drone, "base_link", None) or drone.links[0]
            offset_T = gu.pos_lookat_up_to_T(
                np.asarray(self._config.camera_pos, dtype=np.float32),
                np.asarray(self._config.camera_lookat, dtype=np.float32),
                np.asarray(self._config.camera_up, dtype=np.float32),
            )
            camera.attach(base_link, offset_T)
            self._body_camera = camera
        except Exception as exc:  # pragma: no cover - depends on Genesis renderer/runtime
            self.logger.warning("Failed to add Genesis drone body camera: %s", exc)

    def _build_observation(self) -> dict[str, Any]:
        obs = {
            "base_pos": tensor_to_list(self.base_pos),
            "base_quat": tensor_to_list(self.base_quat),
            "base_lin_vel": tensor_to_list(self.base_lin_vel),
            "base_ang_vel": tensor_to_list(self.base_ang_vel),
            "commands": tensor_to_list(self.commands),
            "reward": self._last_reward,
            "done": self._last_done,
        }
        camera_image = self._read_camera_image()
        if camera_image is not None:
            obs["camera_image"] = camera_image
        return obs

    def _read_camera_image(self) -> Any | None:
        if self._body_camera is None or self._camera_failed:
            return None
        try:
            self._body_camera.move_to_attach()
            rgb = self._body_camera.render(rgb=True, force_render=True)[0]
            if getattr(rgb, "ndim", 0) > 3:
                rgb = rgb[0]
            return tensor_to_image_array(rgb)
        except Exception as exc:  # pragma: no cover - depends on Genesis renderer/runtime
            self._camera_failed = True
            self.logger.warning("Disabled Genesis drone camera after read failure: %s", exc)
            return None

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), self.device)

    def _at_target(self):
        import torch

        return (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )

    def _pre_step_maintain_hover(self):
        import torch

        if not self._being_stepped:
            self.drone.set_propellers_rpm(self._hover_rpm)
            if self.target is not None and self.commands is not None:
                all_envs_idx = torch.arange(self.num_envs, device=self.device)
                self.target.set_pos(self.commands.detach(), zero_velocity=True, envs_idx=all_envs_idx)
        return False

    def _update_observation(self):
        import torch

        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

    def _get_observations(self):
        from tensordict import TensorDict

        return TensorDict(
            {"policy": self.obs_buf},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def reset_idx(self, envs_idx):
        import torch

        if len(envs_idx) == 0:
            return
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0
        self._resample_commands(envs_idx)
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
