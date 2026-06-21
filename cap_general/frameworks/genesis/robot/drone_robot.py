"""CAP wrapper for Genesis drone hover evaluation."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cap_general.core.robot import BaseRobot, BaseRobotConfig


def _load_genesis_deps():
    global copy, gs, inv_quat, math, quat_to_xyz, torch, TensorDict

    import copy
    import math

    import genesis as gs
    import torch
    from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat
    from tensordict import TensorDict

    globals()["transform_by_quat"] = transform_by_quat
    globals()["transform_quat_by_quat"] = transform_quat_by_quat
    return gs


@dataclass
class DroneHoverRobotConfig(BaseRobotConfig):
    """Configuration for the Genesis drone hover example."""

    example_root: str | Path = "/Users/tongmeng/Desktop/codes/genesis-world/examples/drone"
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


@BaseRobot.register()
class DroneHoverRobot(BaseRobot):
    """Genesis drone hover eval environment."""

    name = "Genesis Drone Hover Robot"
    config_cls = DroneHoverRobotConfig

    def __init__(self, config: DroneHoverRobotConfig, logger: logging.Logger | None = None):
        if config.camera_enabled and "camera_image" not in config.image_keys:
            config.image_keys = [*config.image_keys, "camera_image"]
        super().__init__(config=config, logger=logger)
        self._config = config
        self._example_env = None
        self._last_policy_obs = None
        self._last_reward = 0.0
        self._last_done = False
        self._mock_reason: str | None = None
        self._body_camera = None
        self._camera_failed = False

    @classmethod
    def robot_type(cls) -> str:
        return "genesis_drone_hover_robot"

    @property
    def example_env(self) -> Any:
        """Return the underlying genesis-world HoverRobot."""
        self._ensure_example_env()
        return self._example_env

    @property
    def policy_obs(self) -> Any:
        """Return the latest policy observation."""
        if self._last_policy_obs is None and self._example_env is not None:
            if not getattr(self._example_env, "_deferred_build", False) and hasattr(self._example_env, "get_observations"):
                self._last_policy_obs = self._example_env.get_observations()
        return self._last_policy_obs

    @property
    def dt(self) -> float:
        """Return the drone policy control period in seconds."""
        env = self._example_env
        return float(getattr(env, "dt", 1.0)) if env is not None else 1.0

    def unlock_target(self) -> bool:
        """Allow the underlying hover env to resample target commands."""
        self._ensure_example_env()
        if self._example_env is None:
            return False
        self._example_env.lock_commands = False
        return True

    def target_locked(self) -> bool:
        """Return whether the drone currently has a fixed target command."""
        self._ensure_example_env()
        return bool(self._example_env is not None and getattr(self._example_env, "lock_commands", False))

    def hold_current_position(self) -> bool:
        """Set the current drone position as the hover target command."""
        self._ensure_example_env()
        env = self._example_env
        if env is None:
            return False
        return self.set_target_position(env.base_pos)

    def set_target_position(self, target_pos: Any) -> bool:
        """Set a fixed target position command for the drone."""
        self._ensure_example_env()
        env = self._example_env
        if env is None:
            return False
        target = self._target_tensor(target_pos)
        env.lock_commands = True
        env.commands.copy_(target)
        env.rel_pos = env.commands - env.base_pos
        env.last_rel_pos = env.commands - env.last_base_pos
        if getattr(env, "target", None) is not None:
            env.target.set_pos(env.commands, zero_velocity=True)
        if hasattr(env, "_update_observation"):
            env._update_observation()
        if hasattr(env, "get_observations"):
            self._last_policy_obs = env.get_observations()
        return True

    def _target_tensor(self, target_pos: Any) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Setting drone targets requires torch") from exc

        env = self._example_env
        target = torch.as_tensor(target_pos, dtype=env.commands.dtype, device=env.commands.device)
        if target.shape == (3,):
            target = target.reshape(1, 3).expand_as(env.commands)
        if target.shape != env.commands.shape:
            raise ValueError(f"target_pos must have shape (3,) or {tuple(env.commands.shape)}, got {tuple(target.shape)}")
        return target

    def _reset(self, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._ensure_example_env()
        if self._example_env is None:
            obs = self._mock_observation()
            return obs, {"mock": True, "reason": self._mock_reason}
        if getattr(self._example_env, "_deferred_build", False):
            obs = self._mock_observation()
            return obs, {"mock": False, "pending_build": True, "options": options or {}}
        self._example_env.lock_commands = False
        self._last_policy_obs = self._example_env.reset()
        self._last_reward = 0.0
        self._last_done = False
        return self._build_observation(), {"mock": False, "options": options or {}}

    def _step(self, action: Any = None) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._ensure_example_env()
        if self._example_env is None:
            return self._mock_observation(), 0.0, False, False, {"mock": True}

        if action is None:
            action = self._zero_action()
        obs, reward, done, info = self._example_env.step(action)
        self._last_policy_obs = obs
        self._last_reward = float(reward.mean().item()) if hasattr(reward, "mean") else float(reward)
        self._last_done = bool(done.any().item()) if hasattr(done, "any") else bool(done)
        return self._build_observation(), 0.0, self._last_done, False, info

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
            gs = _load_genesis_deps()
        except ImportError as exc:
            self._mock_reason = f"genesis import failed: {exc}"
            self.logger.warning("Genesis drone env running in mock mode: %s", self._mock_reason)
            return

        try:
            scene_resource = self.cap_scene.get_resource("genesis_scene") if self.cap_scene is not None else None
            scene = getattr(scene_resource, "scene", None)
            if scene is None:
                self._mock_reason = "genesis scene resource is not enabled or failed"
                self.logger.warning("Genesis drone env running in mock mode: %s", self._mock_reason)
                return
            env_cfg, obs_cfg, reward_cfg, command_cfg, _train_cfg = self._load_cfgs()
            env_cfg = dict(env_cfg)
            env_cfg["visualize_target"] = self._config.visualize_target
            env_cfg["visualize_camera"] = self._config.visualize_camera and not self._config.camera_enabled
            env_cfg["max_visualize_FPS"] = int(self._config.max_visualize_fps)
            env_cfg["auto_reset"] = bool(self._config.auto_reset)
            if self._config.max_episode_steps is not None:
                env_cfg["episode_length_s"] = float(self._config.max_episode_steps) * 0.01
            env_cfg["_scene"] = scene
            env_cfg["_scene_resource"] = scene_resource
            reward_cfg = dict(reward_cfg)
            reward_cfg["reward_scales"] = {}
            self._example_env = self._build_example_env_with_camera(
                num_envs=self._config.num_envs,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
            )
            if not getattr(self._example_env, "_deferred_build", False):
                self._last_policy_obs = self._example_env.get_observations()
        except Exception as exc:  # pragma: no cover - depends on Genesis runtime
            self._mock_reason = str(exc)
            self.logger.warning("Genesis drone env running in mock mode: %s", exc)

    def _build_example_env_with_camera(self, **kwargs: Any) -> Any:
        env_cfg = dict(kwargs["env_cfg"])
        if not self._config.camera_enabled:
            kwargs["env_cfg"] = env_cfg
            return _GenesisDroneHoverCoreRobot(**kwargs)

        camera_holder: dict[str, Any] = {}
        env_cfg["_before_scene_build"] = lambda scene: self._add_body_camera(scene, camera_holder)
        kwargs["env_cfg"] = env_cfg
        example_env = _GenesisDroneHoverCoreRobot(**kwargs)
        self._body_camera = camera_holder.get("camera")
        return example_env

    def _add_body_camera(self, scene: Any, camera_holder: dict[str, Any]) -> None:
        if camera_holder.get("camera") is not None:
            return
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
            camera_holder["camera"] = camera
        except Exception as exc:  # pragma: no cover - depends on Genesis renderer/runtime
            self.logger.warning("Failed to add Genesis drone body camera: %s", exc)

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
            "base_lin_vel": self._to_list(getattr(env, "base_lin_vel", None)),
            "base_ang_vel": self._to_list(getattr(env, "base_ang_vel", None)),
            "commands": self._to_list(getattr(env, "commands", None)),
            "reward": self._last_reward,
            "done": self._last_done,
            "mock": False,
        }
        camera_image = self._read_camera_image()
        if camera_image is not None:
            obs["camera_image"] = camera_image
        return obs

    def _mock_observation(self) -> dict[str, Any]:
        return {
            "base_pos": [0.0, 0.0, 1.0],
            "base_quat": [1.0, 0.0, 0.0, 0.0],
            "base_lin_vel": [0.0, 0.0, 0.0],
            "base_ang_vel": [0.0, 0.0, 0.0],
            "commands": [],
            "camera_image": None,
            "reward": self._last_reward,
            "done": self._last_done,
            "mock": True,
            "reason": self._mock_reason,
        }

    def _read_camera_image(self) -> Any | None:
        if self._body_camera is None or self._camera_failed:
            return None
        try:
            self._body_camera.move_to_attach()
            rgb = self._body_camera.render(rgb=True, force_render=True)[0]
            if getattr(rgb, "ndim", 0) > 3:
                rgb = rgb[0]
            return self._to_image_array(rgb)
        except Exception as exc:  # pragma: no cover - depends on Genesis renderer/runtime
            self._camera_failed = True
            self.logger.warning("Disabled Genesis drone camera after read failure: %s", exc)
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

# Embedded genesis-world drone env implementation.
def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class _GenesisDroneHoverCoreRobot:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg):
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self._deferred_build = False

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        # use scene owned by the top-level CAP scene resource
        self.scene = env_cfg["_scene"]

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add target
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
        else:
            self.target = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add drone
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        # build scene
        before_scene_build = env_cfg.get("_before_scene_build")
        if before_scene_build is not None:
            before_scene_build(self.scene)
        scene_resource = env_cfg.get("_scene_resource")
        build_kwargs = {"n_envs": num_envs}
        if scene_resource is not None and scene_resource.defer_build(build_kwargs, self._post_build):
            self._deferred_build = True
            return
        self.scene.build(**build_kwargs)
        self._post_build()

    def _post_build(self) -> None:
        self._deferred_build = False
        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
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

        self.extras = dict()  # extra information for logging

        self.reset()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), gs.device)

    def _at_target(self):
        return (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.actions

        # 14468 is hover rpm
        self.drone.set_propellers_rpm((1 + exec_actions * 0.8) * 14468.429183500699)
        # update target pos
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)
        self.scene.step()

        # update buffers
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

        # resample commands
        envs_idx = self._at_target()
        if not self.lock_commands:
            self._resample_commands(envs_idx)

        # check termination
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

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self._update_observation()

        self.last_actions[:] = self.actions[:]

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def _update_observation(self):
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

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        self._update_observation()
        return self.get_observations()

    # ------------ reward functions----------------
    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        return target_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # use rad for yaw_reward
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return yaw_rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew
