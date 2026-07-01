"""CAP wrapper for Genesis GO2 locomotion evaluation."""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cap_general.core.robot import BaseRobot, BaseRobotConfig
from cap_general.core.utils import tensor_to_image_array, tensor_to_list


@dataclass
class Go2RobotConfig(BaseRobotConfig):
    """Configuration for the Genesis GO2 locomotion example."""

    example_root: str | Path = "/Users/archer/Desktop/codes/genesis-world/examples/locomotion"
    log_dir: str | Path = "logs/go2-walking"
    num_envs: int = 1
    image_keys: list[str] = field(default_factory=lambda: ["body_camera_image"])
    camera_enabled: bool = True
    camera_res: tuple[int, int] = (320, 240)
    camera_fov: float = 40.0
    camera_pos: tuple[float, float, float] = (2.0, 0.0, 2.5)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.5)
    camera_attach_to_base: bool = False
    camera_near: float = 0.05
    camera_far: float = 20.0
    turn_action_scale: float = 0.35
    max_episode_steps: int | None = 1_000_000
    base_init_pos: tuple[float, float, float] | None = None


def gs_rand(lower, upper, batch_shape):
    import genesis as gs
    import torch

    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


@BaseRobot.register()
class Go2Robot(BaseRobot):
    """Genesis GO2 locomotion eval environment."""

    robot_type = "genesis_go2"
    config_cls = Go2RobotConfig

    def __init__(self, config: Go2RobotConfig, logger: logging.Logger):
        super().__init__(config=config, logger=logger)
        self._config = config

        # policy / episode state
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
        self.dt: float = 0.02
        self.max_episode_length: int = 0
        self.obs_scales: dict[str, float] = {}
        self.reward_scales: dict[str, float] = {}

        # genesis entities
        self.robot: Any = None

        # buffers (set in post_build)
        self.base_lin_vel: Any = None
        self.base_ang_vel: Any = None
        self.projected_gravity: Any = None
        self.rew_buf: Any = None
        self.reset_buf: Any = None
        self.episode_length_buf: Any = None
        self.commands: Any = None
        self.commands_scale: Any = None
        self.commands_limits: Any = None
        self.actions: Any = None
        self.last_actions: Any = None
        self.dof_pos: Any = None
        self.dof_vel: Any = None
        self.last_dof_vel: Any = None
        self.base_pos: Any = None
        self.base_quat: Any = None
        self.base_euler: Any = None
        self.default_dof_pos: Any = None
        self.extras: dict = {}
        self.obs_buf: Any = None
        self.motors_dof_idx: Any = None
        self.actions_dof_idx: Any = None
        self.reward_functions: dict = {}
        self.episode_sums: dict = {}

        # body camera
        self._body_camera: Any = None
        self._body_camera_failed = False

    def post_build(self, scene: Any) -> None:
        super().post_build(scene)
        import genesis as gs
        import torch
        from genesis.utils.geom import inv_quat, transform_by_quat

        robot_dof_start = self.robot.dof_start
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start - robot_dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.empty((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.commands_limits = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.extras = {}

        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

    @property
    def policy_obs(self) -> Any:
        """Return the latest policy observation."""
        if self._last_policy_obs is None:
            self._last_policy_obs = self._get_observations()
        return self._last_policy_obs

    def set_walk_command(
        self,
        *,
        turn_angle: float = 0.0,
        steps: int | None = None,
        forward_speed: float | None = None,
    ) -> bool:
        """Set the GO2 forward walking command."""
        if forward_speed is not None:
            self.commands[:, 0] = float(forward_speed)
        self.commands[:, 1] = 0.0
        if steps and steps > 0:
            self.commands[:, 2] = float(turn_angle) / max(float(steps) * self.dt, 1e-6)
        else:
            self.commands[:, 2] = 0.0
        self._update_observation()
        self._last_policy_obs = self._get_observations()
        return True

    def stop_command(self) -> bool:
        """Set all GO2 velocity commands to zero."""
        self.commands.zero_()
        self._update_observation()
        self._last_policy_obs = self._get_observations()
        return True

    def apply_turn_to_action(self, action: Any, turn_angle: float) -> Any:
        """Bias policy action so GO2 walks while turning smoothly."""
        import torch

        if not turn_angle:
            return action

        joint_names = list(self.env_cfg.get("joint_names", []))
        if not joint_names:
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
        self._reset_idx()
        self._update_observation()
        self._last_policy_obs = self._get_observations()
        self._last_reward = 0.0
        self._last_done = False
        return self._build_observation(), {"mock": False, "options": options or {}}

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

        self.actions = torch.clip(action, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self._scene.step_scene()

        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self._scene.gs_scene.rigid_solver.get_error_envs_mask()
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        self._reset_idx(self.reset_buf)
        self._update_observation()
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

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

        env_cfg, obs_cfg, reward_cfg, command_cfg, _ = self._load_cfgs()
        env_cfg = dict(env_cfg)
        if self._config.max_episode_steps is not None:
            env_cfg["episode_length_s"] = float(self._config.max_episode_steps) * 0.02
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
        self.simulate_action_latency = True
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.robot = gs_scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )

        if self._config.camera_enabled:
            self._add_body_camera(gs_scene)

    def _load_cfgs(self):
        with (Path(self._config.log_dir).expanduser() / "cfgs.pkl").open("rb") as file:
            return pickle.load(file)

    def _add_body_camera(self, scene: Any) -> None:
        try:
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
            if self._config.camera_attach_to_base:
                from genesis.utils import geom as gu

                base_link = getattr(robot, "base_link", None) or robot.links[0]
                offset_T = gu.pos_lookat_up_to_T(
                    np.asarray(self._config.camera_pos, dtype=np.float32),
                    np.asarray(self._config.camera_lookat, dtype=np.float32),
                    np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
                )
                camera.attach(base_link, offset_T)
            self._body_camera = camera
        except Exception as exc:  # pragma: no cover - depends on Genesis renderer/runtime
            self.logger.warning("Failed to add GO2 body camera: %s", exc)

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
        body_camera_image = self._read_body_camera_image()
        if body_camera_image is not None:
            obs["body_camera_image"] = body_camera_image
        return obs

    def _read_body_camera_image(self) -> Any | None:
        if self._body_camera is None or self._body_camera_failed:
            return None
        try:
            if self._config.camera_attach_to_base:
                self._body_camera.move_to_attach()
            rgb = self._body_camera.render(rgb=True, force_render=True)[0]
            if getattr(rgb, "ndim", 0) > 3:
                rgb = rgb[0]
            return tensor_to_image_array(rgb)
        except Exception as exc:  # pragma: no cover - depends on Genesis renderer/runtime
            self._body_camera_failed = True
            self.logger.warning("Disabled GO2 body camera after read failure: %s", exc)
            return None

    def _resample_commands(self, envs_idx):
        import torch

        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def _get_observations(self) -> Any:
        from tensordict import TensorDict

        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        import torch

        if envs_idx is not None and envs_idx.dtype == torch.bool:
            envs_idx_bool = envs_idx
            envs_idx_int = envs_idx.nonzero(as_tuple=False).reshape(-1)
        else:
            envs_idx_bool = None
            envs_idx_int = envs_idx

        if envs_idx_int is not None and len(envs_idx_int) == 0:
            return

        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx_int, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            mask = envs_idx_bool
            if mask is None:
                mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.base_pos.device)
                mask[envs_idx_int] = True
            torch.where(mask[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(mask[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(mask[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity)
            torch.where(mask[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(mask[:, None], 0.0)
            self.base_ang_vel.masked_fill_(mask[:, None], 0.0)
            self.dof_vel.masked_fill_(mask[:, None], 0.0)
            self.actions.masked_fill_(mask[:, None], 0.0)
            self.last_actions.masked_fill_(mask[:, None], 0.0)
            self.last_dof_vel.masked_fill_(mask[:, None], 0.0)
            self.episode_length_buf.masked_fill_(mask, 0)
            self.reset_buf.masked_fill_(mask, True)

        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            elif envs_idx_bool is not None:
                n_envs = envs_idx_bool.sum()
                mean = value[envs_idx_bool].sum() / n_envs if n_envs > 0 else value.new_tensor(0.0)
            else:
                mean = value[envs_idx_int].mean() if len(envs_idx_int) > 0 else value.new_tensor(0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            elif envs_idx_bool is not None:
                value.masked_fill_(envs_idx_bool, 0.0)
            else:
                value[envs_idx_int] = 0.0

        if envs_idx_bool is not None:
            self._resample_commands(envs_idx_bool)
        elif envs_idx_int is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.base_pos.device)
            mask[envs_idx_int] = True
            self._resample_commands(mask)
        else:
            self._resample_commands(None)

    def _update_observation(self):
        import torch

        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
            ),
            dim=-1,
        )
