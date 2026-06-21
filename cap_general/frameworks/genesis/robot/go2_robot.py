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


def _load_genesis_deps():
    global gs, inv_quat, math, quat_to_xyz, torch, TensorDict

    import math

    import genesis as gs
    import torch
    from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat
    from tensordict import TensorDict

    globals()["transform_by_quat"] = transform_by_quat
    globals()["transform_quat_by_quat"] = transform_quat_by_quat
    return gs


@dataclass
class Go2RobotConfig(BaseRobotConfig):
    """Configuration for the Genesis GO2 locomotion example."""

    example_root: str | Path = "/Users/tongmeng/Desktop/codes/genesis-world/examples/locomotion"
    log_dir: str | Path = "logs/go2-walking"
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


@BaseRobot.register()
class Go2Robot(BaseRobot):
    """Genesis GO2 locomotion eval environment."""

    name = "Genesis GO2 Robot"
    config_cls = Go2RobotConfig

    def __init__(self, config: Go2RobotConfig, logger: logging.Logger | None = None):
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
    def robot_type(cls) -> str:
        return "genesis_go2_robot"

    @property
    def example_env(self) -> Any:
        """Return the underlying genesis-world Go2Robot."""
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
            self.logger.warning("Genesis GO2 env running in mock mode: %s", self._mock_reason)
            return

        try:
            scene_resource = self.cap_scene.get_resource("genesis_scene") if self.cap_scene is not None else None
            scene = getattr(scene_resource, "scene", None)
            if scene is None:
                self._mock_reason = "genesis scene resource is not enabled or failed"
                self.logger.warning("Genesis GO2 env running in mock mode: %s", self._mock_reason)
                return
            env_cfg, obs_cfg, reward_cfg, command_cfg, _train_cfg = self._load_cfgs()
            env_cfg = dict(env_cfg)
            if self._config.max_episode_steps is not None:
                env_cfg["episode_length_s"] = float(self._config.max_episode_steps) * 0.02
            env_cfg["_scene"] = scene
            reward_cfg = dict(reward_cfg)
            reward_cfg["reward_scales"] = {}
            self._example_env = self._build_example_env_with_camera(
                num_envs=self._config.num_envs,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
            )
            if hasattr(self._example_env, "get_observations"):
                self._last_policy_obs = self._example_env.get_observations()
        except Exception as exc:  # pragma: no cover - depends on Genesis runtime
            self._mock_reason = str(exc)
            self.logger.warning("Genesis GO2 env running in mock mode: %s", exc)

    def _build_example_env_with_camera(self, **kwargs: Any) -> Any:
        env_cfg = dict(kwargs["env_cfg"])
        if not self._config.camera_enabled:
            kwargs["env_cfg"] = env_cfg
            return _GenesisGo2CoreRobot(**kwargs)

        camera_holder: dict[str, Any] = {}
        env_cfg["_before_scene_build"] = lambda scene: self._add_body_camera(scene, camera_holder)
        kwargs["env_cfg"] = env_cfg
        example_env = _GenesisGo2CoreRobot(**kwargs)
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

# Embedded genesis-world go2 env implementation.
def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


class _GenesisGo2CoreRobot:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg):
        self.num_envs: int = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]
        self.reward_scales: dict[str, float] = reward_cfg["reward_scales"]

        # use scene owned by the top-level CAP scene resource
        self.scene = env_cfg["_scene"]

        # add plain
        self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )

        # add robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )

        # build
        before_scene_build = env_cfg.get("_before_scene_build")
        if before_scene_build is not None:
            before_scene_build(self.scene)
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # Define global gravity direction vector
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        # Initial state
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

        # initialize buffers
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
        self.commands_limits: tuple[torch.Tensor, torch.Tensor] = tuple(
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
        self.extras = dict()  # extra information for logging

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        self.reset()

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self.scene.step()

        # update buffers
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

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # resample commands
        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()

        # Compute timeout
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        # Reset environment if necessary
        self._reset_idx(self.reset_buf)

        # update observations
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        # reset state
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        # reset buffers
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
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # fill extras
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

        # random sample command upon reset
        self._resample_commands(envs_idx)

    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ),
            dim=-1,
        )

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
