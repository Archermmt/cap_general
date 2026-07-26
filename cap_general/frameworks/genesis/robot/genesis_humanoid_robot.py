"""Genesis G1 humanoid locomotion environment."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cap_general.core.robot import BaseRobot
from cap_general.frameworks.genesis.robot.genesis_go2_robot import GenesisGo2Robot, GenesisGo2RobotConfig


_TASK_SPEEDS = {"stand": 0.0, "walk": 1.0, "run": 5.0, "hurdle": 5.0}
_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_roll_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]
_ALL_DEFAULT_JOINT_ANGLES = {
    **{name: 0.0 for name in _JOINT_NAMES},
    "left_zero_joint": 0.0, "left_one_joint": 0.0, "left_two_joint": 0.0,
    "left_three_joint": 0.0, "left_four_joint": 0.0, "left_five_joint": 0.0,
    "left_six_joint": 0.0, "right_zero_joint": 0.0, "right_one_joint": 0.0,
    "right_two_joint": 0.0, "right_three_joint": 0.0, "right_four_joint": 0.0,
    "right_five_joint": 0.0, "right_six_joint": 0.0,
}
_ALL_DEFAULT_JOINT_ANGLES["left_elbow_roll_joint"] = -1.57
_ALL_DEFAULT_JOINT_ANGLES["right_elbow_roll_joint"] = 1.57


def _default_env_cfg() -> dict[str, Any]:
    return {
        "num_actions": len(_JOINT_NAMES),
        "joint_names": list(_JOINT_NAMES),
        "default_joint_angles": dict(_ALL_DEFAULT_JOINT_ANGLES),
        "kp": 40.0,
        "kd": 1.0,
        "termination_if_roll_greater_than": 35.0,
        "termination_if_pitch_greater_than": 35.0,
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 10.0,
        "init_dof_noise": 0.05,
    }


@dataclass
class GenesisHumanoidRobotConfig(GenesisGo2RobotConfig):
    """Configuration for G1 locomotion tasks ported from HumanoidBench."""

    env_cfg: dict[str, Any] = field(default_factory=_default_env_cfg)
    reward_cfg: dict[str, Any] = field(
        default_factory=lambda: {
            "tracking_sigma": 0.25,
            "base_height_target": 0.75,
            "reward_scales": {
                "tracking_lin_vel": 2.0,
                "tracking_ang_vel": 0.2,
                "upright": 1.0,
                "lin_vel_z": -1.0,
                "base_height": -5.0,
                "body_contact": -0.2,
                "action_rate": -0.01,
                "similar_to_default": -0.05,
            },
        }
    )
    command_cfg: dict[str, Any] = field(
        default_factory=lambda: {
            "num_commands": 3,
            "lin_vel_x_range": [0.0, 1.0],
            "lin_vel_y_range": [0.0, 0.0],
            "ang_vel_range": [-0.5, 0.5],
        }
    )
    asset_path: str | Path | None = None
    task: str = "walk"
    base_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.75)
    camera_pos: tuple[float, float, float] = (0.10, 0.0, 0.65)
    camera_lookat: tuple[float, float, float] = (1.0, 0.0, 0.65)
    camera_fov: float = 70.0
    hurdle_positions: list[tuple[float, float, float]] = field(
        default_factory=lambda: [(2.0, 0.0, 0.10), (4.0, 0.0, 0.14), (6.0, 0.0, 0.18)]
    )
    hurdle_size: tuple[float, float, float] = (0.08, 1.5, 0.20)


@BaseRobot.register()
class GenesisHumanoidRobot(GenesisGo2Robot):
    """Batched Genesis G1 environment for stand, walk, run, and hurdle training."""

    robot_type = "genesis_humanoid"
    config_cls = GenesisHumanoidRobotConfig

    def __init__(self, config: GenesisHumanoidRobotConfig, logger: logging.Logger):
        self.task = self.validate_task(config.task)
        self.hurdles: list[Any] = []
        self.penalized_contact_links: Any = None
        super().__init__(config=config, logger=logger)

    def post_build(self, scene: Any) -> None:
        super().post_build(scene)
        import genesis as gs
        import torch

        self.penalized_contact_links = torch.tensor(
            [self.robot.get_link(name).idx_local for name in ("pelvis", "torso_link")],
            dtype=gs.tc_int,
            device=gs.device,
        )

    @staticmethod
    def validate_task(task: str) -> str:
        normalized = str(task).strip().lower()
        if normalized not in _TASK_SPEEDS:
            raise ValueError(f"Unknown humanoid locomotion task {task!r}; expected one of {sorted(_TASK_SPEEDS)}")
        return normalized

    @staticmethod
    def task_speed(task: str) -> float:
        return _TASK_SPEEDS[GenesisHumanoidRobot.validate_task(task)]

    @staticmethod
    def hurdle_specs(
        positions: list[tuple[float, float, float]], size: tuple[float, float, float]
    ) -> list[dict[str, tuple[float, float, float]]]:
        return [{"position": tuple(position), "size": tuple(size)} for position in positions]

    def set_locomotion_task(self, task: str) -> bool:
        self.task = self.validate_task(task)
        speed = self.task_speed(self.task)
        self.command_cfg["lin_vel_x_range"] = [speed, speed]
        self._set_hurdles_active(self.task == "hurdle")
        if self.commands is not None:
            self.commands[:, 0] = speed
            self.commands[:, 1:].zero_()
            self._update_observation()
            self._last_policy_obs = self._get_observations()
        return True

    def init_genesis(self, gs_scene: Any) -> None:
        import genesis as gs

        env_cfg = dict(self._config.env_cfg)
        if self._config.max_episode_steps is not None:
            env_cfg["episode_length_s"] = float(self._config.max_episode_steps) * 0.02
        env_cfg["base_init_pos"] = list(self._config.base_init_pos)
        self.num_envs = self._config.num_envs
        self.num_actions = int(env_cfg["num_actions"])
        self.num_commands = int(self._config.command_cfg["num_commands"])
        self.cfg = self.env_cfg = env_cfg
        self.obs_cfg = dict(self._config.obs_cfg)
        self.reward_cfg = dict(self._config.reward_cfg)
        self.command_cfg = dict(self._config.command_cfg)
        self.set_locomotion_task(self.task)
        self._train_reward_scales = dict(self.reward_cfg["reward_scales"])
        self.reward_cfg["reward_scales"] = {}
        self.device = gs.device
        self.simulate_action_latency = bool(env_cfg.get("simulate_action_latency", True))
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.obs_scales = self.obs_cfg["obs_scales"]
        self.reward_scales = self.reward_cfg["reward_scales"]

        asset_path = self._resolve_asset_path(self._config.asset_path)
        self.robot = gs_scene.add_entity(
            gs.morphs.MJCF(
                file=str(asset_path),
                pos=tuple(self._config.base_init_pos),
                quat=tuple(env_cfg["base_init_quat"]),
            )
        )
        for spec in self.hurdle_specs(self._config.hurdle_positions, self._config.hurdle_size):
            position = spec["position"] if self.task == "hurdle" else (spec["position"][0], spec["position"][1], -10.0)
            self.hurdles.append(
                gs_scene.add_entity(
                    gs.morphs.Box(size=spec["size"], pos=position, fixed=True),
                    surface=gs.surfaces.Rough(),
                )
            )
        if self._config.camera_enabled:
            self._add_body_camera(gs_scene)

    @staticmethod
    def _resolve_asset_path(configured: str | Path | None) -> Path:
        if not configured:
            raise FileNotFoundError("Set robot.asset_path to the G1 MJCF asset")
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"G1 MJCF asset does not exist: {path}")
        return path.resolve()

    def _reward_upright(self):
        import torch

        return torch.clamp(-self.projected_gravity[:, 2], 0.0, 1.0)

    def _reward_body_contact(self):
        import torch

        forces = self.robot.get_links_net_contact_force()[:, self.penalized_contact_links]
        return torch.any(torch.linalg.vector_norm(forces, dim=-1) > 1.0, dim=1).float()

    def _set_hurdles_active(self, active: bool) -> None:
        if not self.hurdles:
            return
        for hurdle, position in zip(self.hurdles, self._config.hurdle_positions, strict=True):
            target = position if active else (position[0], position[1], -10.0)
            hurdle.set_pos(target)

    def _reset_idx(self, envs_idx=None):
        super()._reset_idx(envs_idx)
        if not self._training:
            return
        noise_scale = float(self.env_cfg.get("init_dof_noise", 0.0))
        if noise_scale <= 0.0:
            return

        import torch

        if envs_idx is None:
            envs_idx_int = None
            count = self.num_envs
        elif envs_idx.dtype == torch.bool:
            envs_idx_int = envs_idx.nonzero(as_tuple=False).reshape(-1)
            count = len(envs_idx_int)
        else:
            envs_idx_int = envs_idx
            count = len(envs_idx_int)
        if count == 0:
            return
        targets = self.default_dof_pos.repeat(count, 1)
        targets += (2.0 * torch.rand_like(targets) - 1.0) * noise_scale
        self.robot.set_dofs_position(
            targets,
            dofs_idx_local=self.motors_dof_idx,
            envs_idx=envs_idx_int,
            zero_velocity=True,
        )
