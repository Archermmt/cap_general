"""Local PyRoKi motion planning model implementation."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from cap_general.core.policy.base_policy import PolicyBase


@dataclass
class PyrokiIkResult:
    """PyRoKi IK result."""

    joint_positions: list[float]


@dataclass
class PyrokiPlanResult:
    """PyRoKi trajectory result."""

    waypoints: list[list[float]]
    dt: float


@dataclass
class PyrokiPolicyConfig:
    """Configuration for PyrokiPolicy."""

    robot_urdf_name: str = "panda_description"
    target_link_name: str = "panda_hand"
    sphere_decomposition_path: str | Path | None = None
    min_distance_from_limits: float = 0.15


@PolicyBase.register()
class PyrokiPolicy(PolicyBase):
    """Local PyRoKi IK and simple trajectory planning model."""

    name = "PyRoKi Policy Model"
    config_cls = PyrokiPolicyConfig

    def __init__(
        self,
        robot_urdf_name: str = "panda_description",
        target_link_name: str = "panda_hand",
        sphere_decomposition_path: str | Path | None = None,
        min_distance_from_limits: float = 0.15,
        config: PyrokiPolicyConfig | None = None,
    ):
        config = config or PyrokiPolicyConfig(
            robot_urdf_name=robot_urdf_name,
            target_link_name=target_link_name,
            sphere_decomposition_path=sphere_decomposition_path,
            min_distance_from_limits=min_distance_from_limits,
        )
        self._robot_urdf_name = config.robot_urdf_name
        self._target_link_name = config.target_link_name
        self._sphere_decomposition_path = config.sphere_decomposition_path
        self._min_distance_from_limits = config.min_distance_from_limits
        self._robot = None
        self._robot_collision = None
        self._pk = None
        self._pks = None

    @classmethod
    def policy_type(cls) -> str:
        return "pyroki"

    def _load_model(self):
        """Lazily load PyRoKi robot locally."""
        if self._robot is not None:
            return

        try:
            import pyroki as pk
            from robot_descriptions.loaders.yourdfpy import load_robot_description
            import capx.integrations.motion.pyroki_snippets as pks
        except ImportError as exc:
            raise ImportError(
                "PyrokiPolicy requires pyroki, robot_descriptions, and "
                "capx.integrations.motion.pyroki_snippets."
            ) from exc

        urdf = load_robot_description(self._robot_urdf_name)
        self._set_min_distance_from_limits(urdf, self._min_distance_from_limits)
        self._robot = pk.Robot.from_urdf(urdf)
        self._robot_collision = self._build_robot_collision(pk, urdf)
        self._pk = pk
        self._pks = pks

    def _build_robot_collision(self, pk, urdf):
        if self._sphere_decomposition_path is None:
            return None
        with Path(self._sphere_decomposition_path).open() as file:
            sphere_decomposition = json.load(file)
        return pk.collision.RobotCollision.from_sphere_decomposition(
            sphere_decomposition=sphere_decomposition,
            urdf=urdf,
        )

    @staticmethod
    def _set_min_distance_from_limits(urdf, min_distance_from_limits: float):
        for joint in urdf.robot.joints:
            if joint.type == "revolute" and joint.limit is not None:
                if joint.limit.lower is not None and joint.limit.upper is not None:
                    joint.limit.lower += min_distance_from_limits
                    joint.limit.upper -= min_distance_from_limits

    def solve_ik(
        self,
        target_pose_wxyz_xyz: list[float] | np.ndarray,
        prev_cfg: list[float] | np.ndarray | None = None,
    ) -> PyrokiIkResult:
        """Solve local inverse kinematics."""
        self._load_model()
        pose = np.asarray(target_pose_wxyz_xyz, dtype=np.float64)
        prev = np.asarray(prev_cfg, dtype=np.float64) if prev_cfg is not None else None
        if prev is None:
            joints = self._pks.solve_ik(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=pose[-3:],
                target_wxyz=pose[:-3],
            )
        else:
            joints = self._pks.solve_ik_vel_cost(
                robot=self._robot,
                target_link_name=self._target_link_name,
                target_position=pose[-3:],
                target_wxyz=pose[:-3],
                prev_cfg=prev,
            )
        return PyrokiIkResult(joint_positions=list(map(float, joints)))

    def plan(
        self,
        start_pose_wxyz_xyz: list[float] | np.ndarray,
        end_pose_wxyz_xyz: list[float] | np.ndarray,
        timesteps: int = 20,
        dt: float = 0.02,
    ) -> PyrokiPlanResult:
        """Plan a local linear-interpolation IK trajectory."""
        self._load_model()
        start = np.asarray(start_pose_wxyz_xyz, dtype=np.float64)
        end = np.asarray(end_pose_wxyz_xyz, dtype=np.float64)
        trajectory = self._plan_trajectory_linear_ik(
            start_pos=start[4:],
            start_wxyz=start[:4],
            end_pos=end[4:],
            end_wxyz=end[:4],
            num_waypoints=timesteps,
        )
        return PyrokiPlanResult(waypoints=np.asarray(trajectory).tolist(), dt=float(dt))

    def _plan_trajectory_linear_ik(
        self,
        start_pos: np.ndarray,
        start_wxyz: np.ndarray,
        end_pos: np.ndarray,
        end_wxyz: np.ndarray,
        num_waypoints: int,
    ) -> np.ndarray:
        positions = np.linspace(start_pos, end_pos, num_waypoints)
        orientations = self._slerp_quaternions(start_wxyz, end_wxyz, num_waypoints)
        trajectory = []
        prev_cfg = None
        for pos, wxyz in zip(positions, orientations):
            if prev_cfg is None:
                result = self.solve_ik(np.concatenate([wxyz, pos]))
            else:
                result = self.solve_ik(np.concatenate([wxyz, pos]), prev_cfg=prev_cfg)
            prev_cfg = np.asarray(result.joint_positions)
            trajectory.append(prev_cfg)
        return np.asarray(trajectory)

    @staticmethod
    def _slerp_quaternions(
        q_start: np.ndarray,
        q_end: np.ndarray,
        num_steps: int,
    ) -> np.ndarray:
        from scipy.spatial.transform import Rotation, Slerp

        r_start = Rotation.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]])
        r_end = Rotation.from_quat([q_end[1], q_end[2], q_end[3], q_end[0]])
        slerp = Slerp([0, 1], Rotation.concatenate([r_start, r_end]))
        quats_xyzw = slerp(np.linspace(0, 1, num_steps)).as_quat()
        return np.column_stack(
            [quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]]
        )

    def inference(self, mode: str = "ik", **kwargs: Any) -> PyrokiIkResult | PyrokiPlanResult:
        """Run local PyRoKi inference."""
        if mode == "ik":
            return self.solve_ik(**kwargs)
        if mode == "plan":
            return self.plan(**kwargs)
        raise ValueError(f"Unsupported PyRoKi inference mode: {mode}")

    @property
    def policy_name(self) -> str:
        return "PyrokiPolicy"
