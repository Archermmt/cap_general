"""PyRoKi model operator."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cap_general.core.operator.base_operator import BaseOperator, to_stage_fn
from cap_general.core.operator.model.base_model_op import ModelOp


@dataclass
class PyrokiIkResult:
    joint_positions: list[float]


@dataclass
class PyrokiPlanResult:
    waypoints: list[list[float]]
    dt: float


@dataclass
class PyrokiConfig:
    robot_urdf_name: str = "panda_description"
    target_link_name: str = "panda_hand"
    sphere_decomposition_path: str | Path | None = None
    min_distance_from_limits: float = 0.15


@BaseOperator.register()
class PyrokiOp(ModelOp):
    """Local PyRoKi IK and simple trajectory planning model."""

    op_type = "pyroki"
    config_cls = PyrokiConfig

    def reset(self) -> None:
        self._robot = None
        self._robot_collision = None
        self._pk = None
        self._pks = None
        super().reset()

    def _load_robot(self) -> None:
        if self._robot is not None:
            return
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        try:
            import capx.integrations.motion.pyroki_snippets as pks
            import pyroki as pk
            from robot_descriptions.loaders.yourdfpy import load_robot_description
        except ImportError as exc:
            raise ImportError(
                "PyrokiOp requires pyroki, robot_descriptions, and capx.integrations.motion.pyroki_snippets."
            ) from exc
        urdf = load_robot_description(self._config.robot_urdf_name)
        self._set_min_distance_from_limits(urdf, self._config.min_distance_from_limits)
        self._robot = pk.Robot.from_urdf(urdf)
        self._robot_collision = self._build_robot_collision(pk, urdf)
        self._pk = pk
        self._pks = pks
        logging.getLogger("jaxls").setLevel(logging.WARNING)

    def _build_robot_collision(self, pk, urdf):
        if self._config.sphere_decomposition_path is None:
            return None
        with Path(self._config.sphere_decomposition_path).open() as file:
            sphere_decomposition = json.load(file)
        return pk.collision.RobotCollision.from_sphere_decomposition(
            sphere_decomposition=sphere_decomposition, urdf=urdf
        )

    @staticmethod
    def _set_min_distance_from_limits(urdf, min_distance_from_limits: float):
        for joint in urdf.robot.joints:
            if joint.type == "revolute" and joint.limit is not None:
                if joint.limit.lower is not None and joint.limit.upper is not None:
                    joint.limit.lower += min_distance_from_limits
                    joint.limit.upper -= min_distance_from_limits

    @to_stage_fn
    def inference(self, inputs: dict[str, Any]) -> dict[str, Any]:
        mode = inputs.get("mode", "ik")
        if mode == "ik":
            return self.solve_ik(inputs)
        if mode == "plan":
            return self.plan(inputs)
        raise ValueError(f"Unsupported PyRoKi inference mode: {mode!r}")

    @to_stage_fn
    def solve_ik(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self._load_robot()
        pose = np.asarray(inputs["target_pose_wxyz_xyz"], dtype=np.float64)
        prev_cfg = inputs.get("prev_cfg")
        prev = np.asarray(prev_cfg, dtype=np.float64) if prev_cfg is not None else None
        if prev is None:
            joints = self._pks.solve_ik(
                robot=self._robot,
                target_link_name=self._config.target_link_name,
                target_position=pose[-3:],
                target_wxyz=pose[:-3],
            )
        else:
            joints = self._pks.solve_ik_vel_cost(
                robot=self._robot,
                target_link_name=self._config.target_link_name,
                target_position=pose[-3:],
                target_wxyz=pose[:-3],
                prev_cfg=prev,
            )
        return {"output": PyrokiIkResult(joint_positions=list(map(float, joints)))}

    @to_stage_fn
    def plan(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self._load_robot()
        start = np.asarray(inputs["start_pose_wxyz_xyz"], dtype=np.float64)
        end = np.asarray(inputs["end_pose_wxyz_xyz"], dtype=np.float64)
        timesteps = inputs.get("timesteps", 20)
        dt = inputs.get("dt", 0.02)
        trajectory = self._plan_trajectory_linear_ik(
            start_pos=start[4:],
            start_wxyz=start[:4],
            end_pos=end[4:],
            end_wxyz=end[:4],
            num_waypoints=timesteps,
        )
        return {"output": PyrokiPlanResult(waypoints=np.asarray(trajectory).tolist(), dt=float(dt))}

    def _plan_trajectory_linear_ik(self, start_pos, start_wxyz, end_pos, end_wxyz, num_waypoints) -> np.ndarray:
        positions = np.linspace(start_pos, end_pos, num_waypoints)
        orientations = self._slerp_quaternions(start_wxyz, end_wxyz, num_waypoints)
        trajectory = []
        prev_cfg = None
        for pos, wxyz in zip(positions, orientations):
            inp = {"target_pose_wxyz_xyz": np.concatenate([wxyz, pos])}
            if prev_cfg is not None:
                inp["prev_cfg"] = prev_cfg
            result = self.solve_ik(inp)
            prev_cfg = np.asarray(result["output"].joint_positions)
            trajectory.append(prev_cfg)
        return np.asarray(trajectory)

    @staticmethod
    def _slerp_quaternions(q_start: np.ndarray, q_end: np.ndarray, num_steps: int) -> np.ndarray:
        from scipy.spatial.transform import Rotation, Slerp

        r_start = Rotation.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]])
        r_end = Rotation.from_quat([q_end[1], q_end[2], q_end[3], q_end[0]])
        slerp = Slerp([0, 1], Rotation.concatenate([r_start, r_end]))
        quats_xyzw = slerp(np.linspace(0, 1, num_steps)).as_quat()
        return np.column_stack([quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]])
