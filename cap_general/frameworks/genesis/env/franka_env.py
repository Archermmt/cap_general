"""Genesis Franka environment controller."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np

from cap_general.core.env import BaseEnv, BaseEnvConfig

_DEFAULT_COLORS = [
    (0.95, 0.12, 0.14, 1.0),
    (0.10, 0.42, 0.95, 1.0),
    (0.10, 0.72, 0.28, 1.0),
    (0.98, 0.78, 0.10, 1.0),
    (0.62, 0.22, 0.92, 1.0),
]
_DEFAULT_WORKSPACE = ((0.35, 0.65), (-0.25, 0.25), (0.02, 0.02))


@dataclass
class ObjConfig:
    """Configuration for one object in the Genesis scene."""

    name: str
    type: str = "box"
    position: tuple[float, float, float] | str | None = "random"
    color: tuple[float, float, float, float] | str | None = "random"
    size: float | tuple[float, float, float] | str | None = "random"
    radius: float | str | None = None
    height: float | str | None = None
    position_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        _DEFAULT_WORKSPACE
    )
    size_range: tuple[float, float] = (0.035, 0.055)
    radius_range: tuple[float, float] = (0.02, 0.035)
    height_range: tuple[float, float] = (0.04, 0.08)
    color_range: tuple[float, float] = (0.05, 0.95)


def _default_objects() -> list[ObjConfig]:
    return [
        ObjConfig(name=f"object_{idx}", type="box", color="random", size="random")
        for idx in range(5)
    ]


@dataclass
class FrankaEnvConfig(BaseEnvConfig):
    """Configuration for FrankaEnv."""

    robot: Any | None = None
    backend: str = "cpu"
    use_gui: bool = False
    sim_step: float = 0.01
    objects: list[ObjConfig | dict[str, Any]] = field(default_factory=_default_objects)


@BaseEnv.register()
class FrankaEnv(BaseEnv):
    """Genesis scene with one Franka arm and configured objects."""

    name = "Genesis Franka Env"
    config_cls = FrankaEnvConfig

    def __init__(
        self,
        config: FrankaEnvConfig,
        logger: logging.Logger | None = None,
    ):
        super().__init__(config=config, logger=logger)
        self._robot = config.robot
        self._backend = str(config.backend)
        self._use_gui = bool(config.use_gui)
        self._sim_step = float(config.sim_step)
        self._object_configs = [self._coerce_obj_config(obj) for obj in config.objects]
        self._rng = np.random.default_rng(self._seed)

        self._scene = None
        self._objects: list[Any] = []
        self._object_specs: list[dict[str, Any]] = []
        self._genesis_unavailable_logged = False

    @classmethod
    def env_type(cls) -> str:
        return "genesis_franka"

    @property
    def robot(self) -> Any | None:
        """Return the wrapped Genesis robot instance."""
        return self._robot

    @property
    def scene(self) -> Any | None:
        """Return the Genesis scene when it is available."""
        return self._scene

    @property
    def objects(self) -> list[Any]:
        """Return Genesis object entities."""
        return self._objects

    def attach(self, robot: Any):
        """Attach a Genesis robot instance."""
        self._robot = robot

    def set_joint_positions(self, positions: list[float]) -> bool:
        """Set the robot's joint positions."""
        if self._robot is None:
            self.logger.info("[Mock] set_joint_positions(%s)", positions)
            return True

        try:
            if hasattr(self._robot, "set_joint_positions"):
                self._robot.set_joint_positions(positions)
            elif hasattr(self._robot, "set_qpos"):
                self._robot.set_qpos(positions)
            else:
                raise AttributeError("Robot does not support setting joint positions")
            return True
        except Exception as exc:
            self.logger.warning("Error setting joint positions: %s", exc)
            return False

    def get_joint_positions(self) -> list[float]:
        """Get the current joint positions."""
        if self._robot is None:
            self.logger.info("[Mock] get_joint_positions()")
            return [0.0] * 7

        try:
            if hasattr(self._robot, "get_joint_positions"):
                return list(self._robot.get_joint_positions())
            if hasattr(self._robot, "get_qpos"):
                return self._robot.get_qpos().tolist()
            raise AttributeError("Robot does not support getting joint positions")
        except Exception as exc:
            self.logger.warning("Error getting joint positions: %s", exc)
            return [0.0] * 7

    def set_gripper_position(self, width: float) -> bool:
        """Set the gripper opening width."""
        if self._robot is None:
            self.logger.info("[Mock] set_gripper_position(width=%s)", width)
            return True

        try:
            if hasattr(self._robot, "set_gripper_position"):
                self._robot.set_gripper_position(width)
            else:
                raise AttributeError("Robot does not support set_gripper_position")
            return True
        except Exception as exc:
            self.logger.warning("Error setting gripper position: %s", exc)
            return False

    def get_ee_pose(self) -> list[float]:
        """Get the end-effector pose as [x, y, z, qw, qx, qy, qz]."""
        if self._robot is None:
            self.logger.info("[Mock] get_ee_pose()")
            return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        try:
            if hasattr(self._robot, "get_ee_pose"):
                return list(self._robot.get_ee_pose())
            if hasattr(self._robot, "get_link"):
                ee_link = self._robot.get_link("hand")
                return self._tensor_to_flat_list(ee_link.get_pos()) + self._tensor_to_flat_list(ee_link.get_quat())
            raise AttributeError("Robot does not support get_ee_pose or get_link('hand')")
        except Exception as exc:
            self.logger.warning("Error getting EE pose: %s", exc)
            return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    def move_to_pose(self, x: float, y: float, z: float, duration: float = 1.0) -> bool:
        """Move the end-effector to a Cartesian position."""
        if self._robot is None:
            self.logger.info(
                "[Mock] move_to_pose(x=%s, y=%s, z=%s, duration=%s)",
                x,
                y,
                z,
                duration,
            )
            return True

        try:
            if hasattr(self._robot, "move_to_pose"):
                self._robot.move_to_pose(x, y, z, duration)
            else:
                raise AttributeError("Robot does not support move_to_pose")
            return True
        except Exception as exc:
            self.logger.warning("Error moving to pose: %s", exc)
            return False

    def grasp(self) -> bool:
        """Close the gripper to grasp an object."""
        return self.set_gripper_position(0.0)

    def release(self) -> bool:
        """Open the gripper to release an object."""
        return self.set_gripper_position(0.08)

    def _reset(
        self,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the Genesis scene and return the initial observation."""
        options = options or {}
        self._ensure_scene()
        if self._scene is not None and hasattr(self._scene, "reset"):
            self._scene.reset()
        elif self._robot is not None and hasattr(self._robot, "reset"):
            self._robot.reset()
        self._reset_objects()
        obs = self._build_observation()
        return obs, {"seed": self._seed, "options": options, "object_count": len(self._object_specs)}

    def _step(self, action: Any = None) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        """Apply one low-level action and return a Gymnasium step tuple."""
        action = action or {}
        if "joint_positions" in action:
            self.set_joint_positions(action["joint_positions"])
        if "gripper_width" in action:
            self.set_gripper_position(float(action["gripper_width"]))
        if "pose" in action:
            pose = action["pose"]
            self.move_to_pose(float(pose[0]), float(pose[1]), float(pose[2]))

        sim_steps = int(action.get("sim_steps", 1)) if isinstance(action, dict) else 1
        for _ in range(max(sim_steps, 1)):
            self.step_simulation()

        return self._build_observation(), 0.0, False, False, {"action": action}

    def step_simulation(self) -> None:
        """Advance the Genesis simulation by one timestep."""
        if self._scene is not None:
            self._scene.step()

    def get_observation(self, folder: str | Path) -> dict[str, Any]:
        """Return the latest robot and object state."""
        return self._build_observation()

    def _ensure_scene(self) -> None:
        if self._scene is not None:
            return
        try:
            import genesis as gs
        except ImportError:
            if not self._genesis_unavailable_logged:
                self.logger.warning("Genesis is not importable; FrankaEnv is running in mock mode")
                self._genesis_unavailable_logged = True
            return

        try:
            backend = getattr(gs, self._backend)
            gs.init(backend=backend)
        except Exception as exc:
            message = str(exc)
            if "already" not in message.lower() and "initialized" not in message.lower():
                raise

        self._scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, -2.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                res=(1280, 960),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=self._sim_step),
            show_viewer=self._use_gui,
        )
        self._scene.add_entity(gs.morphs.Plane())
        if self._robot is None:
            self._robot = self._scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

        self._object_specs = self._sample_object_specs()
        self._objects = [
            self._scene.add_entity(
                self._build_morph(gs, spec),
                surface=gs.surfaces.Default(color=spec["color"]),
            )
            for spec in self._object_specs
        ]
        self._scene.build()

    def _sample_object_specs(self) -> list[dict[str, Any]]:
        used_random_colors = list(_DEFAULT_COLORS)
        self._rng.shuffle(used_random_colors)
        specs = []
        for idx, config in enumerate(self._object_configs):
            specs.append(self._sample_object_spec(config, idx, used_random_colors))
        return specs

    def _sample_object_spec(
        self,
        config: ObjConfig,
        idx: int,
        random_colors: list[tuple[float, float, float, float]],
    ) -> dict[str, Any]:
        obj_type = config.type.lower()
        color = self._resolve_color(config, idx, random_colors)
        size = self._resolve_size(config)
        radius_value = "random" if obj_type in {"sphere", "cylinder"} and config.radius is None else config.radius
        height_value = "random" if obj_type == "cylinder" and config.height is None else config.height
        radius = self._resolve_scalar(radius_value, config.radius_range)
        height = self._resolve_scalar(height_value, config.height_range)
        position = self._resolve_position(config, obj_type, size=size, radius=radius, height=height)
        return {
            "name": config.name,
            "type": obj_type,
            "position": position,
            "color": color,
            "size": size,
            "radius": radius,
            "height": height,
        }

    def _reset_objects(self) -> None:
        if not self._object_specs:
            self._object_specs = self._sample_object_specs()
        else:
            self._resample_reset_positions()
        if not self._objects:
            return
        for entity, spec in zip(self._objects, self._object_specs):
            if hasattr(entity, "set_pos"):
                entity.set_pos(spec["position"])
            if hasattr(entity, "set_vel"):
                entity.set_vel((0.0, 0.0, 0.0))

    def _resample_reset_positions(self) -> None:
        for idx, config in enumerate(self._object_configs):
            if config.position == "random":
                spec = self._object_specs[idx]
                spec["position"] = self._resolve_position(
                    config,
                    spec["type"],
                    size=spec["size"],
                    radius=spec["radius"],
                    height=spec["height"],
                )

    def _build_observation(self) -> dict[str, Any]:
        objects = []
        for idx, spec in enumerate(self._object_specs):
            entity = self._objects[idx] if idx < len(self._objects) else None
            position = self._entity_pos(entity, fallback=spec["position"])
            velocity = self._entity_vel(entity)
            objects.append(
                {
                    "name": spec["name"],
                    "type": spec["type"],
                    "color": list(spec["color"]),
                    "position": position,
                    "velocity": velocity,
                    "size": spec["size"],
                    "radius": spec["radius"],
                    "height": spec["height"],
                }
            )
        return {
            "joint_positions": self.get_joint_positions(),
            "ee_pose": self.get_ee_pose(),
            "objects": objects,
            "object_positions": {obj["name"]: obj["position"] for obj in objects},
        }

    @staticmethod
    def _coerce_obj_config(config: ObjConfig | dict[str, Any]) -> ObjConfig:
        if isinstance(config, ObjConfig):
            return config
        return ObjConfig(**config)

    @staticmethod
    def _build_morph(gs: Any, spec: dict[str, Any]) -> Any:
        obj_type = spec["type"]
        if obj_type in {"box", "cube"}:
            return gs.morphs.Box(size=spec["size"], pos=spec["position"])
        if obj_type == "sphere":
            return gs.morphs.Sphere(radius=spec["radius"], pos=spec["position"])
        if obj_type == "cylinder":
            return gs.morphs.Cylinder(radius=spec["radius"], height=spec["height"], pos=spec["position"])
        morph_cls = getattr(gs.morphs, obj_type[:1].upper() + obj_type[1:], None)
        if morph_cls is None:
            raise ValueError(f"Unsupported Genesis object type: {obj_type!r}")
        return morph_cls(pos=spec["position"])

    def _resolve_color(
        self,
        config: ObjConfig,
        idx: int,
        random_colors: list[tuple[float, float, float, float]],
    ) -> tuple[float, float, float, float]:
        if config.color == "random" or config.color is None:
            if idx < len(random_colors):
                return random_colors[idx]
            low, high = config.color_range
            return tuple(self._rng.uniform(low, high, size=3).tolist()) + (1.0,)
        return tuple(map(float, config.color))

    def _resolve_size(self, config: ObjConfig) -> tuple[float, float, float] | None:
        if config.size == "random" or config.size is None:
            edge = float(self._rng.uniform(*config.size_range))
            return (edge, edge, edge)
        if isinstance(config.size, int | float):
            edge = float(config.size)
            return (edge, edge, edge)
        return tuple(map(float, config.size))

    def _resolve_scalar(
        self,
        value: float | str | None,
        value_range: tuple[float, float],
    ) -> float | None:
        if value == "random":
            return float(self._rng.uniform(*value_range))
        if value is None:
            return None
        return float(value)

    def _resolve_position(
        self,
        config: ObjConfig,
        obj_type: str,
        *,
        size: tuple[float, float, float] | None,
        radius: float | None,
        height: float | None,
    ) -> tuple[float, float, float]:
        if config.position != "random" and config.position is not None:
            return tuple(map(float, config.position))
        x_range, y_range, z_range = config.position_range
        x = float(self._rng.uniform(*x_range))
        y = float(self._rng.uniform(*y_range))
        z = self._default_z(obj_type, size=size, radius=radius, height=height)
        if z_range[0] != z_range[1]:
            z = float(self._rng.uniform(*z_range))
        return (x, y, z)

    @staticmethod
    def _default_z(
        obj_type: str,
        *,
        size: tuple[float, float, float] | None,
        radius: float | None,
        height: float | None,
    ) -> float:
        if obj_type in {"box", "cube"} and size is not None:
            return float(size[2]) / 2.0
        if obj_type == "sphere" and radius is not None:
            return float(radius)
        if obj_type == "cylinder" and height is not None:
            return float(height) / 2.0
        return _DEFAULT_WORKSPACE[2][0]

    @staticmethod
    def _entity_pos(entity: Any | None, fallback: Any) -> list[float]:
        if entity is None or not hasattr(entity, "get_pos"):
            return list(map(float, fallback))
        return np.asarray(entity.get_pos(), dtype=float).reshape(-1).tolist()

    @staticmethod
    def _tensor_to_flat_list(value: Any) -> list[float]:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=float).reshape(-1).tolist()

    @staticmethod
    def _entity_vel(entity: Any | None) -> list[float]:
        if entity is None or not hasattr(entity, "get_vel"):
            return [0.0, 0.0, 0.0]
        return np.asarray(entity.get_vel(), dtype=float).reshape(-1).tolist()
