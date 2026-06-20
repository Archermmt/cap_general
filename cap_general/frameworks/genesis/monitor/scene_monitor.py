"""Genesis scene monitor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cap_general.core.monitor import BaseMonitor, BaseMonitorConfig


@dataclass
class SceneConfig:
    """Configuration for the process-global Genesis scene."""

    show_viewer: bool = False
    sim_options: dict[str, Any] = field(default_factory=dict)
    rigid_options: dict[str, Any] = field(default_factory=dict)
    viewer_options: dict[str, Any] = field(default_factory=dict)
    vis_options: dict[str, Any] = field(default_factory=dict)
    profiling_options: dict[str, Any] = field(default_factory=dict)


_SCENE: Any | None = None
_SCENE_CONFIG: SceneConfig | None = None


def get_scene(config: SceneConfig | dict[str, Any] | None = None, *, gs: Any | None = None) -> Any:
    """Create or return the process-global Genesis scene."""
    global _SCENE, _SCENE_CONFIG

    if _SCENE is not None:
        return _SCENE
    if gs is None:
        import genesis as gs

    scene_config = coerce_scene_config(config)
    kwargs: dict[str, Any] = {"show_viewer": scene_config.show_viewer}
    if scene_config.sim_options:
        kwargs["sim_options"] = gs.options.SimOptions(**_resolve_options(scene_config.sim_options, gs))
    if scene_config.rigid_options:
        kwargs["rigid_options"] = gs.options.RigidOptions(**_resolve_options(scene_config.rigid_options, gs))
    if scene_config.viewer_options:
        kwargs["viewer_options"] = gs.options.ViewerOptions(**_resolve_options(scene_config.viewer_options, gs))
    if scene_config.vis_options:
        kwargs["vis_options"] = gs.options.VisOptions(**_resolve_options(scene_config.vis_options, gs))
    if scene_config.profiling_options:
        kwargs["profiling_options"] = gs.options.ProfilingOptions(
            **_resolve_options(scene_config.profiling_options, gs)
        )
    _SCENE = gs.Scene(**kwargs)
    _SCENE_CONFIG = scene_config
    return _SCENE


def coerce_scene_config(config: SceneConfig | dict[str, Any] | None) -> SceneConfig:
    """Return a SceneConfig from a dataclass, dict, or None."""
    if config is None:
        return SceneConfig()
    if isinstance(config, SceneConfig):
        return config
    if isinstance(config, dict):
        return SceneConfig(**config)
    raise TypeError(f"Expected SceneConfig or dict, got {type(config).__name__}")


def reset_scene() -> None:
    """Forget the process-global scene reference."""
    global _SCENE, _SCENE_CONFIG
    _SCENE = None
    _SCENE_CONFIG = None


def _resolve_options(options: dict[str, Any], gs: Any) -> dict[str, Any]:
    resolved = dict(options)
    solver = resolved.get("constraint_solver")
    if isinstance(solver, str):
        resolved["constraint_solver"] = getattr(gs.constraint_solver, solver)
    return resolved


@dataclass
class GenesisSceneMonitorConfig(BaseMonitorConfig):
    """Configuration for rendering the global Genesis scene."""

    backend: str | None = None
    scene: SceneConfig | dict[str, Any] = field(default_factory=SceneConfig)
    image_key: str = "viewer"
    camera_name: str = "monitor_camera"
    camera_res: tuple[int, int] = (640, 480)
    camera_pos: tuple[float, float, float] = (3.0, 0.0, 3.0)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.8)
    camera_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    camera_fov: float = 45.0
    camera_near: float = 0.02
    camera_far: float = 20.0


@BaseMonitor.register()
class GenesisSceneMonitor(BaseMonitor):
    """Monitor that renders the process-global Genesis scene camera."""

    name = "Genesis Scene Monitor"
    config_cls = GenesisSceneMonitorConfig

    @classmethod
    def monitor_type(cls) -> str:
        return "genesis_scene"

    def __init__(
        self,
        config: GenesisSceneMonitorConfig | None = None,
        logger=None,
    ):
        super().__init__(config=config or GenesisSceneMonitorConfig(), logger=logger)
        self._config = config or GenesisSceneMonitorConfig()
        self._scene_config = coerce_scene_config(self._config.scene)
        self._scene = self._create_scene()
        self._camera = None
        self._camera_failed = False

    @property
    def scene(self) -> Any | None:
        """Return the Genesis scene owned by this monitor."""
        return self._scene

    def _create_scene(self) -> Any | None:
        try:
            import genesis as gs
        except ImportError as exc:
            self._logger.warning("Genesis scene monitor is disabled because Genesis is not importable: %s", exc)
            return None

        try:
            if self._config.backend:
                gs.init(backend=getattr(gs, self._config.backend))
            else:
                gs.init()
        except Exception as exc:
            message = str(exc)
            if "already" not in message.lower() and "initialized" not in message.lower():
                self._logger.warning("Genesis scene monitor failed to initialize Genesis: %s", exc)
                return None
        return get_scene(self._scene_config, gs=gs)

    def _get_monitor_obs(self) -> dict[str, Any]:
        """Render an image from the global Genesis scene camera."""
        image = self._render_scene_image()
        if image is None:
            return {}
        return {self._config.image_key: image}

    def _render_scene_image(self) -> np.ndarray | None:
        if self._camera_failed:
            return None
        try:
            scene = self._scene
            if scene is None:
                return None
            camera = self._ensure_camera(scene)
            rgb = camera.render(rgb=True, force_render=True)[0]
            if getattr(rgb, "ndim", 0) > 3:
                rgb = rgb[0]
            return self._to_image_array(rgb)
        except Exception as exc:  # pragma: no cover - depends on Genesis renderer/runtime
            self._camera_failed = True
            self._logger.warning("Disabled Genesis scene monitor after render failure: %s", exc)
            return None

    def _ensure_camera(self, scene: Any) -> Any:
        if self._camera is not None:
            return self._camera
        camera = getattr(scene, self._config.camera_name, None)
        if camera is None:
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
            setattr(scene, self._config.camera_name, camera)
        self._camera = camera
        return camera

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
