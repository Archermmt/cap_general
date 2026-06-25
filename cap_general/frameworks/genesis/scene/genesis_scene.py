"""Genesis shared scene resource."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cap_general.core.scene import BaseScene, BaseSceneConfig
from cap_general.core.utils import save_image


@dataclass
class GenesisSceneConfig(BaseSceneConfig):
    """Configuration for a Genesis-backed multi-agent scene."""

    backend: str | None = None
    show_viewer: bool = False
    sim_options: dict[str, Any] = field(default_factory=dict)
    rigid_options: dict[str, Any] = field(default_factory=dict)
    viewer_options: dict[str, Any] = field(default_factory=dict)
    vis_options: dict[str, Any] = field(default_factory=dict)
    profiling_options: dict[str, Any] = field(default_factory=dict)
    image_key: str = "viewer"
    camera_name: str = "scene_camera"
    camera_res: tuple[int, int] = (640, 480)
    camera_pos: tuple[float, float, float] = (3.0, 0.0, 3.0)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.8)
    camera_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    camera_fov: float = 45.0
    camera_near: float = 0.02
    camera_far: float = 20.0


_SCENE: Any | None = None
_SCENE_CONFIG: GenesisSceneConfig | None = None


def get_scene(config: GenesisSceneConfig, *, gs: Any | None = None) -> Any:
    """Create or return the process-global Genesis scene."""
    global _SCENE, _SCENE_CONFIG

    if _SCENE is not None:
        return _SCENE
    if gs is None:
        import genesis as gs

    _SCENE = create_scene_from_config(config, gs=gs)
    _SCENE_CONFIG = config
    return _SCENE


def create_scene_from_config(config: GenesisSceneConfig, *, gs: Any | None = None) -> Any:
    """Create a fresh Genesis scene from config."""
    if gs is None:
        import genesis as gs

    kwargs: dict[str, Any] = {"show_viewer": config.show_viewer}
    if config.sim_options:
        kwargs["sim_options"] = gs.options.SimOptions(**_resolve_options(config.sim_options, gs))
    if config.rigid_options:
        kwargs["rigid_options"] = gs.options.RigidOptions(**_resolve_options(config.rigid_options, gs))
    if config.viewer_options:
        kwargs["viewer_options"] = gs.options.ViewerOptions(**_resolve_options(config.viewer_options, gs))
    if config.vis_options:
        kwargs["vis_options"] = gs.options.VisOptions(**_resolve_options(config.vis_options, gs))
    if config.profiling_options:
        kwargs["profiling_options"] = gs.options.ProfilingOptions(**_resolve_options(config.profiling_options, gs))
    return gs.Scene(**kwargs)


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


class GenesisScene(BaseScene):
    """Scene that owns one shared Genesis scene plus optional observation camera."""

    name = "Genesis Scene"
    config_cls = GenesisSceneConfig

    @classmethod
    def scene_type(cls) -> str:
        """Return the registry key for Genesis scene configs."""
        return "genesis_scene"

    def __init__(self, config: GenesisSceneConfig | dict[str, Any], logger=None):
        self._scene = None
        self._camera = None
        self._camera_failed = False
        self._defer_scene_build = False
        self._scene_built = False
        self._build_kwargs: dict[str, Any] | None = None
        self._post_build_callbacks: list[Callable[[], None]] = []
        self._pre_step_callbacks: list[Callable[[], bool | None]] = []
        self._step_owner_thread = threading.current_thread()
        self._step_requests: queue.Queue[Future[None]] = queue.Queue()
        super().__init__(config=config, logger=logger)

    @staticmethod
    def _coerce_config(config: GenesisSceneConfig | dict[str, Any]) -> GenesisSceneConfig:
        if isinstance(config, GenesisSceneConfig):
            return config
        if isinstance(config, dict):
            return GenesisSceneConfig(**{key: value for key, value in config.items() if key != "type"})
        raise TypeError(f"Expected GenesisSceneConfig or dict, got {type(config).__name__}")

    def _before_build_agents(self) -> None:
        self._scene = self._create_scene()
        self._step_owner_thread = threading.current_thread()
        self._attach_thread_stepper()
        self._add_default_ground_plane()
        self._defer_scene_build = True

    def _attach_thread_stepper(self) -> None:
        if self._scene is not None:
            setattr(self._scene, "_cap_step_scene", self.step_scene)
            setattr(self._scene, "_cap_register_pre_step_callback", self.register_pre_step_callback)
            setattr(self._scene, "register_pre_step_callback", self.register_pre_step_callback)

    def register_pre_step_callback(self, callback: Callable[[], bool | None]) -> None:
        """Register a callback to run on the scene owner thread before each step."""
        self._pre_step_callbacks.append(callback)

    def _run_pre_step_callbacks(self) -> bool:
        veto_step = False
        for callback in list(self._pre_step_callbacks):
            veto_step = bool(callback()) or veto_step
        return veto_step

    def step_scene(self) -> None:
        """Step Genesis on the scene owner thread, even when requested by a worker."""
        if self._scene is None:
            return
        if threading.current_thread() is self._step_owner_thread:
            if not self._run_pre_step_callbacks():
                self._scene.step()
            return
        future: Future[None] = Future()
        self._step_requests.put(future)
        future.result()

    def _process_background_events(self) -> bool:
        processed = False
        while True:
            try:
                future = self._step_requests.get_nowait()
            except queue.Empty:
                break
            if future.cancelled():
                processed = True
                continue
            try:
                if self._scene is not None and not self._run_pre_step_callbacks():
                    self._scene.step()
                future.set_result(None)
            except BaseException as exc:
                future.set_exception(exc)
            processed = True
        return processed

    def _add_default_ground_plane(self) -> None:
        if self._scene is None:
            return
        try:
            import genesis as gs
        except ImportError:
            return
        self._scene.add_entity(gs.morphs.Plane())

    def _after_build_agents(self) -> None:
        self._defer_scene_build = False
        if self._scene is None or self._scene_built:
            return
        if self._build_kwargs is None:
            return
        self._logger.info("Building Genesis scene with kwargs=%s", self._build_kwargs)
        self._scene.build(**self._build_kwargs)
        self._scene_built = True
        callbacks = list(self._post_build_callbacks)
        self._post_build_callbacks.clear()
        for callback in callbacks:
            callback()

    @property
    def scene(self) -> Any | None:
        """Return the Genesis scene owned by this scene wrapper."""
        return self._scene

    def _create_scene(self) -> Any | None:
        try:
            import genesis as gs
        except ImportError as exc:
            self._logger.warning("Genesis scene is disabled because Genesis is not importable: %s", exc)
            return None

        try:
            if self._config.backend:
                gs.init(backend=getattr(gs, self._config.backend))
            else:
                gs.init()
        except Exception as exc:
            message = str(exc)
            if "already" not in message.lower() and "initialized" not in message.lower():
                self._logger.warning("Genesis scene failed to initialize Genesis: %s", exc)
                return None
        return get_scene(self._config, gs=gs)

    def defer_build(self, build_kwargs: dict[str, Any], post_build: Callable[[], None]) -> bool:
        """Defer the shared Genesis scene build until all robots are created."""
        if not self._defer_scene_build:
            return False
        self._merge_build_kwargs(build_kwargs)
        self._post_build_callbacks.append(post_build)
        return True

    def _merge_build_kwargs(self, build_kwargs: dict[str, Any]) -> None:
        if self._build_kwargs is None:
            self._build_kwargs = dict(build_kwargs)
            return
        for key, value in build_kwargs.items():
            if key not in self._build_kwargs:
                self._build_kwargs[key] = value
            elif self._build_kwargs[key] != value:
                self._logger.warning(
                    "Ignoring conflicting Genesis scene.build kwarg %s=%r; using %r",
                    key,
                    value,
                    self._build_kwargs[key],
                )

    def get_observation(self, folder: str | Path) -> dict[str, Any]:
        """Render and save the shared scene camera image."""
        image = self._render_scene_image()
        if image is None:
            return {"images": {}, "main_image": None}
        image_dir = Path(folder)
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = save_image(image_dir / f"{self._config.image_key}.png", image)
        return {"images": {self._config.image_key: image_path}, "main_image": image_path}

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
            self._logger.warning("Disabled Genesis scene camera after render failure: %s", exc)
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

    def get_resource(self, name: str) -> Any:
        """Return this Genesis scene by resource name."""
        if name == "genesis_scene":
            return self
        return None


GenesisScene._registry[GenesisScene.scene_type()] = GenesisScene
