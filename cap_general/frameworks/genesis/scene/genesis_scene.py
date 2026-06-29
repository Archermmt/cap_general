"""Genesis shared scene resource."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cap_general.core.scene import AgentSpec, BaseScene, BaseSceneConfig
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
    build_kwargs: dict[str, Any] = field(default_factory=lambda: {"n_envs": 1})
    image_key: str = "viewer"
    camera_name: str = "scene_camera"
    camera_res: tuple[int, int] = (640, 480)
    camera_pos: tuple[float, float, float] = (3.0, 0.0, 3.0)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.8)
    camera_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    camera_fov: float = 45.0
    camera_near: float = 0.02
    camera_far: float = 20.0
    lock_viewer_rotation: bool = True
    idle_render_fps: float = 30.0


def _resolve_options(options: dict[str, Any], gs: Any) -> dict[str, Any]:
    resolved = dict(options)
    solver = resolved.get("constraint_solver")
    if isinstance(solver, str):
        resolved["constraint_solver"] = getattr(gs.constraint_solver, solver)
    return resolved


@BaseScene.register()
class GenesisScene(BaseScene):
    """Scene that owns one shared Genesis scene plus optional observation camera."""

    name = "Genesis Scene"
    config_cls = GenesisSceneConfig

    @classmethod
    def scene_type(cls) -> str:
        """Return the registry key for Genesis scene configs."""
        return "genesis"

    def __init__(self, config: GenesisSceneConfig | dict[str, Any], logger=None):
        self._gs_scene = None
        self._camera = None
        self._camera_failed = False
        self._post_build_callbacks: list[Callable[[], None]] = []
        self._pre_step_callbacks: list[Callable[[], bool | None]] = []
        self._idle_render_task: asyncio.Task | None = None
        self._step_lock = threading.Lock()
        self._agent_locks: dict[Any, asyncio.Lock] = {}
        super().__init__(config=config, logger=logger)

    def _build_agents(self, specs: list[AgentSpec | dict[str, Any]]) -> None:
        import genesis as gs

        if self._config.backend:
            gs.init(backend=getattr(gs, self._config.backend))
        else:
            gs.init()

        scene_kwargs: dict[str, Any] = {"show_viewer": self._config.show_viewer}
        if self._config.sim_options:
            scene_kwargs["sim_options"] = gs.options.SimOptions(**_resolve_options(self._config.sim_options, gs))
        if self._config.rigid_options:
            scene_kwargs["rigid_options"] = gs.options.RigidOptions(**_resolve_options(self._config.rigid_options, gs))
        if self._config.viewer_options:
            scene_kwargs["viewer_options"] = gs.options.ViewerOptions(
                **_resolve_options(self._config.viewer_options, gs)
            )
        if self._config.vis_options:
            scene_kwargs["vis_options"] = gs.options.VisOptions(**_resolve_options(self._config.vis_options, gs))
        if self._config.profiling_options:
            scene_kwargs["profiling_options"] = gs.options.ProfilingOptions(
                **_resolve_options(self._config.profiling_options, gs)
            )
        self._gs_scene = gs.Scene(**scene_kwargs)
        self._gs_scene._cap_step_scene = self.step_scene
        self._gs_scene._cap_register_pre_step_callback = self.register_pre_step_callback
        self._gs_scene.add_entity(gs.morphs.Plane())
        super()._build_agents(specs)
        self._logger.info("Building Genesis scene with kwargs=%s", self._config.build_kwargs)
        self._gs_scene.build(**self._config.build_kwargs)
        self._lock_viewer_rotation()
        callbacks = list(self._post_build_callbacks)
        self._post_build_callbacks.clear()
        for callback in callbacks:
            callback()
        if self._config.show_viewer and self._config.idle_render_fps > 0:
            self._start_idle_render_loop()

    def register_pre_step_callback(self, callback: Callable[[], bool | None]) -> None:
        """Register a callback to run before each scene step."""
        self._pre_step_callbacks.append(callback)

    def _run_pre_step_callbacks(self) -> bool:
        veto_step = False
        for callback in list(self._pre_step_callbacks):
            veto_step = bool(callback()) or veto_step
        return veto_step

    def _lock_viewer_rotation(self) -> None:
        if not self._config.lock_viewer_rotation:
            return
        viewer = getattr(getattr(self._gs_scene, "_visualizer", None), "_viewer", None)
        pyrender_viewer = getattr(viewer, "_pyrender_viewer", None)
        viewer_flags = getattr(pyrender_viewer, "viewer_flags", None)
        if isinstance(viewer_flags, dict):
            viewer_flags["rotate"] = False

    def step_scene(self) -> None:
        """Step the Genesis scene."""
        self._real_step()

    def _real_step(self) -> None:
        """Unconditional scene step."""
        with self._step_lock:
            self._lock_viewer_rotation()
            if not self._run_pre_step_callbacks():
                self._gs_scene.step()

    def _start_idle_render_loop(self) -> None:
        """Schedule the idle render loop on the running event loop, if available."""
        if self._idle_render_task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._idle_render_task = loop.create_task(self._idle_render_loop())

    async def _idle_render_loop(self) -> None:
        """Continuously step the scene while no agent task is running."""
        interval = 1.0 / self._config.idle_render_fps
        while True:
            await asyncio.sleep(interval)
            # Skip when an agent task is actively executing
            if any(info.task is not None and not info.task.done() for info in self._agents.values()):
                continue
            self._real_step()

    @property
    def scene(self) -> Any:
        """Return the Genesis scene owned by this scene wrapper."""
        return self._gs_scene

    def defer_build(self, post_build: Callable[[], None]) -> None:
        """Defer the shared Genesis scene build until all robots are created."""
        self._post_build_callbacks.append(post_build)

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
            scene = self._gs_scene
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

    async def _on_server_started(self) -> None:
        """Start the idle render loop once the MCP server's event loop is live."""
        self._start_idle_render_loop()

    async def _dispatch_task(
        self,
        method: Callable[..., Any],
        kwargs: dict[str, Any],
    ) -> Any:
        """Run a Genesis agent method on the main thread under its agent lock."""
        agent_lock = self._agent_locks.setdefault(method.__self__, asyncio.Lock())
        async with agent_lock:
            return method(**kwargs)
