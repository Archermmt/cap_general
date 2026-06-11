"""Genesis Franka cube manipulation agent."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from cap_general.core.agent import BaseAgent, BaseAgentConfig


@dataclass
class FrankaCubeAgentConfig(BaseAgentConfig):
    """Configuration for FrankaCubeAgent."""

    env: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_franka"})
    policys: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_steps: int = 999999
    use_gui: bool = False
    sim_step: float = 0.01
    horizon: int = 100


@BaseAgent.register()
class FrankaCubeAgent(BaseAgent):
    """Agent that runs a Franka cube grasp/lift task in Genesis."""

    name = "Genesis Franka Cube Agent"
    config_cls = FrankaCubeAgentConfig

    def __init__(
        self,
        config: FrankaCubeAgentConfig | dict[str, Any],
    ):
        config_obj = self._load_config(config=config)
        super().__init__(config=config_obj)
        self.use_gui = config_obj.use_gui
        self.sim_step = config_obj.sim_step
        self.horizon = config_obj.horizon
        self.scene = None
        self.cube = None
        self.initial_cube_height = 0.0

    @classmethod
    def agent_type(cls) -> str:
        return "genesis_franka_cube"

    def setup_scene(self):
        """Set up the Genesis scene with Franka robot and cube."""
        try:
            import genesis as gs

            self.scene = gs.Scene(
                viewer_options=gs.options.ViewerOptions(
                    camera_pos=(2.0, -2.0, 2.0),
                    camera_lookat=(0.0, 0.0, 0.5),
                    res=(1280, 960),
                    max_FPS=60,
                ),
                sim_options=gs.options.SimOptions(dt=self.sim_step),
                show_viewer=self.use_gui,
            )

            self.scene.add_entity(gs.morphs.Plane())
            genesis_robot = self.scene.add_entity(
                gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
            )
            self.env.attach(genesis_robot)

            self.cube = self.scene.add_entity(
                gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.5, 0.0, 0.02)),
                surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0, 1.0)),
            )
            self.scene.build()
            self.initial_cube_height = float(self.cube.get_pos()[2])
        except ImportError:
            raise ImportError(
                "Genesis is required for FrankaCubeAgent. "
                "Install it according to the Genesis documentation."
            )

    def get_observation(self) -> Dict[str, Any]:
        """Get current cube-task observation."""
        if self.env.robot is None or self.cube is None:
            raise RuntimeError("Scene not initialized. Call setup_scene() first.")

        cube_pos = self.cube.get_pos().tolist()
        cube_vel = self.cube.get_vel().tolist()
        return {
            "joint_positions": self.env.get_joint_positions(),
            "ee_pose": self.env.get_ee_pose(),
            "cube_position": cube_pos,
            "cube_velocity": cube_vel,
            "cube_height": cube_pos[2],
        }

    def compute_reward(self) -> float:
        """Compute reward from cube lift height."""
        if self.cube is None:
            return 0.0
        height_diff = float(self.cube.get_pos()[2]) - self.initial_cube_height
        return min(max(height_diff / 0.1, 0.0), 1.0)

    def is_success(self) -> bool:
        """Check whether the cube has been lifted enough."""
        if self.cube is None:
            return False
        height_diff = float(self.cube.get_pos()[2]) - self.initial_cube_height
        return height_diff >= 0.08

    def functions(self) -> Dict[str, Callable[..., Any]]:
        """Return cube-task functions exposed to generated code."""
        return {
            "get_observation": self.get_observation,
            "compute_reward": self.compute_reward,
            "is_success": self.is_success,
            "step_simulation": self.step_simulation,
            "run": self.run,
        }

    def step_simulation(self):
        """Advance the simulation by one timestep."""
        if self.scene is not None:
            self.scene.step()

    def reset(self):
        """Reset the Genesis task state."""
        if self.scene is None:
            return
        self.scene.reset()
        if self.cube is not None:
            self.cube.set_pos((0.5, 0.0, 0.02))
            self.cube.set_vel((0.0, 0.0, 0.0))
            self.initial_cube_height = float(self.cube.get_pos()[2])

    def run(
        self,
        actions_fn: Callable[[Dict[str, Any]], Any] | None = None,
        max_steps: int | None = None,
    ) -> Dict[str, Any]:
        """Run a complete cube grasp/lift episode."""
        if self.scene is None:
            self.setup_scene()

        self.reset()
        steps = max_steps or self.horizon
        total_reward = 0.0
        success = False

        for step in range(steps):
            obs = self.get_observation()
            if actions_fn is not None:
                actions_fn(obs)

            self.step_simulation()
            reward = self.compute_reward()
            total_reward += reward

            if self.is_success():
                success = True
                break

        return {
            "total_reward": total_reward,
            "success": success,
            "steps": step + 1 if steps > 0 else 0,
            "final_cube_height": float(self.cube.get_pos()[2]) if self.cube else 0.0,
        }
