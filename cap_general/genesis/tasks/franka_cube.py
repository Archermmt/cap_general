"""Franka cube lift task for Genesis CAP."""

from typing import Optional, Dict, Any
import numpy as np


class FrankaCubeLiftTask:
    """Task for lifting a cube using a Franka robot in Genesis.

    This task sets up a simple scene with a Franka robot and a red cube,
    then evaluates the robot's ability to lift the cube.
    """

    def __init__(
        self, use_gui: bool = False, sim_step: float = 0.01, horizon: int = 100
    ):
        """Initialize the cube lift task.

        Args:
            use_gui: Whether to enable GUI visualization.
            sim_step: Simulation timestep in seconds.
            horizon: Maximum number of simulation steps.
        """
        self.use_gui = use_gui
        self.sim_step = sim_step
        self.horizon = horizon
        self.scene = None
        self.robot = None
        self.cube = None
        self.initial_cube_height = 0.0

    def setup_scene(self):
        """Set up the Genesis scene with Franka robot and cube."""
        try:
            import genesis as gs

            # Initialize scene
            self.scene = gs.Scene(
                viewer_options=gs.options.ViewerOptions(
                    camera_pos=(2.0, -2.0, 2.0),
                    camera_lookat=(0.0, 0.0, 0.5),
                    res=(1280, 960),
                    max_FPS=60,
                ),
                sim_options=gs.options.SimOptions(
                    dt=self.sim_step,
                ),
                show_viewer=self.use_gui,
            )

            # Add plane
            plane = self.scene.add_entity(
                gs.morphs.Plane(),
            )

            # Add Franka robot
            self.robot = self.scene.add_entity(
                gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
            )

            # Add red cube
            self.cube = self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.04, 0.04, 0.04),
                    pos=(0.5, 0.0, 0.02),
                ),
                surface=gs.surfaces.Default(
                    color=(1.0, 0.0, 0.0, 1.0),  # Red
                ),
            )

            # Store initial cube height
            self.initial_cube_height = self.cube.get_pos()[2]

            # Build scene
            self.scene.build()

        except ImportError:
            raise ImportError(
                "Genesis is required for FrankaCubeLiftTask. "
                "Install it according to the Genesis documentation."
            )

    def get_observation(self) -> Dict[str, Any]:
        """Get current task observation.

        Returns:
            Dictionary containing robot state and cube state.
        """
        if self.robot is None or self.cube is None:
            raise RuntimeError("Scene not initialized. Call setup_scene() first.")

        # Get robot joint positions
        joint_positions = self.robot.get_qpos().tolist()

        # Get cube position
        cube_pos = self.cube.get_pos().tolist()
        cube_vel = self.cube.get_vel().tolist()

        return {
            "joint_positions": joint_positions,
            "cube_position": cube_pos,
            "cube_velocity": cube_vel,
            "cube_height": cube_pos[2],
        }

    def compute_reward(self) -> float:
        """Compute reward based on cube height.

        Reward is proportional to how high the cube has been lifted
        from its initial position.

        Returns:
            Reward value between 0.0 and 1.0.
        """
        if self.cube is None:
            return 0.0

        current_height = self.cube.get_pos()[2]
        height_diff = current_height - self.initial_cube_height

        # Normalize reward: 0.1m lift = full reward
        reward = min(max(height_diff / 0.1, 0.0), 1.0)

        return reward

    def is_success(self) -> bool:
        """Check if the task has been successfully completed.

        Success is defined as lifting the cube at least 0.08 meters.

        Returns:
            True if task is successful.
        """
        if self.cube is None:
            return False

        current_height = self.cube.get_pos()[2]
        height_diff = current_height - self.initial_cube_height

        return height_diff >= 0.08

    def reset(self):
        """Reset the task to initial state."""
        if self.scene is not None:
            self.scene.reset()

            # Reset cube position
            if self.cube is not None:
                self.cube.set_pos((0.5, 0.0, 0.02))
                self.cube.set_vel((0.0, 0.0, 0.0))

                # Update initial height
                self.initial_cube_height = self.cube.get_pos()[2]

    def step_simulation(self):
        """Advance the simulation by one timestep."""
        if self.scene is not None:
            self.scene.step()

    def run_episode(self, actions_fn, max_steps: int = 100) -> Dict[str, Any]:
        """Run a complete episode with the given action function.

        Args:
            actions_fn: Function that takes observation and returns action.
            max_steps: Maximum number of steps to run.

        Returns:
            Dictionary with episode results.
        """
        if self.scene is None:
            self.setup_scene()

        self.reset()

        total_reward = 0.0
        success = False

        for step in range(max_steps):
            # Get observation
            obs = self.get_observation()

            # Get action from policy
            action = actions_fn(obs)

            # Execute action (this would interface with robot control)
            # For now, just simulate
            self.step_simulation()

            # Compute reward
            reward = self.compute_reward()
            total_reward += reward

            # Check success
            if self.is_success():
                success = True
                break

        return {
            "total_reward": total_reward,
            "success": success,
            "steps": step + 1,
            "final_cube_height": self.cube.get_pos()[2] if self.cube else 0.0,
        }
