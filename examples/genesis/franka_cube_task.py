"""Example: Franka cube lift task with CAP.

This example demonstrates how to use the Genesis CAP framework to execute
a cube lifting task using a static policy (for testing) or a Hugging Face
model (if available).

Usage:
    # Run with static policy (no model download required):
    uv run python examples/cap/franka_cube_task.py

    # Run with local Hugging Face model:
    uv run python examples/cap/franka_cube_task.py --model-path /path/to/model
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse

from cap_general.core.models import (
    CallablePolicyModel,
    HuggingFacePolicyModel,
    StaticPolicyModel,
)
from cap_general.genesis.apis.franka import GenesisFrankaApi
from cap_general.genesis.tasks.franka_cube import FrankaCubeLiftTask

from cap_general.core.env import CapEnv


def create_static_policy_code() -> str:
    """Create a simple static policy for lifting the cube.

    This is a deterministic policy that doesn't require a model.
    """
    return """
# Simple cube lift policy
# Move robot to grasp position
grasp_success = False
lift_height = 0.0

# Open gripper
release()

# Move to cube position (simplified - would need IK in real scenario)
target_joint_positions = [0.0, -0.3, 0.0, -2.0, 0.0, 1.5, 0.0]
set_joint_positions(target_joint_positions)

# Close gripper to grasp
grasp_success = grasp()

# Lift the cube by moving joints
if grasp_success:
    lift_positions = [0.0, -0.5, 0.0, -2.5, 0.0, 2.0, 0.0]
    set_joint_positions(lift_positions)
    lift_height = 0.15

# Calculate reward based on lift height
reward = min(lift_height / 0.1, 1.0)

# Mark task as complete
done = True
"""


def create_callable_policy():
    """Create a callable policy that generates code based on prompt."""

    def policy_generator(prompt: str) -> str:
        """Simple rule-based policy generator."""

        # Check if this is about grasping
        if "grasp" in prompt.lower() or "close" in prompt.lower():
            return """
# Grasp the cube
grasp()
reward = 0.3
done = False
"""

        # Check if this is about lifting
        elif "lift" in prompt.lower() or "raise" in prompt.lower():
            return """
# Lift the cube
lift_positions = [0.0, -0.5, 0.0, -2.5, 0.0, 2.0, 0.0]
set_joint_positions(lift_positions)
reward = 0.8
done = True
"""

        # Default: move to initial position
        else:
            return """
# Move to starting position
initial_positions = [0.0, -0.3, 0.0, -2.0, 0.0, 1.5, 0.0]
set_joint_positions(initial_positions)
reward = 0.1
done = False
"""

    return CallablePolicyModel(generator_fn=policy_generator, model_name="RuleBasedPolicy")


def run_with_static_policy():
    """Run the task with a static policy."""
    print("=" * 60)
    print("Running Franka Cube Lift Task with Static Policy")
    print("=" * 60)

    # Create API
    franka_api = GenesisFrankaApi(robot=None)  # No actual robot for this example

    # Get API documentation
    api_docs = franka_api._function_doc()

    # Create task description
    task_description = """
    Use the Franka robot to pick up and lift a red cube.
    The cube is located at position (0.5, 0.0, 0.02).
    Your goal is to lift it at least 0.08 meters from its initial position.
    
    Available actions:
    - set_joint_positions(positions): Set robot joint angles
    - get_joint_positions(): Get current joint angles
    - grasp(): Close gripper
    - release(): Open gripper
    - move_to_pose(x, y, z, duration): Move end-effector
    """

    # Create static policy model
    static_code = create_static_policy_code()
    policy_model = StaticPolicyModel(code=static_code)

    # Create CAP environment
    env = CapEnv(
        task_description=task_description,
        api_docs=api_docs,
        policy_model=policy_model,
        max_steps=5,
    )

    # Add API methods to executor's globals so they can be called
    env.executor.globals.update(
        {
            "set_joint_positions": franka_api.set_joint_positions,
            "get_joint_positions": franka_api.get_joint_positions,
            "grasp": franka_api.grasp,
            "release": franka_api.release,
            "move_to_pose": franka_api.move_to_pose,
        }
    )

    # Run the task
    print("\nExecuting task...")
    result = env.run()

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Task Completed!")
    print(f"{'=' * 60}")
    print(f"Total steps: {result.total_steps}")
    print(f"Final reward: {result.final_reward:.2f}")
    print(f"Success: {result.success}")

    if result.last_step:
        print(f"\nLast step execution:")
        print(f"  Success: {result.last_step.success}")
        if result.last_step.execution_result.stdout:
            print(f"  Output: {result.last_step.execution_result.stdout[:200]}")
        if result.last_step.execution_result.error:
            print(f"  Error: {result.last_step.execution_result.error}")

    return result


def run_with_callable_policy():
    """Run the task with a callable (rule-based) policy."""
    print("\n" + "=" * 60)
    print("Running Franka Cube Lift Task with Rule-Based Policy")
    print("=" * 60)

    # Create API
    franka_api = GenesisFrankaApi(robot=None)
    api_docs = franka_api._function_doc()

    # Create task description
    task_description = """
    Use the Franka robot to pick up and lift a red cube.
    First grasp the cube, then lift it up.
    """

    # Create callable policy model
    policy_model = create_callable_policy()

    # Create CAP environment
    env = CapEnv(
        task_description=task_description,
        api_docs=api_docs,
        policy_model=policy_model,
        max_steps=3,
    )

    # Add API methods to executor's globals
    env.executor.globals.update(
        {
            "set_joint_positions": franka_api.set_joint_positions,
            "get_joint_positions": franka_api.get_joint_positions,
            "grasp": franka_api.grasp,
            "release": franka_api.release,
            "move_to_pose": franka_api.move_to_pose,
        }
    )

    # Run the task
    print("\nExecuting multi-step task...")
    result = env.run()

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Task Completed!")
    print(f"{'=' * 60}")
    print(f"Total steps: {result.total_steps}")
    print(f"Final reward: {result.final_reward:.2f}")
    print(f"Success: {result.success}")

    # Print each step
    for i, step in enumerate(result.steps, 1):
        print(f"\nStep {i}:")
        print(f"  Success: {step.success}")
        print(f"  Done: {step.done}")
        print(f"  Reward: {step.reward:.2f}")
        print(f"  Code preview: {step.generated_code[:100]}...")

    return result


def run_with_huggingface_model(model_path: str):
    """Run the task with a Hugging Face model."""
    print("\n" + "=" * 60)
    print(f"Running Franka Cube Lift Task with Hugging Face Model")
    print(f"Model: {model_path}")
    print("=" * 60)

    try:
        # Create API
        franka_api = GenesisFrankaApi(robot=None)
        api_docs = franka_api._function_doc()

        # Create task description
        task_description = """
        Use the Franka robot to pick up and lift a red cube.
        Generate Python code to control the robot.
        """

        # Load Hugging Face model
        print("\nLoading model (this may take a while)...")
        policy_model = HuggingFacePolicyModel(
            model_path=model_path,
            device="cpu",  # Use "cuda" if GPU available
        )

        # Create CAP environment
        env = CapEnv(
            task_description=task_description,
            api_docs=api_docs,
            policy_model=policy_model,
            max_steps=3,
        )

        # Run the task
        print("\nExecuting task with AI model...")
        result = env.run()

        # Print results
        print(f"\n{'=' * 60}")
        print(f"Task Completed!")
        print(f"{'=' * 60}")
        print(f"Total steps: {result.total_steps}")
        print(f"Final reward: {result.final_reward:.2f}")
        print(f"Success: {result.success}")

        return result

    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install transformers: pip install transformers")
        return None
    except Exception as e:
        print(f"\nError running with Hugging Face model: {e}")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Franka Cube Lift Task Example")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to Hugging Face model (optional)",
    )

    args = parser.parse_args()

    print("Genesis CAP - Franka Cube Lift Task Example")
    print("=" * 60)

    # Run with static policy (always works)
    run_with_static_policy()

    # Run with callable policy (rule-based)
    run_with_callable_policy()

    # Run with Hugging Face model (if provided)
    if args.model_path:
        run_with_huggingface_model(args.model_path)

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
