"""LIBERO VLA agent."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cap_general.core.agent import BaseAgent, BaseAgentConfig
from cap_general.frameworks.libero.robot.libero_robot import _binarize_gripper_open, build_example_from_obs


@dataclass
class LiberoAgentConfig(BaseAgentConfig):
    """Configuration for LiberoAgent."""

    robot: dict[str, Any] = field(default_factory=lambda: {"type": "libero"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)


@BaseAgent.register()
class LiberoAgent(BaseAgent):
    """Agent that runs LIBERO subtasks with a configured VLA policy."""

    agent_type = "libero"
    config_cls = LiberoAgentConfig

    def _execute_rules(self) -> str:
        """Return valid rules for execute for the loaded LIBERO suite."""
        task_suite = getattr(self._robot, "_task_suite", None)
        if task_suite is None:
            return ""

        suite_name = getattr(self._robot, "_task_suite_name", "unknown")
        tasks = [task_suite.get_task(i).language for i in range(task_suite.get_num_tasks())]
        task_list = "\n".join(f"  - {task}" for task in tasks)
        return (
            f"Task suite: {suite_name}\n"
            "Available task descriptions (pass verbatim to libero_vla_episode):\n"
            f"{task_list}\n"
            "IMPORTANT: libero_vla_episode(task=...) only accepts the exact strings "
            "listed above. Map the user's goal to one or more of these descriptions "
            "when decomposing subtasks."
        )

    def _options_doc(self, method_name: str) -> str:
        """Return LIBERO-specific options docs for supported methods."""
        if method_name == "reset":
            base_doc = super()._options_doc(method_name)
            extra_doc = (
                "episode_idx: LIBERO initial state index to load when reset_level is "
                "1 or 2. Defaults to 0."
            )
            return f"{base_doc}\n{extra_doc}" if base_doc else extra_doc
        if method_name == "train":
            return (
                "starvla_root: StarVLA repository root.\n"
                "config_yaml: Training YAML path relative to starvla_root.\n"
                "base_vlm: Base VLM path or model identifier.\n"
                "data_root: LIBERO LeRobot dataset root.\n"
                "data_mix: Dataset mixture name (default libero_all).\n"
                "run_root_dir: Parent directory for training outputs.\n"
                "run_id: Training run identifier.\n"
                "num_processes: Number of Accelerate workers (default 1).\n"
                "per_device_batch_size: Per-device VLA batch size (default 16).\n"
                "save_interval: Checkpoint interval in steps (default 10000).\n"
                "logging_frequency: Logging interval in steps (default 100).\n"
                "eval_interval: Evaluation interval in steps (default 100).\n"
                "freeze_modules: Space-separated module names to freeze.\n"
                "wandb_project: Weights & Biases project name.\n"
                "wandb_entity: Weights & Biases entity name.\n"
                "wandb_mode: Optional WANDB_MODE environment value.\n"
                "overrides: Additional StarVLA dotlist arguments."
            )
        return super()._options_doc(method_name)

    def functions(self) -> dict[str, Callable[..., Any]]:
        """Return LIBERO functions exposed to generated code."""
        return {"libero_vla_episode": self.libero_vla_episode}

    def libero_vla_episode(self, task: str, max_steps: int = 300, policy_name: str = "starvla") -> bool:
        """Run a full LIBERO episode using the configured VLA policy.

        Args:
            task: Task description string. It must match one task language in the
                current LIBERO suite.
            max_steps: Maximum number of robot control steps.
            vla_policy: Name of the configured VLA policy to run.

        Returns:
            True when the LIBERO success predicate is reached.
        """
        self._robot.set_task_goal(task)
        obs, done = self._robot.last_obs, False
        for step_idx in range(max_steps):
            example = build_example_from_obs(obs, task)
            response = self._run_policy(policy_name, inputs={"example": example, "step": step_idx})
            raw = response.get("raw_action", response)
            action = np.concatenate(
                [
                    np.asarray(raw["world_vector"], dtype=np.float32).reshape(-1)[:3],
                    np.asarray(raw["rotation_delta"], dtype=np.float32).reshape(-1)[:3],
                    _binarize_gripper_open(raw["open_gripper"]),
                ]
            )
            obs, _reward, terminated, truncated, _info = self._robot.step(action.tolist())
            done = bool(terminated or truncated)
            if done:
                break
        return done

    def _train(self, policy: Any, epoch: int, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Train a StarVLA policy with the official LIBERO Accelerate entrypoint."""
        policy_name = policy.name
        starvla_root = Path(options.get("starvla_root", "/Users/tongmeng/Desktop/codes/starVLA")).expanduser()
        config_yaml = options.get("config_yaml", "examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
        run_root_dir = Path(options.get("run_root_dir", self.train_dir)).expanduser().resolve()
        run_id = options.get("run_id", f"{policy_name}_{int(time.time())}")
        output_dir = run_root_dir / run_id
        final_checkpoint = output_dir / "final_model" / "pytorch_model.pt"

        command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            options.get("accelerate_config", "starVLA/config/deepseeds/deepspeed_zero2.yaml"),
            "--num_processes",
            str(options.get("num_processes", os.environ.get("NUM_PROCESSES", 1))),
            "starVLA/training/train_starvla.py",
            "--config_yaml",
            str(config_yaml),
            "--framework.name",
            str(options.get("framework_name", "QwenPI")),
            "--framework.qwenvl.base_vlm",
            str(options.get("base_vlm", "playground/Pretrained_models/Qwen3.5-0.8B")),
            "--datasets.vla_data.data_root_dir",
            str(options.get("data_root", "playground/Datasets/LEROBOT_LIBERO_DATA")),
            "--datasets.vla_data.data_mix",
            str(options.get("data_mix", "libero_all")),
            "--datasets.vla_data.per_device_batch_size",
            str(options.get("per_device_batch_size", 16)),
            "--trainer.vla_data.video_backend",
            str(options.get("video_backend", "torchvision_av")),
            "--trainer.freeze_modules",
            str(options.get("freeze_modules", "")),
            "--trainer.max_train_steps",
            str(epoch),
            "--trainer.save_interval",
            str(options.get("save_interval", 10000)),
            "--trainer.logging_frequency",
            str(options.get("logging_frequency", 100)),
            "--trainer.eval_interval",
            str(options.get("eval_interval", 100)),
            "--run_root_dir",
            str(run_root_dir),
            "--run_id",
            str(run_id),
            "--wandb_project",
            str(options.get("wandb_project", "starVLA_Libero")),
            "--wandb_entity",
            str(options.get("wandb_entity", "jinhuiye")),
        ]
        command.extend(map(str, options.get("overrides", [])))

        env = os.environ.copy()
        for key, default in {
            "NCCL_SOCKET_IFNAME": "bond0",
            "NCCL_IB_HCA": "mlx5_2,mlx5_3",
            "NCCL_BLOCKING_WAIT": "1",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_TIMEOUT": "10000",
            "NCCL_SOCKET_TIMEOUT_MS": "360000",
        }.items():
            env.setdefault(key, default)
        if options.get("wandb_mode"):
            env["WANDB_MODE"] = str(options["wandb_mode"])

        run_root_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=starvla_root, env=env, check=True)
        if not final_checkpoint.is_file():
            raise FileNotFoundError(f"StarVLA training completed without final checkpoint: {final_checkpoint}")

        summary_path = output_dir / "summary.jsonl"
        summary = None
        if summary_path.is_file():
            lines = [line for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                summary = json.loads(lines[-1])
        return (
            {
                "policy_name": policy_name,
                "stage": "vla",
                "epoch": epoch,
                "train_dir": str(output_dir),
                "checkpoint": str(final_checkpoint),
                "summary": summary,
            },
            {"ckpt_path": str(final_checkpoint)},
        )
