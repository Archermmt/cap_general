"""LIBERO train job — StarVLA Accelerate training subprocess."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cap_general.core.graph.cap_node import CapNode
from cap_general.core.pipeline.job.base_job import BaseJob
from cap_general.core.pipeline.job.train_job import TrainJob, TrainJobConfig

if TYPE_CHECKING:
    from cap_general.core.policy import BasePolicy


@dataclass
class LiberoTrainJobConfig(TrainJobConfig):
    """Configuration for a LiberoTrainJob."""


@BaseJob.register()
class LiberoTrainJob(TrainJob):
    """Train job that launches StarVLA fine-tuning via Accelerate.

    Reads the checkpoint path produced by training and writes it back into
    every policy graph node's config, then returns the updated graph dict.
    """

    job_type = "libero_train"
    config_cls = LiberoTrainJobConfig

    def _execute(
        self,
        policy: BasePolicy,
        robot: Any,
        options: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Run StarVLA training and return ``(policy_config_dict, report)``."""
        epoch = options.get("epoch", self._config.epoch)
        policy_name = policy.name

        starvla_root = Path(options.get("starvla_root", "/Users/archer/Desktop/codes/starVLA")).expanduser()
        config_yaml = options.get("config_yaml", "examples/LIBERO/train_files/starvla_cotrain_libero.yaml")

        train_dir = Path(options.get("train_dir", self._config.export_dir)).expanduser().resolve()
        run_root_dir = Path(options.get("run_root_dir", train_dir)).expanduser().resolve()
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

        policy_dict = policy.to_dict()
        for node_dict in policy_dict["graph"]["nodes"]:
            if CapNode.is_group(node_dict, "model"):
                node_dict["config"]["ckpt_path"] = str(final_checkpoint)
        report = {"ckpt_path": str(final_checkpoint), "run_id": run_id, "output_dir": str(output_dir)}
        return policy_dict, report

    def options_doc(self) -> str:
        return (
            "epoch: max training steps (default: config value)\n"
            "starvla_root: StarVLA repository root\n"
            "config_yaml: training YAML path relative to starvla_root\n"
            "base_vlm: base VLM path or model identifier\n"
            "data_root: LIBERO LeRobot dataset root\n"
            "data_mix: dataset mixture name (default libero_all)\n"
            "run_root_dir: parent directory for training outputs\n"
            "run_id: training run identifier\n"
            "num_processes: number of Accelerate workers (default 1)\n"
            "per_device_batch_size: per-device VLA batch size (default 16)\n"
            "save_interval: checkpoint interval in steps (default 10000)\n"
            "logging_frequency: logging interval in steps (default 100)\n"
            "eval_interval: evaluation interval in steps (default 100)\n"
            "freeze_modules: space-separated module names to freeze\n"
            "wandb_project: Weights & Biases project name\n"
            "wandb_entity: Weights & Biases entity name\n"
            "wandb_mode: optional WANDB_MODE environment value\n"
            "overrides: additional StarVLA dotlist arguments"
        )
