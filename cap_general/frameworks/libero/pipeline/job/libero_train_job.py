"""LIBERO train job using StarVLA's native training components."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cap_general.core.pipeline.job.base_job import BaseJob
from cap_general.core.pipeline.job.train_job import TrainJob, TrainJobConfig
from cap_general.core.policy.graph import CapNode

if TYPE_CHECKING:
    from cap_general.core.policy import BasePolicy


@dataclass
class LiberoTrainJobConfig(TrainJobConfig):
    """Configuration for a LiberoTrainJob."""

    starvla_root: str = "/Users/tongmeng/Desktop/codes/starVLA"
    accelerate_config: dict[str, Any] = field(default_factory=dict)
    deepspeed_config: dict[str, Any] = field(default_factory=dict)
    train_config: dict[str, Any] = field(default_factory=dict)


@BaseJob.register()
class LiberoTrainJob(TrainJob):
    """Train job that runs StarVLA fine-tuning via Accelerate.

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
        starvla_root = Path(self._config.starvla_root).expanduser()
        if str(starvla_root) not in sys.path:
            sys.path.insert(0, str(starvla_root))

        import torch
        import wandb
        from accelerate import Accelerator, DeepSpeedPlugin
        from accelerate.utils import set_seed
        from omegaconf import OmegaConf
        from starVLA.dataloader import build_dataloader
        from starVLA.model.framework.base_framework import build_framework
        from starVLA.model.framework.share_tools import apply_config_compat
        from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
        from starVLA.training.trainer_utils.trainer_tools import (
            TrainerUtils,
            normalize_dotlist_args,
            setup_optimizer_and_scheduler,
        )
        from tqdm import tqdm

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if options.get("wandb_mode"):
            os.environ["WANDB_MODE"] = str(options["wandb_mode"])

        requested_processes = int(
            options.get(
                "num_processes",
                self._config.accelerate_config.get("num_processes", os.environ.get("WORLD_SIZE", 1)),
            )
        )
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        visible_devices = torch.cuda.device_count()
        use_deepspeed = world_size > 1 or (requested_processes > 1 and visible_devices > 1)
        deepspeed_plugin = None
        if use_deepspeed:
            for key, default in {
                "NCCL_SOCKET_IFNAME": "bond0",
                "NCCL_IB_HCA": "mlx5_2,mlx5_3",
                "NCCL_BLOCKING_WAIT": "1",
                "NCCL_ASYNC_ERROR_HANDLING": "1",
                "NCCL_TIMEOUT": "10000",
                "NCCL_SOCKET_TIMEOUT_MS": "360000",
            }.items():
                os.environ.setdefault(key, default)
            deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self._config.deepspeed_config)

        def resolve_local_path(value: str) -> str:
            path = Path(value).expanduser()
            candidate = starvla_root / path
            return str(candidate.resolve()) if not path.is_absolute() and candidate.exists() else str(path)

        cfg = OmegaConf.create(self._config.train_config)
        runtime_cfg = OmegaConf.create(
            {
                "framework": {
                    "name": options.get("framework_name", "QwenPI"),
                    "qwenvl": {
                        "base_vlm": resolve_local_path(
                            options.get("base_vlm", "playground/Pretrained_models/Qwen3.5-0.8B")
                        )
                    },
                },
                "datasets": {
                    "vla_data": {
                        "data_root_dir": resolve_local_path(
                            options.get("data_root", "playground/Datasets/LEROBOT_LIBERO_DATA")
                        ),
                        "data_mix": options.get("data_mix", "libero_all"),
                        "per_device_batch_size": options.get("per_device_batch_size", 16),
                        "video_backend": options.get("video_backend", "torchvision_av"),
                    }
                },
                "trainer": {
                    "freeze_modules": options.get("freeze_modules", ""),
                    "max_train_steps": int(epoch),
                    "save_interval": int(options.get("save_interval", 10000)),
                    "logging_frequency": int(options.get("logging_frequency", 100)),
                    "eval_interval": int(options.get("eval_interval", 100)),
                },
                "wandb_project": options.get("wandb_project", "starVLA_Libero"),
                "wandb_entity": options.get("wandb_entity", "jinhuiye"),
            }
        )
        override_args = normalize_dotlist_args(list(map(str, options.get("overrides", []))))
        cfg = OmegaConf.merge(cfg, runtime_cfg, OmegaConf.from_dotlist(override_args))
        cfg = apply_config_compat(cfg)

        train_dir = Path(options.get("train_dir", self._config.export_dir)).expanduser().resolve()
        run_root_dir = Path(options.get("run_root_dir", train_dir)).expanduser().resolve()
        run_id = options.get("run_id", f"{policy_name}_{int(time.time())}")
        output_dir = run_root_dir / run_id
        final_checkpoint = output_dir / "final_model" / "pytorch_model.pt"
        cfg.run_root_dir = str(run_root_dir)
        cfg.run_id = str(run_id)
        cfg.output_dir = str(output_dir)
        cfg = wrap_config(cfg)

        mixed_precision = options.get("mixed_precision", "bf16" if torch.cuda.is_available() else "no")
        accelerator = Accelerator(
            gradient_accumulation_steps=int(cfg.trainer.gradient_accumulation_steps),
            mixed_precision=mixed_precision,
            deepspeed_plugin=deepspeed_plugin,
        )
        set_seed(int(getattr(cfg, "seed", 3047)) + accelerator.process_index)

        if accelerator.is_main_process:
            (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            full_cfg = cfg.unwrap() if isinstance(cfg, AccessTrackedConfig) else cfg
            OmegaConf.save(full_cfg, output_dir / "config.full.yaml", resolve=True)
        accelerator.wait_for_everyone()

        model = build_framework(cfg)
        dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

        completed_steps = 0
        checkpoint_path = None
        if getattr(cfg.trainer, "is_resume", False):
            checkpoints = list((output_dir / "checkpoints").glob("steps_*_pytorch_model.pt"))
            if checkpoints:
                checkpoint_path = max(
                    checkpoints,
                    key=lambda path: int(re.search(r"steps_(\d+)", path.name).group(1)),
                )
                completed_steps = int(re.search(r"steps_(\d+)", checkpoint_path.name).group(1))
        if checkpoint_path is None and getattr(cfg.trainer, "pretrained_checkpoint", None):
            checkpoint_path = Path(cfg.trainer.pretrained_checkpoint).expanduser()
            if not checkpoint_path.is_absolute():
                checkpoint_path = starvla_root / checkpoint_path
        if checkpoint_path is not None:
            if checkpoint_path.suffix == ".safetensors":
                from safetensors.torch import load_file

                state_dict = load_file(str(checkpoint_path))
            else:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)

        model = TrainerUtils.freeze_backbones(model, freeze_modules=cfg.trainer.freeze_modules)
        optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=model, cfg=cfg)
        for _ in range(completed_steps):
            lr_scheduler.step()
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

        if accelerator.is_main_process:
            wandb.init(
                name=cfg.run_id,
                dir=str(output_dir / "wandb"),
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                group="vla-train",
            )

        data_iterator = iter(dataloader)
        progress_bar = tqdm(
            total=int(cfg.trainer.max_train_steps),
            initial=completed_steps,
            disable=not accelerator.is_local_main_process,
        )
        while completed_steps < int(cfg.trainer.max_train_steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                batch = next(data_iterator)

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                with accelerator.autocast():
                    output_dict = model.forward(batch)
                    loss = output_dict["action_loss"]
                accelerator.backward(loss)
                if cfg.trainer.gradient_clipping is not None:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clipping)
                optimizer.step()
                if accelerator.sync_gradients:
                    lr_scheduler.step()

            if not accelerator.sync_gradients:
                continue
            completed_steps += 1
            progress_bar.update(1)

            metrics = {"action_dit_loss": float(loss.detach().item())}
            if completed_steps % int(cfg.trainer.eval_interval) == 0:
                try:
                    examples = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloader)
                    examples = next(data_iterator)
                actions = np.asarray([example["action"] for example in examples])
                prediction = accelerator.unwrap_model(model).predict_action(
                    examples=examples,
                    use_ddim=True,
                    num_ddim_steps=20,
                )["normalized_actions"]
                metrics["mse_score"] = float(np.linalg.norm(prediction - actions) / np.prod(actions.shape))

            if accelerator.is_main_process and completed_steps % int(cfg.trainer.logging_frequency) == 0:
                metrics["epoch"] = round(completed_steps / len(dataloader), 2)
                wandb.log(metrics, step=completed_steps)

            if completed_steps % int(cfg.trainer.save_interval) == 0:
                checkpoint = output_dir / "checkpoints" / f"steps_{completed_steps}_pytorch_model.pt"
                if accelerator.is_main_process:
                    torch.save(accelerator.get_state_dict(model), checkpoint)
                    with (output_dir / "summary.jsonl").open("a", encoding="utf-8") as file:
                        file.write(json.dumps({"steps": completed_steps}) + "\n")
                    if isinstance(cfg, AccessTrackedConfig):
                        cfg.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                accelerator.wait_for_everyone()

        progress_bar.close()
        if accelerator.is_main_process:
            final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(accelerator.get_state_dict(model), final_checkpoint)
            if isinstance(cfg, AccessTrackedConfig):
                cfg.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            wandb.finish()
        accelerator.wait_for_everyone()

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
            "base_vlm: base VLM path or model identifier\n"
            "data_root: LIBERO LeRobot dataset root\n"
            "data_mix: dataset mixture name (default libero_all)\n"
            "run_root_dir: parent directory for training outputs\n"
            "run_id: training run identifier\n"
            "num_processes: requested worker count; DeepSpeed is used only when multiple devices are available\n"
            "mixed_precision: Accelerator precision mode (default bf16 on CUDA, otherwise no)\n"
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
