"""RSL-RL model operator."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cap_general.core.operator.base_operator import BaseOperator, to_stage_fn
from cap_general.core.operator.model.base_model_op import ModelOp


@dataclass
class RslRlConfig:
    """Configuration for RslRlOp."""

    ckpt_dir: str | Path | None = "logs"
    ckpt_path: str | Path | None = None
    ckpt_step: int = -1
    obs_dim: int | None = None
    action_dim: int | None = None
    actor_cfg: dict[str, Any] | None = None
    obs_groups: dict[str, Any] | None = None
    checkpoint_pattern: str = "model_*.pt"
    device: str | None = None


@BaseOperator.register()
class RslRlOp(ModelOp):
    """Load and run an RSL-RL inference actor directly."""

    op_type = "rsl_rl"
    config_cls = RslRlConfig

    def reset(self) -> None:
        self._actor = self._load_actor()
        super().reset()

    def get_model(self) -> Any:
        return self._actor

    @to_stage_fn
    def inference(self, inputs: dict[str, Any]) -> dict[str, Any]:
        obs = inputs["obs"]
        obs_device = getattr(obs, "device", None)
        if obs_device is not None:
            self._actor.to(obs_device)
        self._actor.eval()
        return self._actor(obs)

    @to_stage_fn
    def update(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self._actor.load_state_dict(inputs["state_dict"])
        return {}

    def _load_actor(self) -> Any:
        try:
            import torch
            from rsl_rl.utils import resolve_callable
            from tensordict import TensorDict
        except ImportError as exc:
            raise ImportError("RslRlOp requires torch, tensordict, and rsl-rl-lib") from exc

        device = self._config.device or "cpu"

        if self._config.actor_cfg is None:
            raise ValueError("RslRlOp requires actor_cfg in config")
        if self._config.obs_groups is None:
            raise ValueError("RslRlOp requires obs_groups in config")
        actor_cfg = copy.deepcopy(self._config.actor_cfg)
        obs_groups = copy.deepcopy(self._config.obs_groups)
        actor_class = resolve_callable(actor_cfg.pop("class_name"))
        checkpoint_path = self._resolve_checkpoint_path()
        actor_state = None
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            actor_state = checkpoint["actor_state_dict"]

        if actor_state is not None:
            input_dim, output_dim = self._dimensions_from_state(actor_state)
        else:
            if self._config.obs_dim is None or self._config.action_dim is None:
                raise ValueError("RslRlOp requires obs_dim and action_dim when no checkpoint can be loaded")
            input_dim = int(self._config.obs_dim)
            output_dim = int(self._config.action_dim)
        actor_obs_groups = obs_groups["actor"]
        if len(actor_obs_groups) != 1:
            raise ValueError("RslRlOp requires exactly one actor observation group")
        obs = TensorDict(
            {actor_obs_groups[0]: torch.zeros((1, input_dim), device=device)},
            batch_size=[1],
        )
        actor = actor_class(obs, obs_groups, "actor", output_dim, **actor_cfg).to(device)
        if actor_state is not None:
            actor.load_state_dict(actor_state)
        actor.eval()
        return actor

    def _resolve_checkpoint_path(self) -> Path | None:
        if self._config.ckpt_path:
            checkpoint_path = Path(self._config.ckpt_path).expanduser()
            if checkpoint_path.is_file():
                return checkpoint_path
            self._logger.warning(
                "RslRlOp checkpoint path does not exist: %s; using randomly initialized actor weights",
                checkpoint_path,
            )
            return None

        if not self._config.ckpt_dir:
            self._logger.warning("RslRlOp checkpoint is not configured; using randomly initialized actor weights")
            return None
        ckpt_dir = Path(self._config.ckpt_dir).expanduser()
        if not ckpt_dir.is_dir():
            self._logger.warning(
                "RslRlOp checkpoint directory does not exist: %s; using randomly initialized actor weights",
                ckpt_dir,
            )
            return None
        checkpoint_path = self._checkpoint_path(ckpt_dir)
        if not checkpoint_path.is_file():
            self._logger.warning(
                "RslRlOp checkpoint was not found in %s; using randomly initialized actor weights",
                ckpt_dir,
            )
            return None
        return checkpoint_path

    @staticmethod
    def _dimensions_from_state(actor_state: dict[str, Any]) -> tuple[int, int]:
        mlp_weights = sorted(
            (
                (int(key.split(".")[1]), value)
                for key, value in actor_state.items()
                if key.startswith("mlp.") and key.endswith(".weight") and value.ndim == 2
            ),
            key=lambda item: item[0],
        )
        if not mlp_weights:
            raise ValueError("RSL-RL checkpoint contains no MLP actor weights")
        input_dim = int(mlp_weights[0][1].shape[1])
        output_dim = int(actor_state.get("distribution.std_param", mlp_weights[-1][1]).shape[0])
        return input_dim, output_dim

    def _checkpoint_path(self, ckpt_dir: Path) -> Path:
        ckpt_step = self._config.ckpt_step
        if ckpt_step == -1:
            checkpoint_files = list(ckpt_dir.glob(self._config.checkpoint_pattern))
            if not checkpoint_files:
                return ckpt_dir / "__missing_checkpoint__"
            return max(checkpoint_files, key=self._checkpoint_number)
        return ckpt_dir / f"model_{ckpt_step}.pt"

    @staticmethod
    def _checkpoint_number(path: Path) -> int:
        match = re.search(r"\d+", path.stem)
        return int(match.group()) if match else -1
