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
    ckpt_step: int = -1
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
        if self._actor is None:
            raise RuntimeError("RslRlOp has no actor weights; train or configure an existing ckpt_dir first")
        obs = inputs["obs"]
        obs_device = getattr(obs, "device", None)
        if obs_device is not None:
            self._actor.to(obs_device)
        self._actor.eval()
        return self._actor(obs)

    @to_stage_fn
    def update(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._actor is None:
            raise RuntimeError("RslRlOp has no actor to update")
        self._actor.load_state_dict(inputs["state_dict"])
        return {}

    def _load_actor(self) -> Any | None:
        ckpt_dir = Path(self._config.ckpt_dir).expanduser() if self._config.ckpt_dir else None
        if ckpt_dir is None or not ckpt_dir.is_dir():
            return None

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
        checkpoint = torch.load(
            self._checkpoint_path(ckpt_dir),
            map_location=device,
            weights_only=False,
        )
        actor_state = checkpoint["actor_state_dict"]
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
        actor_obs_groups = obs_groups["actor"]
        if len(actor_obs_groups) != 1:
            raise ValueError("RslRlOp requires exactly one actor observation group")
        obs = TensorDict(
            {actor_obs_groups[0]: torch.zeros((1, input_dim), device=device)},
            batch_size=[1],
        )
        actor = actor_class(obs, obs_groups, "actor", output_dim, **actor_cfg).to(device)
        actor.load_state_dict(actor_state)
        actor.eval()
        return actor

    def _checkpoint_path(self, ckpt_dir: Path) -> Path:
        ckpt_step = self._config.ckpt_step
        if ckpt_step == -1:
            checkpoint_files = list(ckpt_dir.glob(self._config.checkpoint_pattern))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
            return max(checkpoint_files, key=self._checkpoint_number)
        return ckpt_dir / f"model_{ckpt_step}.pt"

    @staticmethod
    def _checkpoint_number(path: Path) -> int:
        match = re.search(r"\d+", path.stem)
        return int(match.group()) if match else -1
