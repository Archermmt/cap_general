"""RSL-RL inference policy."""

from __future__ import annotations

import copy
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cap_general.core.policy.base_policy import BasePolicy, BasePolicyConfig

if TYPE_CHECKING:
    from logging import Logger


@dataclass
class RslRlPolicyConfig(BasePolicyConfig):
    """Configuration for loading an RSL-RL OnPolicyRunner checkpoint."""

    log_dir: str | Path = "logs"
    ckpt: int | None = None
    checkpoint_pattern: str = "model_*.pt"
    train_cfg_index: int = 4
    cfgs_filename: str = "cfgs.pkl"
    device: str | None = None


@BasePolicy.register()
class RslRlPolicy(BasePolicy):
    """Load and run an RSL-RL inference policy."""

    name = "RSL-RL Policy"
    config_cls = RslRlPolicyConfig

    def __init__(self, config: RslRlPolicyConfig, logger: Logger):
        super().__init__(config=config, logger=logger)
        self._policy = self._load_policy()

    @classmethod
    def policy_type(cls) -> str:
        return "rsl_rl"

    def reset(self, *args: Any, **kwargs: Any) -> None:
        """Reset the loaded policy to evaluation mode."""
        self._policy.eval()

    def inference(self, obs: Any = None) -> Any:
        """Run inference for an observation tensor dict."""
        obs_device = getattr(obs, "device", None)
        if obs_device is not None:
            self._policy.to(obs_device)
        self._policy.eval()
        return self._policy(obs)

    def update(self, *, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Load trained weights into the current policy."""
        self._policy.load_state_dict(state_dict)
        return {}

    def _load_policy(self) -> Any:
        try:
            import torch
            from rsl_rl.utils import resolve_callable
            from tensordict import TensorDict
        except ImportError as exc:
            raise ImportError("RslRlPolicy requires torch, tensordict, and rsl-rl-lib") from exc

        log_dir = Path(self._config.log_dir).expanduser()
        with (log_dir / self._config.cfgs_filename).open("rb") as file:
            train_cfg = copy.deepcopy(pickle.load(file)[self._config.train_cfg_index])
        actor_cfg = train_cfg["actor"]
        actor_class = resolve_callable(actor_cfg.pop("class_name"))
        checkpoint = torch.load(
            self._checkpoint_path(log_dir),
            map_location=self._config.device or "cpu",
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
        actor_obs_groups = train_cfg["obs_groups"]["actor"]
        if len(actor_obs_groups) != 1:
            raise ValueError("RslRlPolicy requires exactly one actor observation group")
        device = self._config.device or "cpu"
        obs = TensorDict(
            {actor_obs_groups[0]: torch.zeros((1, input_dim), device=device)},
            batch_size=[1],
        )
        policy = actor_class(obs, train_cfg["obs_groups"], "actor", output_dim, **actor_cfg).to(device)
        policy.load_state_dict(actor_state)
        policy.eval()
        return policy

    def _checkpoint_path(self, log_dir: Path) -> Path:
        if self._config.ckpt is not None:
            return log_dir / f"model_{self._config.ckpt}.pt"
        checkpoint_files = list(log_dir.glob(self._config.checkpoint_pattern))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
        return max(checkpoint_files, key=self._checkpoint_number)

    @staticmethod
    def _checkpoint_number(path: Path) -> int:
        match = re.search(r"\d+", path.stem)
        return int(match.group()) if match else -1
