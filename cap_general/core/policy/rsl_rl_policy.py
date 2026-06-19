"""RSL-RL inference policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cap_general.core.policy.base_policy import BasePolicy, BasePolicyConfig


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

    def __init__(self, config: RslRlPolicyConfig, logger=None):
        super().__init__(config=config, logger=logger)
        self._policy = None
        self._loaded_env_id: int | None = None
        self._train_cfg: dict[str, Any] | None = None

    @classmethod
    def policy_type(cls) -> str:
        return "rsl_rl"

    def reset(self, *args: Any, **kwargs: Any) -> None:
        """Clear loaded inference policy; it is reloaded lazily for the active env."""
        self._policy = None
        self._loaded_env_id = None
        self._train_cfg = None

    def inference(self, obs: Any = None, *, env: Any | None = None) -> Any:
        """Run inference for an observation tensor dict."""
        if env is None:
            raise ValueError("RslRlPolicy.inference requires env=...")
        self._ensure_loaded(env)
        return self._policy(obs)

    def _ensure_loaded(self, env: Any) -> None:
        if self._policy is not None and self._loaded_env_id == id(env):
            return
        try:
            import genesis as gs
            from rsl_rl.runners import OnPolicyRunner
        except ImportError as exc:
            raise ImportError("RslRlPolicy requires genesis and rsl-rl-lib") from exc

        log_dir = Path(self._config.log_dir).expanduser()
        train_cfg = self._load_train_cfg(log_dir)
        runner_device = self._config.device or gs.device
        runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=runner_device)
        runner.load(str(self._checkpoint_path(log_dir)))
        self._policy = runner.get_inference_policy(device=runner_device)
        self._loaded_env_id = id(env)
        self._train_cfg = train_cfg

    def _load_train_cfg(self, log_dir: Path) -> dict[str, Any]:
        import pickle

        with (log_dir / self._config.cfgs_filename).open("rb") as file:
            cfgs = pickle.load(file)
        return cfgs[self._config.train_cfg_index]

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
