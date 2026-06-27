"""Genesis grasp behavior-cloning policy wrapper."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cap_general.core.policy import BasePolicy, BasePolicyConfig
from cap_general.frameworks.genesis.utils import load_module_from_file

if TYPE_CHECKING:
    from logging import Logger


@dataclass
class BehaviorCloningPolicyConfig(BasePolicyConfig):
    """Configuration for Genesis manipulation BehaviorCloning checkpoints."""

    example_root: str | Path = "/Users/tongmeng/Desktop/codes/genesis-world/examples/manipulation"
    log_dir: str | Path = "logs/grasp_bc"
    checkpoint: str | Path | None = None
    cfgs_filename: str = "cfgs.pkl"
    bc_cfg_index: int = 4
    device: str | None = None


@BasePolicy.register()
class BehaviorCloningPolicy(BasePolicy):
    """Load and run Genesis manipulation behavior-cloning policies."""

    name = "Genesis Behavior Cloning Policy"
    config_cls = BehaviorCloningPolicyConfig

    def __init__(self, config: BehaviorCloningPolicyConfig, logger: Logger):
        super().__init__(config=config, logger=logger)
        self._policy = None
        self._loaded_env_id: int | None = None

    @classmethod
    def policy_type(cls) -> str:
        return "genesis_behavior_cloning"

    def reset(self, *args: Any, **kwargs: Any) -> None:
        """Clear loaded checkpoint; policy is restored lazily for the active env."""
        self._policy = None
        self._loaded_env_id = None

    def inference(self, *, env: Any, rgb_obs: Any, ee_pose: Any) -> Any:
        """Run the BC policy on stereo images and end-effector pose."""
        self._ensure_loaded(env)
        return self._policy(rgb_obs, ee_pose)

    def _ensure_loaded(self, env: Any) -> None:
        if self._policy is not None and self._loaded_env_id == id(env):
            return
        try:
            import genesis as gs
        except ImportError as exc:
            raise ImportError("BehaviorCloningPolicy requires genesis") from exc

        import pickle

        example_root = Path(self._config.example_root).expanduser()
        module = load_module_from_file(
            "cap_general_genesis_behavior_cloning",
            example_root / "behavior_cloning.py",
        )
        log_dir = Path(self._config.log_dir).expanduser()
        with (log_dir / self._config.cfgs_filename).open("rb") as file:
            cfgs = pickle.load(file)
        bc_cfg = cfgs[self._config.bc_cfg_index]
        bc_runner = module.BehaviorCloning(env, bc_cfg, None, device=self._config.device or gs.device)
        bc_runner.load(str(self._checkpoint_path(log_dir)))
        self._policy = bc_runner._policy
        self._policy.eval()
        self._loaded_env_id = id(env)

    def _checkpoint_path(self, log_dir: Path) -> Path:
        if self._config.checkpoint is not None:
            return Path(self._config.checkpoint).expanduser()
        checkpoint_files = list(log_dir.glob("checkpoint_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
        return max(checkpoint_files, key=self._checkpoint_number)

    @staticmethod
    def _checkpoint_number(path: Path) -> int:
        match = re.search(r"\d+", path.stem)
        return int(match.group()) if match else -1
