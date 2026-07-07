"""Genesis policy evaluation job."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cap_general.core.pipeline.job.eval_job import EvalJob, EvalJobConfig
from cap_general.core.utils import tensor_to_scalar

if TYPE_CHECKING:
    from cap_general.core.policy import BasePolicy
    from cap_general.core.robot import BaseRobot


@dataclass
class GenesisEvalJobConfig(EvalJobConfig):
    """Configuration for Genesis policy evaluation."""

    stage: str = "rl"
    max_steps: int = 100


@EvalJob.register()
class GenesisEvalJob(EvalJob):
    """Evaluate a Genesis RL or BC policy using each episode's final reward."""

    job_type = "genesis_eval"
    config_cls = GenesisEvalJobConfig

    def _execute(
        self,
        policy: BasePolicy,
        robot: BaseRobot,
        options: dict[str, Any],
    ) -> tuple[None, dict[str, Any]]:
        stage = str(options.get("stage", self._config.stage))
        epoch = int(options.get("epoch", self._config.epoch))
        max_steps = int(options.get("max_steps", self._config.max_steps))
        if stage not in {"rl", "bc"}:
            raise ValueError(f"Unsupported Genesis evaluation stage: {stage!r}")
        if epoch <= 0:
            raise ValueError("Genesis evaluation epoch must be greater than zero")
        if max_steps <= 0:
            raise ValueError("Genesis evaluation max_steps must be greater than zero")

        rewards: list[float] = []
        for _ in range(epoch):
            robot.reset()
            obs = robot.policy_obs
            final_reward = 0.0
            for _ in range(max_steps):
                if stage == "rl":
                    inputs = {"obs": obs}
                else:
                    inputs = {
                        "env": robot,
                        "rgb_obs": robot.get_stereo_rgb_images(normalize=True).float(),
                        "ee_pose": robot.robot.ee_pose.float(),
                    }
                result = policy.run("inference", inputs)
                if not result.success:
                    raise RuntimeError(f"Policy evaluation failed for stage {stage!r}")
                obs, reward, terminated, truncated, _info = robot.step(result.output)
                scalar_reward = tensor_to_scalar(reward)
                final_reward = float(scalar_reward) if isinstance(scalar_reward, (int, float)) else 0.0
                if self._is_done(terminated) or self._is_done(truncated):
                    break
                obs = robot.policy_obs
            rewards.append(final_reward)

        return None, {
            "average_reward": sum(rewards) / len(rewards),
            "rewards": rewards,
            "epoch": epoch,
            "stage": stage,
        }

    @staticmethod
    def _is_done(done: Any) -> bool:
        if hasattr(done, "any"):
            done = done.any()
        if hasattr(done, "item"):
            done = done.item()
        return bool(done)

    def options_doc(self) -> str:
        return (
            "epoch: number of evaluation episodes (default: config value)\n"
            "stage: 'rl' or 'bc' policy input mode (default: config value)\n"
            "max_steps: maximum steps per evaluation episode (default: config value)"
        )
