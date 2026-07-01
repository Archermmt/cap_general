"""Shared agent helpers for Genesis environments."""

from __future__ import annotations

from typing import Any

from cap_general.core.agent import BaseAgent
from cap_general.core.utils import tensor_mean_value, tensor_to_scalar


class GenesisBaseAgent(BaseAgent):
    """Base class for agents backed by a Genesis robot environment."""

    agent_type = "genesis_base"

    def init_genesis(self, gs_scene: Any) -> None:
        self._robot.init_genesis(gs_scene)

    def _options_doc(self, method_name: str) -> str:
        if method_name != "train":
            return super()._options_doc(method_name)
        return (
            "seed: Random seed metadata for the training run (default 1).\n"
            "train_cfg: Dict of overrides for the training config.\n"
            "record_epoch: Training-epoch interval for summary history samples (default 50).\n"
            "summary_interval: Backward-compatible alias for record_epoch."
        )

    @staticmethod
    def _summary(*, stage: str, interval: int) -> dict[str, Any]:
        return {
            "stage": stage,
            "sampling_interval": max(int(interval), 1),
            "history": [],
            "latest": None,
            "best": None,
            "available_metrics": [],
            "notes": [],
        }

    @staticmethod
    def _merge_summary(
        summary: dict[str, Any],
        point: dict[str, Any],
        *,
        best_metric: str,
        best_mode: str,
    ) -> None:
        clean = {key: tensor_to_scalar(value) for key, value in point.items()}
        clean = {key: value for key, value in clean.items() if value is not None}
        if not clean:
            return
        summary["latest"] = dict(clean)
        summary["available_metrics"] = sorted({*summary["available_metrics"], *clean})

        best_value = clean.get(best_metric)
        current = summary.get("best")
        if not isinstance(best_value, (int, float)):
            return
        if current is not None and isinstance(current.get("value"), (int, float)):
            if best_mode == "max" and float(best_value) <= float(current["value"]):
                return
            if best_mode == "min" and float(best_value) >= float(current["value"]):
                return
        summary["best"] = {
            "by": best_metric,
            "mode": best_mode,
            "iteration": clean.get("iteration"),
            "total_steps": clean.get("total_steps"),
            "value": float(best_value),
        }

    @staticmethod
    def _episode_metrics(ep_extras: list[dict[str, Any]]) -> dict[str, Any]:
        aggregated: dict[str, list[float]] = {}
        for ep_info in ep_extras:
            if not isinstance(ep_info, dict):
                continue
            for key, value in ep_info.items():
                scalar = tensor_mean_value(value)
                if scalar is None:
                    continue
                metric_name = f"mean_episode_{str(key).replace('/', '_')}"
                aggregated.setdefault(metric_name, []).append(float(scalar))
        return {key: sum(values) / len(values) for key, values in aggregated.items() if values}

    def _capture_rl_summary(
        self,
        runner: Any,
        *,
        interval: int,
        num_learning_iterations: int,
        best_metric: str,
        init_at_random_ep_len: bool = True,
    ) -> dict[str, Any]:
        summary = self._summary(stage="rl", interval=interval)
        step_interval = (
            int(runner.cfg["num_steps_per_env"])
            * int(runner.env.num_envs)
            * int(getattr(runner, "gpu_world_size", 1))
        )
        summary["notes"].append("RL metrics sampled from runner logger once per training iteration")
        next_sample_iteration = summary["sampling_interval"]
        original_log = runner.logger.log

        def wrapped_log(*args: Any, **kwargs: Any) -> Any:
            nonlocal next_sample_iteration
            extras = self._episode_metrics(list(getattr(runner.logger, "ep_extras", [])))
            result = original_log(*args, **kwargs)
            collect_time = float(kwargs.get("collect_time", 0.0))
            learn_time = float(kwargs.get("learn_time", 0.0))
            iteration_time = collect_time + learn_time
            done_iterations = int(kwargs.get("it", 0)) + 1 - int(kwargs.get("start_it", 0))
            remaining_iterations = int(kwargs.get("total_it", 0)) - int(kwargs.get("start_it", 0)) - done_iterations
            point = {
                "stage": "rl",
                "iteration": int(kwargs.get("it", 0)) + 1,
                "total_steps": int(getattr(runner.logger, "tot_timesteps", 0)),
                "steps_per_second": int(step_interval / iteration_time) if iteration_time > 0 else None,
                "collection_time": collect_time,
                "learning_time": learn_time,
                "iteration_time": iteration_time,
                "time_elapsed_seconds": float(getattr(runner.logger, "tot_time", 0.0)),
                "eta_seconds": (
                    float(getattr(runner.logger, "tot_time", 0.0)) / done_iterations * max(remaining_iterations, 0)
                    if done_iterations > 0
                    else None
                ),
                "mean_action_std": tensor_mean_value(kwargs.get("action_std")),
                "mean_reward": tensor_mean_value(getattr(runner.logger, "rewbuffer", None)),
                "mean_episode_length": tensor_mean_value(getattr(runner.logger, "lenbuffer", None)),
            }
            for key, value in (kwargs.get("loss_dict", {}) or {}).items():
                point[f"mean_{key}_loss"] = tensor_mean_value(value)
            point.update(extras)
            self._merge_summary(summary, point, best_metric=best_metric, best_mode="max")
            iteration = point["iteration"]
            if isinstance(iteration, int) and iteration >= next_sample_iteration:
                summary["history"].append(dict(summary["latest"]))
                while iteration >= next_sample_iteration:
                    next_sample_iteration += summary["sampling_interval"]
            return result

        runner.logger.log = wrapped_log
        try:
            runner.learn(
                num_learning_iterations=num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
        finally:
            runner.logger.log = original_log
        if summary["latest"] is not None and (
            not summary["history"] or summary["history"][-1].get("iteration") != summary["latest"].get("iteration")
        ):
            summary["history"].append(dict(summary["latest"]))
        return summary

    @staticmethod
    def _load_model_to_runner(model: Any, runner: Any, env: Any) -> None:
        """Load model weights into an RSL-RL or behavior-cloning runner."""
        if hasattr(runner, "alg"):
            runner_policy = runner.alg.get_policy().to(env.device)
        else:
            runner_policy = runner._policy
        runner_policy.load_state_dict(model.state_dict())
