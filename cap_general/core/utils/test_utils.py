"""Shared test helpers."""

from __future__ import annotations

import json
from typing import Any

from cap_general.core.utils.config import parse_cli_overrides


def print_execution_summary(prefix: str, result: dict[str, Any]) -> None:
    print(
        f"{prefix} Execution summary: "
        f"steps: {result.get('step_start')}..{result.get('step_end')}, "
        f"duration: {result.get('duration')}, "
        f"reward: {result.get('reward')}"
    )


def print_train_summary(prefix: str, result: dict[str, Any]) -> None:
    train_result = result.get("result", {})
    print(
        f"{prefix} Train summary: "
        f"epoch: {result.get('train_epoch')}, "
        f"method: {train_result.get('method')}, "
        f"epoch: {train_result.get('epoch')}, "
        f"dir: {train_result.get('train_dir')}"
    )


def print_record(prefix: str, record: dict[str, Any]) -> None:
    info = record.get("info", {})
    print(
        f"\n{prefix} Record summary: "
        f"step: {info.get('total_step')}, "
        f"duration: {info.get('total_duration')}, "
        f"execute: {info.get('total_execute')}"
    )


def single_agent_result(results: dict[str, Any]) -> Any:
    """Return the only value from a single-agent scene response."""
    if len(results) != 1:
        raise ValueError(f"Expected one agent result, got {list(results)}")
    return next(iter(results.values()))


def parse_args_with_config_overrides(parser: Any) -> tuple[Any, list[str]]:
    """Parse known test arguments and return remaining config overrides."""
    args, override_args = parser.parse_known_args()
    return args, parse_cli_overrides(override_args)


async def call_tool(session: Any, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    result = await session.call_tool(name, args or {})
    if not result.content:
        return {}
    try:
        return json.loads(result.content[0].text)
    except json.JSONDecodeError:
        return {"_raw": result.content[0].text}
