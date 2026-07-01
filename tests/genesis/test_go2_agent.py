"""Test Go2Agent locally or remotely through MCP.

Local mode:
    python tests/genesis/test_go2_agent.py

Remote mode:
    capcmd server --config configs/genesis/go2_agent.yaml
    python tests/genesis/test_go2_agent.py --remote --config configs/genesis/go2_agent.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cap_general.core.utils import test_utils

_DEFAULT_MAX_STEPS = 100
_DEFAULT_CONFIG = "configs/genesis/go2_agent.yaml"
_DEFAULT_TASK_NUM = 5
_DEFAULT_AGENT = "go2"
ROUND_NUM = 10


def test_go2_config_initial_pose_matches_eval() -> None:
    """Go2 should start where the genesis-world eval viewer is looking."""
    import yaml

    with Path(_DEFAULT_CONFIG).open() as file:
        config = yaml.safe_load(file)
    robot_cfg = config["agents"][0]["config"]["robot"]
    assert robot_cfg["base_init_pos"] == [0.0, 0.0, 0.42]
    assert robot_cfg["camera_pos"] == [2.0, 0.0, 2.5]
    assert robot_cfg["camera_lookat"] == [0.0, 0.0, 0.5]
    assert robot_cfg["camera_attach_to_base"] is False
    assert config["viewer_options"]["camera_pos"] == [2.0, 0.0, 2.5]
    assert config["viewer_options"]["camera_lookat"] == [0.0, 0.0, 0.5]


def _random_turn_angles(task_num: int) -> list[float]:
    """Create turn angles that complete one circle every ROUND_NUM tasks."""
    if task_num <= 0:
        return []
    return [2.0 * math.pi * (idx % ROUND_NUM) / ROUND_NUM for idx in range(task_num)]


def _make_code(max_steps: int, turn_angle: float = 0.0) -> str:
    """Build Go2Agent execution code with values baked in."""
    return f"""\
walk_result = walk_forward(max_steps={max_steps}, turn_angle={turn_angle!r})
stand_result = stand_still(time_s=3.0)
RESULT = {{
    "success": True,
    "steps": walk_result.get("steps"),
    "turn_angle": walk_result.get("turn_angle"),
    "stand_duration": stand_result.get("duration"),
    "stand_steps": stand_result.get("steps"),
    "mock": walk_result.get("mock", False) or stand_result.get("mock", False),
}}
"""


def _make_train_request(train_ep: int) -> dict:
    """Build a lightweight Go2Agent training request for smoke tests."""
    return {"policy_name": "runner", "epoch": train_ep, "options": {"record_epoch": 50}}


def _make_local_scene(config: str, config_overrides: list[str] | None = None):
    import cap_general.frameworks.genesis  # noqa: F401
    from cap_general.core.scene import BaseScene

    return BaseScene.from_yaml(config, overrides=config_overrides)


async def _run_local(
    config: str,
    max_steps: int,
    task_num: int,
    train_ep: int,
    config_overrides: list[str] | None = None,
) -> dict:
    """Run Go2Agent episodes in-process."""
    scene = _make_local_scene(config, config_overrides)
    scene.reset({_DEFAULT_AGENT: {}})
    if train_ep > 0:
        print("\n[test] --- Train smoke test ---")
        await scene.train({_DEFAULT_AGENT: _make_train_request(train_ep)})
        status = await scene.monitor([_DEFAULT_AGENT])
        result = test_utils.single_agent_result(status)["result"]
        if not result.get("ok", False):
            raise AssertionError(result.get("error") or result)
        test_utils.print_train_summary("[test]", result)
    print(f"[test] agent_doc {test_utils.single_agent_result(scene.agent_doc([_DEFAULT_AGENT]))}")
    turn_angles = _random_turn_angles(task_num)
    for task_idx, turn_angle in enumerate(turn_angles):
        print(f"\n[test] --- Task {task_idx + 1}/{task_num}: turn_angle={turn_angle:.3f} ---")
        await scene.execute({_DEFAULT_AGENT: _make_code(max_steps, turn_angle=turn_angle)})
        status = await scene.monitor([_DEFAULT_AGENT])
        result = test_utils.single_agent_result(status)["result"]
        test_utils.print_execution_summary("[test]", result)
    record = test_utils.single_agent_result(scene.record([_DEFAULT_AGENT]))
    test_utils.print_record("[test]", record)
    return record


async def _run_remote(
    config: str,
    max_steps: int,
    task_num: int,
    train_ep: int,
    config_overrides: list[str] | None = None,
) -> dict:
    """Run Go2Agent episodes through MCP."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    from cap_general.core.scene import BaseScene

    url = BaseScene.get_server_url(config, overrides=config_overrides)
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_names = [tool.name for tool in (await session.list_tools()).tools]
            print(f"[mcp_test]({url}) Available tools: {tool_names}")
            await test_utils.call_tool(session, "reset", {"agent_options": {_DEFAULT_AGENT: {}}})
            if train_ep > 0:
                print("\n[mcp_test] --- Train smoke test ---")
                await test_utils.call_tool(
                    session,
                    "train",
                    {"agent_options": {_DEFAULT_AGENT: _make_train_request(train_ep)}},
                )
                status = await test_utils.call_tool(session, "monitor", {"agents": [_DEFAULT_AGENT]})
                result = test_utils.single_agent_result(status)["result"]
                if not result.get("ok", False):
                    raise AssertionError(result.get("error") or result)
                test_utils.print_train_summary("[mcp_test]", result)
            agent_doc = await test_utils.call_tool(session, "agent_doc", {"agents": [_DEFAULT_AGENT]})
            agent_doc = test_utils.single_agent_result(agent_doc)
            print(f"[mcp_test] agent_doc {agent_doc}")
            turn_angles = _random_turn_angles(task_num)
            for task_idx, turn_angle in enumerate(turn_angles):
                print(f"\n[mcp_test] --- Task {task_idx + 1}/{task_num}: turn_angle={turn_angle:.3f} ---")
                result = await test_utils.call_tool(
                    session,
                    "execute",
                    {"agent_codes": {_DEFAULT_AGENT: _make_code(max_steps, turn_angle=turn_angle)}},
                )
                result = await test_utils.call_tool(
                    session, "monitor", {"agents": [_DEFAULT_AGENT]}
                )
                result = test_utils.single_agent_result(result)["result"]
                test_utils.print_execution_summary("[mcp_test]", result)
            record = await test_utils.call_tool(
                session, "record", {"agents": [_DEFAULT_AGENT]}
            )
            record = test_utils.single_agent_result(record)
            test_utils.print_record("[mcp_test]", record)
            return record


def run_go2_test(
    config: str | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
    task_num: int = _DEFAULT_TASK_NUM,
    remote: bool = False,
    train_ep: int = 0,
    config_overrides: list[str] | None = None,
) -> dict:
    """Run Go2Agent episodes in-process or through MCP."""
    if remote:
        if not config:
            raise ValueError("Remote Go2Agent test requires --config")
        return asyncio.run(_run_remote(config, max_steps, task_num, train_ep, config_overrides))
    return asyncio.run(_run_local(config or _DEFAULT_CONFIG, max_steps, task_num, train_ep, config_overrides))


def test_local_go2_agent() -> None:
    """Smoke test: run a Go2Agent episode in-process."""
    result = run_go2_test(config=_DEFAULT_CONFIG)
    assert isinstance(result, dict)
    assert result.get("ok"), result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis Go2Agent evaluation - local or MCP")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument("--task-num", type=int, default=_DEFAULT_TASK_NUM)
    parser.add_argument("--remote", action="store_true", default=False)
    parser.add_argument("--train_ep", type=int, default=0)
    args, config_overrides = test_utils.parse_args_with_config_overrides(parser)

    result = run_go2_test(
        config=args.config,
        max_steps=args.max_steps,
        task_num=args.task_num,
        remote=args.remote,
        train_ep=args.train_ep,
        config_overrides=config_overrides,
    )
    print("\n[PASS]")
