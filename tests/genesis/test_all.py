"""Test a Genesis scene with Grasp, Go2, and Drone agents.

Local mode:
    python tests/genesis/test_all.py

Remote mode:
    capcmd server --config configs/genesis/all_agent.yaml
    python tests/genesis/test_all.py --remote --config configs/genesis/all_agent.yaml
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cap_general.core.utils import test_utils

_DEFAULT_MAX_STEPS = 100
_DEFAULT_TASK_NUM = 6
_DEFAULT_CONFIG = "configs/genesis/all_agent.yaml"
_AGENTS = ("grasp", "go", "drone")
_ROUND_NUM = 10


def _target_positions(task_num: int) -> list[list[float]]:
    """Create deterministic drone target positions."""
    base_targets = [
        [0.5, 0.0, 1.0],
        [0.5, 0.3, 1.1],
        [0.0, 0.3, 1.2],
        [-0.4, 0.0, 1.1],
        [0.0, -0.3, 1.0],
    ]
    return [base_targets[idx % len(base_targets)] for idx in range(task_num)]


def _make_grasp_code(max_steps: int) -> str:
    return f"""\
result = grasp_episode(stage="rl", max_steps={max_steps})
RESULT = {{
    "success": True,
    "stage": result.get("stage"),
    "steps": result.get("steps"),
    "mock": result.get("mock", False),
}}
"""


def _make_go_code(max_steps: int, task_idx: int) -> str:
    turn_angle = 2.0 * math.pi * (task_idx % _ROUND_NUM) / _ROUND_NUM
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


def _make_drone_code(max_steps: int, target_pos: list[float]) -> str:
    return f"""\
follow_result = follow_target(target_pos={target_pos!r}, max_steps={max_steps})
hover_result = hover(time_s=3.0)
RESULT = {{
    "success": True,
    "target_pos": follow_result.get("target_pos"),
    "hover_duration": hover_result.get("duration"),
}}
"""


def _task(agent: str, task_idx: int, max_steps: int, drone_targets: list[list[float]]) -> str:
    if agent == "grasp":
        return _make_grasp_code(max_steps)
    if agent == "go":
        return _make_go_code(max_steps, task_idx)
    if agent == "drone":
        return _make_drone_code(max_steps, drone_targets[task_idx])
    raise ValueError(f"Unsupported agent: {agent}")


def _make_local_scene(config: str):
    import cap_general.frameworks.genesis  # noqa: F401
    from cap_general.core.scene import BaseScene

    return BaseScene.from_yaml(config)


def _run_local(config: str, max_steps: int, task_num: int) -> dict[str, dict]:
    """Run round-robin Genesis agent tasks in-process."""
    scene = _make_local_scene(config)
    for agent in _AGENTS:
        scene.reset(agent=agent, options={})
        print(f"[test] {agent} agent_doc {scene.agent_doc(agent=agent)}")

    records: dict[str, dict] = {}
    drone_targets = _target_positions(task_num)
    for task_idx in range(task_num):
        agent = _AGENTS[task_idx % len(_AGENTS)]
        print(f"\n[test] --- Task {task_idx + 1}/{task_num}: agent={agent} ---")
        result = scene.execute(agent=agent, code=_task(agent, task_idx, max_steps, drone_targets))
        test_utils.print_execution_summary("[test]", result)

    for agent in _AGENTS:
        record = scene.record(agent=agent, step_idx=-1)
        test_utils.print_record(f"[test][{agent}]", record)
        records[agent] = record
    return records


async def _run_remote(config: str, max_steps: int, task_num: int) -> dict[str, dict]:
    """Run round-robin Genesis agent tasks through MCP."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    from cap_general.core.scene import BaseScene

    url = BaseScene.get_server_url(config)
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_names = [tool.name for tool in (await session.list_tools()).tools]
            print(f"[mcp_test]({url}) Available tools: {tool_names}")
            for agent in _AGENTS:
                await test_utils.call_tool(session, "reset", {"agent": agent, "options": {}})
                agent_doc = await test_utils.call_tool(session, "agent_doc", {"agent": agent})
                print(f"[mcp_test] {agent} agent_doc {agent_doc}")

            drone_targets = _target_positions(task_num)
            for task_idx in range(task_num):
                agent = _AGENTS[task_idx % len(_AGENTS)]
                print(f"\n[mcp_test] --- Task {task_idx + 1}/{task_num}: agent={agent} ---")
                result = await test_utils.call_tool(
                    session,
                    "execute",
                    {"agent": agent, "code": _task(agent, task_idx, max_steps, drone_targets)},
                )
                test_utils.print_execution_summary("[mcp_test]", result)

            records: dict[str, dict] = {}
            for agent in _AGENTS:
                record = await test_utils.call_tool(session, "record", {"agent": agent, "step_idx": -1})
                test_utils.print_record(f"[mcp_test][{agent}]", record)
                records[agent] = record
            return records


def run_all_agent_test(
    config: str | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
    task_num: int = _DEFAULT_TASK_NUM,
    remote: bool = False,
) -> dict[str, dict]:
    """Run Grasp, Go2, and Drone agents in round-robin order."""
    if remote:
        if not config:
            raise ValueError("Remote all-agent test requires --config")
        return asyncio.run(_run_remote(config, max_steps, task_num))
    return _run_local(config or _DEFAULT_CONFIG, max_steps, task_num)


def test_local_all_agent_scene() -> None:
    """Smoke test: run a multi-agent Genesis scene in-process."""
    result = run_all_agent_test(config=_DEFAULT_CONFIG)
    assert isinstance(result, dict)
    assert set(result) == set(_AGENTS)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genesis multi-agent scene evaluation - local or MCP")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument("--task-num", type=int, default=_DEFAULT_TASK_NUM)
    parser.add_argument("--remote", action="store_true", default=False)
    args = parser.parse_args()

    run_all_agent_test(
        config=args.config,
        max_steps=args.max_steps,
        task_num=args.task_num,
        remote=args.remote,
    )
    print("\n[PASS]")
