"""Test Go2Agent locally or remotely through MCP.

Local mode:
    python tests/genesis/test_go2_agent.py

Remote mode:
    capcmd server --config configs/genesis/go2_agent.yaml
    python tests/genesis/test_go2_agent.py --remote --config configs/genesis/go2_agent.yaml
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
_DEFAULT_CONFIG = "configs/genesis/go2_agent.yaml"
_DEFAULT_TASK_NUM = 5
_DEFAULT_AGENT = "go2"
ROUND_NUM = 10


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


def _make_local_scene(config: str):
    import cap_general.frameworks.genesis  # noqa: F401
    from cap_general.core.scene import BaseScene

    return BaseScene.from_yaml(config)


def _run_local(config: str, max_steps: int, task_num: int) -> dict:
    """Run Go2Agent episodes in-process."""
    scene = _make_local_scene(config)
    scene.reset(agent=_DEFAULT_AGENT, options={})
    print(f"[test] agent_doc {scene.agent_doc(agent=_DEFAULT_AGENT)}")
    turn_angles = _random_turn_angles(task_num)
    for task_idx, turn_angle in enumerate(turn_angles):
        print(f"\n[test] --- Task {task_idx + 1}/{task_num}: turn_angle={turn_angle:.3f} ---")
        result = scene.execute(agent=_DEFAULT_AGENT, code=_make_code(max_steps, turn_angle=turn_angle))
        test_utils.print_execution_summary("[test]", result)
    record = scene.record(agent=_DEFAULT_AGENT, step_idx=-1)
    test_utils.print_record("[test]", record)
    return record


async def _run_remote(config: str, max_steps: int, task_num: int) -> dict:
    """Run Go2Agent episodes through MCP."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    from cap_general.core.scene import BaseScene

    url = BaseScene.get_server_url(config)
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_names = [tool.name for tool in (await session.list_tools()).tools]
            print(f"[mcp_test]({url}) Available tools: {tool_names}")
            await test_utils.call_tool(session, "reset", {"agent": _DEFAULT_AGENT, "options": {}})
            agent_doc = await test_utils.call_tool(session, "agent_doc", {"agent": _DEFAULT_AGENT})
            print(f"[mcp_test] agent_doc {agent_doc}")
            turn_angles = _random_turn_angles(task_num)
            for task_idx, turn_angle in enumerate(turn_angles):
                print(f"\n[mcp_test] --- Task {task_idx + 1}/{task_num}: turn_angle={turn_angle:.3f} ---")
                result = await test_utils.call_tool(
                    session,
                    "execute",
                    {"agent": _DEFAULT_AGENT, "code": _make_code(max_steps, turn_angle=turn_angle)},
                )
                test_utils.print_execution_summary("[mcp_test]", result)
            record = await test_utils.call_tool(session, "record", {"agent": _DEFAULT_AGENT, "step_idx": -1})
            test_utils.print_record("[mcp_test]", record)
            return record


def run_go2_test(
    config: str | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
    task_num: int = _DEFAULT_TASK_NUM,
    remote: bool = False,
) -> dict:
    """Run Go2Agent episodes in-process or through MCP."""
    if remote:
        if not config:
            raise ValueError("Remote Go2Agent test requires --config")
        return asyncio.run(_run_remote(config, max_steps, task_num))
    return _run_local(config or _DEFAULT_CONFIG, max_steps, task_num)


def test_local_go2_agent() -> None:
    """Smoke test: run a Go2Agent episode in-process."""
    result = run_go2_test(config=_DEFAULT_CONFIG)
    assert isinstance(result, dict)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genesis Go2Agent evaluation - local or MCP")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument("--task-num", type=int, default=_DEFAULT_TASK_NUM)
    parser.add_argument("--trial-num", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--remote", action="store_true", default=False)
    args = parser.parse_args()
    task_num = args.trial_num if args.trial_num is not None else args.task_num

    result = run_go2_test(
        config=args.config,
        max_steps=args.max_steps,
        task_num=task_num,
        remote=args.remote,
    )
    print("\n[PASS]")
