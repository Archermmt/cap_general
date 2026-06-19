"""Test DroneAgent locally or remotely through MCP.

Local mode:
    python tests/genesis/test_drone_agent.py

Remote mode:
    capcmd server --config configs/genesis/drone_agent.yaml
    python tests/genesis/test_drone_agent.py --remote --config configs/genesis/drone_agent.yaml
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cap_general.core.utils import test_utils

_DEFAULT_MAX_STEPS = 300
_DEFAULT_CONFIG = "configs/genesis/drone_agent.yaml"
_DEFAULT_TASK_NUM = 1


def _make_code(max_steps: int) -> str:
    """Build DroneAgent execution code with values baked in."""
    return f"""\
hover_result = hover(max_steps={max_steps})
RESULT = {{
    "success": True,
    "steps": hover_result.get("steps"),
    "mock": hover_result.get("mock", False),
}}
"""


def _make_local_agent(config: str):
    import cap_general.frameworks.genesis  # noqa: F401
    from cap_general.core.agent import BaseAgent

    return BaseAgent.from_yaml(config)


def _run_local(config: str, max_steps: int, task_num: int) -> dict:
    """Run DroneAgent hover tasks in-process."""
    agent = _make_local_agent(config)
    agent.reset(options={})
    print(f"[test] agent_doc {agent.agent_doc()}")
    for task_idx in range(task_num):
        print(f"\n[test] --- Task {task_idx + 1}/{task_num} ---")
        result = agent.execute(_make_code(max_steps))
        test_utils.print_execution_summary("[test]", result)
    record = agent.record(step_idx=-1)
    test_utils.print_record("[test]", record)
    return record


async def _run_remote(config: str, max_steps: int, task_num: int) -> dict:
    """Run DroneAgent hover tasks through MCP."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    from cap_general.core.agent import BaseAgent

    url = BaseAgent.get_server_url(config)
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_names = [tool.name for tool in (await session.list_tools()).tools]
            print(f"[mcp_test]({url}) Available tools: {tool_names}")
            await test_utils.call_tool(session, "reset", {"options": {}})
            agent_doc = await test_utils.call_tool(session, "agent_doc")
            print(f"[mcp_test] agent_doc {agent_doc}")
            for task_idx in range(task_num):
                print(f"\n[mcp_test] --- Task {task_idx + 1}/{task_num} ---")
                result = await test_utils.call_tool(session, "execute", {"code": _make_code(max_steps)})
                test_utils.print_execution_summary("[mcp_test]", result)
            record = await test_utils.call_tool(session, "record", {"step_idx": -1})
            test_utils.print_record("[mcp_test]", record)
            return record


def run_drone_test(
    config: str | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
    task_num: int = _DEFAULT_TASK_NUM,
    remote: bool = False,
) -> dict:
    """Run DroneAgent hover tasks in-process or through MCP."""
    if remote:
        if not config:
            raise ValueError("Remote DroneAgent test requires --config")
        return asyncio.run(_run_remote(config, max_steps, task_num))
    return _run_local(config or _DEFAULT_CONFIG, max_steps, task_num)


def test_local_drone_agent() -> None:
    """Smoke test: run a DroneAgent hover task in-process."""
    result = run_drone_test(config=_DEFAULT_CONFIG)
    assert isinstance(result, dict)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genesis DroneAgent evaluation - local or MCP")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument("--task-num", type=int, default=_DEFAULT_TASK_NUM)
    parser.add_argument("--remote", action="store_true", default=False)
    args = parser.parse_args()

    result = run_drone_test(
        config=args.config,
        max_steps=args.max_steps,
        task_num=args.task_num,
        remote=args.remote,
    )
    print("\n[PASS]")
