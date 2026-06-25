"""Test a Genesis scene with three Grasp agents.

Local mode:
    python tests/genesis/test_agents.py

Remote mode:
    capcmd server --config configs/genesis/all_agent.yaml
    python tests/genesis/test_agents.py --remote --config configs/genesis/all_agent.yaml
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cap_general.core.utils import test_utils

_DEFAULT_MAX_STEPS = 100
_DEFAULT_TASK_NUM = 9
_DEFAULT_CONFIG = "configs/genesis/all_agent.yaml"
_AGENTS = ("grasp_0", "grasp_1", "grasp_2")


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


def _make_local_scene(config: str):
    import cap_general.frameworks.genesis  # noqa: F401
    from cap_general.core.scene import BaseScene

    return BaseScene.from_yaml(config)


async def _run_local(config: str, max_steps: int, task_num: int) -> dict[str, dict]:
    """Run three Genesis GraspAgent tasks in-process."""
    scene = _make_local_scene(config)
    for agent in _AGENTS:
        scene.reset(agent=agent, options={})
        print(f"[test] {agent} agent_doc {scene.agent_doc(agent=agent)}")

    records: dict[str, dict] = {}
    for task_idx in range(task_num):
        agent = _AGENTS[task_idx % len(_AGENTS)]
        print(f"\n[test] --- Task {task_idx + 1}/{task_num}: agent={agent} ---")
        await scene.execute(agent=agent, code=_make_grasp_code(max_steps))
        status = await scene.monitor(agent=agent, wait_finish=True)
        result = status["result"]
        test_utils.print_execution_summary("[test]", result)

    for agent in _AGENTS:
        record = scene.record(agent=agent, step_idx=-1)
        test_utils.print_record(f"[test][{agent}]", record)
        records[agent] = record
    return records


async def _run_remote(config: str, max_steps: int, task_num: int) -> dict[str, dict]:
    """Run three Genesis GraspAgent tasks through MCP."""
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

            for task_idx in range(task_num):
                agent = _AGENTS[task_idx % len(_AGENTS)]
                print(f"\n[mcp_test] --- Task {task_idx + 1}/{task_num}: agent={agent} ---")
                result = await test_utils.call_tool(
                    session,
                    "execute",
                    {"agent": agent, "code": _make_grasp_code(max_steps)},
                )
                result = await test_utils.call_tool(session, "monitor", {"agent": agent, "wait_finish": True})
                result = result["result"]
                test_utils.print_execution_summary("[mcp_test]", result)

            records: dict[str, dict] = {}
            for agent in _AGENTS:
                record = await test_utils.call_tool(session, "record", {"agent": agent, "step_idx": -1})
                test_utils.print_record(f"[mcp_test][{agent}]", record)
                records[agent] = record
            return records


def run_agents_test(
    config: str | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
    task_num: int = _DEFAULT_TASK_NUM,
    remote: bool = False,
) -> dict[str, dict]:
    """Run three Grasp agents in round-robin order."""
    if remote:
        if not config:
            raise ValueError("Remote agents test requires --config")
        return asyncio.run(_run_remote(config, max_steps, task_num))
    return asyncio.run(_run_local(config or _DEFAULT_CONFIG, max_steps, task_num))


def test_local_agents_scene() -> None:
    """Smoke test: run three Grasp agents in one Genesis scene."""
    result = run_agents_test(config=_DEFAULT_CONFIG)
    assert isinstance(result, dict)
    assert set(result) == set(_AGENTS)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genesis three-Grasp-agent scene evaluation - local or MCP")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument("--task-num", type=int, default=_DEFAULT_TASK_NUM)
    parser.add_argument("--remote", action="store_true", default=False)
    args = parser.parse_args()

    run_agents_test(
        config=args.config,
        max_steps=args.max_steps,
        task_num=args.task_num,
        remote=args.remote,
    )
    print("\n[PASS]")
