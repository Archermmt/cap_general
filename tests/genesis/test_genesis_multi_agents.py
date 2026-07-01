"""Test a Genesis scene with three Grasp agents.

Local mode:
    /Users/archer/anaconda3/envs/simu/bin/python tests/genesis/test_genesis_multi_agents.py
    /Users/archer/anaconda3/envs/simu/bin/python tests/genesis/test_genesis_multi_agents.py --parallel

Remote mode:
    capcmd server --config configs/genesis/genesis_multi_agents.yaml
    /Users/archer/anaconda3/envs/simu/bin/python tests/genesis/test_genesis_multi_agents.py --remote --config configs/genesis/genesis_multi_agents.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cap_general.core.utils import test_utils

_DEFAULT_MAX_STEPS = 100
_DEFAULT_TASK_NUM = 9
_DEFAULT_CONFIG = "configs/genesis/genesis_multi_agents.yaml"
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


def _make_local_scene(config: str, config_overrides: list[str] | None = None):
    import cap_general.frameworks.genesis  # noqa: F401
    from cap_general.core.scene import BaseScene

    return BaseScene.from_yaml(config, overrides=config_overrides)


async def _run_local(
    config: str,
    max_steps: int,
    task_num: int,
    parallel: bool = False,
    config_overrides: list[str] | None = None,
) -> dict[str, dict]:
    """Run three Genesis GraspAgent tasks in-process."""
    scene = _make_local_scene(config, config_overrides)
    scene.reset({agent: {} for agent in _AGENTS})
    agent_docs = scene.agent_doc(list(_AGENTS))
    for response_key, agent_doc in agent_docs.items():
        print(f"[test] {response_key} agent_doc {agent_doc}")

    records: dict[str, dict] = {}
    for task_idx in range(task_num):
        task_agents = _AGENTS if parallel else (_AGENTS[task_idx % len(_AGENTS)],)
        print(f"\n[test] --- Task {task_idx + 1}/{task_num}: agents={task_agents} ---")
        await scene.execute({agent: _make_grasp_code(max_steps) for agent in task_agents})
        statuses = await scene.monitor(list(task_agents))
        for agent, status in statuses.items():
            test_utils.print_execution_summary(f"[test][{agent}]", status["result"])

    all_records = scene.record(list(_AGENTS))
    for agent, record in all_records.items():
        test_utils.print_record(f"[test][{agent}]", record)
        records[agent] = record
    return records


async def _run_remote(
    config: str,
    max_steps: int,
    task_num: int,
    parallel: bool = False,
    config_overrides: list[str] | None = None,
) -> dict[str, dict]:
    """Run three Genesis GraspAgent tasks through MCP."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    from cap_general.core.scene import BaseScene

    url = BaseScene.get_server_url(config, overrides=config_overrides)
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_names = [tool.name for tool in (await session.list_tools()).tools]
            print(f"[mcp_test]({url}) Available tools: {tool_names}")
            await test_utils.call_tool(session, "reset", {"agent_options": {agent: {} for agent in _AGENTS}})
            agent_docs = await test_utils.call_tool(session, "agent_doc", {"agents": list(_AGENTS)})
            for response_key, agent_doc in agent_docs.items():
                print(f"[mcp_test] {response_key} agent_doc {agent_doc}")

            for task_idx in range(task_num):
                task_agents = _AGENTS if parallel else (_AGENTS[task_idx % len(_AGENTS)],)
                print(f"\n[mcp_test] --- Task {task_idx + 1}/{task_num}: agents={task_agents} ---")
                await test_utils.call_tool(
                    session,
                    "execute",
                    {"agent_codes": {agent: _make_grasp_code(max_steps) for agent in task_agents}},
                )
                statuses = await test_utils.call_tool(
                    session,
                    "monitor",
                    {"agents": list(task_agents)},
                )
                for agent, status in statuses.items():
                    test_utils.print_execution_summary(f"[mcp_test][{agent}]", status["result"])

            records = await test_utils.call_tool(session, "record", {"agents": list(_AGENTS)})
            for agent, record in records.items():
                test_utils.print_record(f"[mcp_test][{agent}]", record)
            return records


def run_multi_agents_test(
    config: str | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
    task_num: int = _DEFAULT_TASK_NUM,
    remote: bool = False,
    parallel: bool = False,
    config_overrides: list[str] | None = None,
) -> dict[str, dict]:
    """Run Grasp agents round-robin, or all together when parallel is enabled."""
    if remote:
        if not config:
            raise ValueError("Remote agents test requires --config")
        return asyncio.run(
            _run_remote(config, max_steps, task_num, parallel=parallel, config_overrides=config_overrides)
        )
    return asyncio.run(
        _run_local(
            config or _DEFAULT_CONFIG,
            max_steps,
            task_num,
            parallel=parallel,
            config_overrides=config_overrides,
        )
    )


def test_local_multi_agents_scene() -> None:
    """Smoke test: run three Grasp agents in one Genesis scene."""
    result = run_multi_agents_test(config=_DEFAULT_CONFIG)
    assert isinstance(result, dict)
    assert set(result) == set(_AGENTS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis three-Grasp-agent scene evaluation - local or MCP")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument("--task-num", type=int, default=_DEFAULT_TASK_NUM)
    parser.add_argument("--remote", action="store_true", default=False)
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Run all agents in every execute batch instead of round-robin execution",
    )
    args, config_overrides = test_utils.parse_args_with_config_overrides(parser)

    run_multi_agents_test(
        config=args.config,
        max_steps=args.max_steps,
        task_num=args.task_num,
        remote=args.remote,
        parallel=args.parallel,
        config_overrides=config_overrides,
    )
    print("\n[PASS]")
