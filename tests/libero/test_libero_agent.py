"""Test LiberoAgent locally or remotely through MCP.

Local mode:
    python tests/libero/test_libero_agent.py

Remote mode:
    capcmd server --config configs/libero_agent.yaml
    python tests/libero/test_libero_agent.py --remote

Full usage:
    python tests/libero/test_libero_agent.py [--remote] [--config PATH]

Nanobot test:
    让七仔把碗放到炉子上，再把炉子拧开，然后把盘子挪到炉子前面，最后把抽屉打开。
"""

from __future__ import annotations

import asyncio
import os
import platform

from cap_general.core.utils.test_utils import call_tool, print_execution_summary, print_record

if platform.system() == "Darwin":
    if os.environ.get("MUJOCO_GL") == "egl":
        os.environ["MUJOCO_GL"] = "cgl"
    else:
        os.environ.setdefault("MUJOCO_GL", "cgl")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_DEFAULT_CONFIG = "configs/libero_agent.yaml"
_DEFAULT_MAX_STEPS = 300
_DEFAULT_TRIAL_NUM = 1

TASKS = [
    "put the bowl on the stove",
    "turn on the stove",
    "push the plate to the front of the stove",
    "open the middle drawer of the cabinet",
]


def _make_code(task: str, max_steps: int) -> str:
    """Build oracle code with values baked in; no exec-globals variables needed."""
    return f"""\
success = libero_vla_episode(task={task!r}, max_steps={max_steps})
print(f"Episode result: {{'SUCCESS' if success else 'FAIL'}}")
RESULT = {{"success": success, "task": {task!r}}}
"""


def _run_local(config: str, max_steps: int, trial_num: int) -> dict:
    from cap_general.core.agent import BaseAgent
    from cap_general.frameworks.libero.agent import LiberoAgent  # noqa: F401

    print(f"[test] Loading LiberoAgent from: {config}")
    agent = BaseAgent.from_yaml(config)
    agent.reset(options={"episode_idx": 0})
    print(f"[test] agent_doc {agent.agent_doc()}")
    ok = True
    for task_idx, current_task in enumerate(TASKS):
        print(f"\n[test] ========== Task {task_idx + 1}/{len(TASKS)}: {current_task!r} ==========")
        for trial_idx in range(trial_num):
            print(f"[test] --- Trial {trial_idx + 1}/{trial_num} ---")
            if trial_idx == 0:
                result = agent.execute(_make_code(current_task, max_steps))
            else:
                result = agent.retry()
            ok = ok and bool(result.get("ok"))
            print_execution_summary("[test]", result)
    record = agent.record(step_idx=-1)
    print_record("[test]", record)
    return {"ok": ok, "record": record}


async def _run_remote(config: str, max_steps: int, trial_num: int) -> dict:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    from cap_general.core.agent import BaseAgent

    url = BaseAgent.get_server_url(config)
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_names = [tool.name for tool in (await session.list_tools()).tools]
            print(f"[mcp_test]({url}) Available tools: {tool_names}")
            await call_tool(session, "reset", {"options": {"episode_idx": 0}})
            agent_doc = await call_tool(session, "agent_doc")
            print(f"[mcp_test] agent_doc {agent_doc}")
            ok = True
            for task_idx, current_task in enumerate(TASKS):
                print(f"\n[mcp_test] ========== Task {task_idx + 1}/{len(TASKS)}: {current_task!r} ==========")
                for trial_idx in range(trial_num):
                    print(f"[mcp_test] --- Trial {trial_idx + 1}/{trial_num} ---")
                    if trial_idx == 0:
                        result = await call_tool(session, "execute", {"code": _make_code(current_task, max_steps)})
                    else:
                        result = await call_tool(session, "retry")
                    ok = ok and bool(result.get("ok"))
                    print_execution_summary("[mcp_test]", result)
            record = await call_tool(session, "record", {"step_idx": -1})
            print_record("[mcp_test]", record)
            return {"ok": ok, "record": record}


def run_libero_test(config, max_steps, trial_num, remote) -> dict:
    """Run LIBERO VLA episodes in-process or through MCP."""
    if remote:
        return asyncio.run(_run_remote(config, max_steps, trial_num))
    return _run_local(config, max_steps, trial_num)


def test_local_libero(config: str = _DEFAULT_CONFIG) -> None:
    """Smoke test: run episodes in-process and assert no execution error."""
    result = run_libero_test(config=config, max_steps=300)
    assert result["ok"], "one or more episodes failed"


def test_mcp_libero(config: str = _DEFAULT_CONFIG) -> None:
    """Smoke test: run episodes through MCP and assert no execution error."""
    result = run_libero_test(config=config, max_steps=300, remote=True)
    assert result["ok"], "one or more episodes failed"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LIBERO VLA evaluation - local or MCP")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument("--trial-num", type=int, default=_DEFAULT_TRIAL_NUM)
    parser.add_argument("--remote", action="store_true", default=False)
    args = parser.parse_args()

    result = run_libero_test(config=args.config, max_steps=args.max_steps, trial_num=args.trial_num, remote=args.remote)
    print(f"\n{'[PASS]' if result['ok'] else '[FAIL]'}")
