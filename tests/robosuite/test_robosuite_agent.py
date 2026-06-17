"""Test RobosuiteAgent locally or remotely through MCP.

Local mode:
    python tests/robosuite/test_robosuite_agent.py

Remote mode:
    capcmd server --config configs/robosuite_agent.yaml
    python tests/robosuite/test_robosuite_agent.py --remote

Full usage:
    python tests/robosuite/test_robosuite_agent.py [--remote] [--config PATH] [--privileged]

Nanobot test:
    让七仔把红色方块放到绿色方块上面。
"""

from __future__ import annotations

import asyncio
import os
import platform
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cap_general.core.utils.test_utils import call_tool, print_execution_summary, print_record
from cap_general.frameworks.robosuite import ORACLE_CODE

if platform.system() == "Darwin":
    if os.environ.get("MUJOCO_GL") == "egl":
        os.environ["MUJOCO_GL"] = "cgl"
    else:
        os.environ.setdefault("MUJOCO_GL", "cgl")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_DEFAULT_CONFIG = "configs/robosuite_agent.yaml"
_DEFAULT_MAX_STEPS = 1500
_DEFAULT_TRIAL_NUM = 1


def _run_local(config: str, max_steps: int, trial_num: int) -> dict:
    from cap_general.core.agent import BaseAgent
    from cap_general.frameworks.robosuite import RobosuiteAgent  # noqa: F401

    print(f"[test] Loading RobosuiteAgent from: {config}")
    agent = BaseAgent.from_yaml(config)
    agent.reset(options={})
    print(f"[test] agent_doc {agent.agent_doc()}")
    ok = True
    for trial_idx in range(trial_num):
        print(f"\n[test] --- Trial {trial_idx + 1}/{trial_num} ---")
        if trial_idx == 0:
            result = agent.execute(ORACLE_CODE)
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
            await call_tool(session, "reset", {"options": {}})
            agent_doc = await call_tool(session, "agent_doc")
            print(f"[mcp_test] agent_doc {agent_doc}")
            ok = True
            for trial_idx in range(trial_num):
                print(f"\n[mcp_test] --- Trial {trial_idx + 1}/{trial_num} ---")
                if trial_idx == 0:
                    result = await call_tool(session, "execute", {"code": ORACLE_CODE})
                else:
                    result = await call_tool(session, "retry")
                ok = ok and bool(result.get("ok"))
                print_execution_summary("[mcp_test]", result)
            record = await call_tool(session, "record", {"step_idx": -1})
            print_record("[mcp_test]", record)
            return {"ok": ok, "record": record}


def run_robosuite_test(config: str, max_steps: int, trial_num: int, remote: bool = False) -> dict:
    """Run Robosuite pick-and-place episodes in-process or through MCP."""
    if remote:
        return asyncio.run(_run_remote(config, max_steps, trial_num))
    return _run_local(config, max_steps, trial_num)


def test_local_robosuite(config: str = _DEFAULT_CONFIG) -> None:
    """Smoke test: run oracle pick-and-place in-process and assert no execution error."""
    result = run_robosuite_test(
        config=config,
        max_steps=_DEFAULT_MAX_STEPS,
        trial_num=_DEFAULT_TRIAL_NUM,
    )
    assert result["ok"], "one or more episodes failed"


def test_mcp_robosuite(config: str = _DEFAULT_CONFIG) -> None:
    """Smoke test: run oracle pick-and-place through MCP and assert no execution error."""
    result = run_robosuite_test(
        config=config,
        max_steps=_DEFAULT_MAX_STEPS,
        trial_num=_DEFAULT_TRIAL_NUM,
        remote=True,
    )
    assert result["ok"], "one or more episodes failed"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Robosuite pick-and-place evaluation - local or MCP")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument("--trial-num", type=int, default=_DEFAULT_TRIAL_NUM)
    parser.add_argument("--remote", action="store_true", default=False)
    args = parser.parse_args()

    result = run_robosuite_test(
        config=args.config,
        max_steps=args.max_steps,
        trial_num=args.trial_num,
        remote=args.remote,
    )
    print(f"\n{'[PASS]' if result['ok'] else '[FAIL]'}")
