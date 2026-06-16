"""Test FrankaAgent locally or remotely through MCP.

Local mode:
    python tests/genesis/test_franka_api.py

Remote mode:
    capcmd server --config configs/franka_agent.yaml
    python tests/genesis/test_franka_api.py --remote --config configs/franka_agent.yaml
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cap_general.core.utils.test_utils import call_tool, print_execution_summary, print_record

_DEFAULT_MAX_STEPS = 2
_DEFAULT_TRIAL_NUM = 1


def _make_code(max_steps: int) -> str:
    """Build FrankaAgent execution code with values baked in."""
    return f"""\
result = franka_episode(max_steps={max_steps})
objects = result.get("obs", {{}}).get("objects", [])
print(f"Franka episode steps={{result.get('steps')}} objects={{len(objects)}}")
RESULT = {{
    "success": True,
    "steps": result.get("steps"),
    "object_count": len(objects),
    "object_types": [obj.get("type") for obj in objects],
}}
"""


def _make_local_agent():
    from cap_general.frameworks.genesis.agent import FrankaAgent, FrankaAgentConfig

    return FrankaAgent(FrankaAgentConfig(record_dir="outputs/genesis_test"))


def _run_local(max_steps: int, trial_num: int) -> dict:
    """Run FrankaAgent episodes in-process."""
    agent = _make_local_agent()
    agent.reset(options={})
    print(f"[test] agent_doc {agent.agent_doc()}")
    ok = True
    last_result = None
    for trial_idx in range(trial_num):
        print(f"\n[test] --- Trial {trial_idx + 1}/{trial_num} ---")
        if trial_idx == 0:
            result = agent.execute(_make_code(max_steps))
        else:
            result = agent.retry()
        last_result = result
        ok = ok and bool(result.get("ok"))
        ok = ok and result.get("result", {}).get("object_count") == 5
        print_execution_summary("[test]", result)
    record = agent.record(step_idx=-1)
    print_record("[test]", record)
    return {"ok": ok, "last_result": last_result, "record": record}


async def _run_remote(config: str, max_steps: int, trial_num: int) -> dict:
    """Run FrankaAgent episodes through MCP."""
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
            last_result = None
            for trial_idx in range(trial_num):
                print(f"\n[mcp_test] --- Trial {trial_idx + 1}/{trial_num} ---")
                if trial_idx == 0:
                    result = await call_tool(session, "execute", {"code": _make_code(max_steps)})
                else:
                    result = await call_tool(session, "retry")
                last_result = result
                ok = ok and bool(result.get("ok"))
                ok = ok and result.get("result", {}).get("object_count") == 5
                print_execution_summary("[mcp_test]", result)
            record = await call_tool(session, "record", {"step_idx": -1})
            print_record("[mcp_test]", record)
            return {"ok": ok, "last_result": last_result, "record": record}


def run_franka_test(
    config: str | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
    trial_num: int = _DEFAULT_TRIAL_NUM,
    remote: bool = False,
) -> dict:
    """Run FrankaAgent episodes in-process or through MCP."""
    if remote:
        if not config:
            raise ValueError("Remote FrankaAgent test requires --config")
        return asyncio.run(_run_remote(config, max_steps, trial_num))
    return _run_local(max_steps, trial_num)


def test_local_franka_agent() -> None:
    """Smoke test: run a FrankaAgent episode in-process."""
    result = run_franka_test()
    assert result["ok"], "FrankaAgent local execution failed"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genesis FrankaAgent evaluation - local or MCP")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument("--trial-num", type=int, default=_DEFAULT_TRIAL_NUM)
    parser.add_argument("--remote", action="store_true", default=False)
    args = parser.parse_args()

    result = run_franka_test(
        config=args.config,
        max_steps=args.max_steps,
        trial_num=args.trial_num,
        remote=args.remote,
    )
    print(f"\n{'[PASS]' if result['ok'] else '[FAIL]'}")
