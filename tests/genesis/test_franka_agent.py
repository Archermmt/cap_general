"""Test FrankaAgent locally or remotely through MCP.

Local mode:
    python tests/genesis/test_franka_agent.py

Remote mode:
    capcmd server --config configs/genesis/franka_agent.yaml
    python tests/genesis/test_franka_agent.py --remote --config configs/genesis/franka_agent.yaml
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

_DEFAULT_MAX_STEPS = 2
_DEFAULT_TRIAL_NUM = 1
_DEFAULT_CONFIG = "configs/genesis/franka_agent.yaml"
_DEFAULT_AGENT = "franka"


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


def _make_local_scene(config: str, config_overrides: list[str] | None = None):
    import cap_general.frameworks.genesis  # noqa: F401
    from cap_general.core.scene import BaseScene

    return BaseScene.from_yaml(config, overrides=config_overrides)


async def _run_local(
    config: str, max_steps: int, trial_num: int, config_overrides: list[str] | None = None
) -> dict:
    """Run FrankaAgent episodes in-process."""
    scene = _make_local_scene(config, config_overrides)
    scene.reset({_DEFAULT_AGENT: {}})
    print(f"[test] agent_doc {test_utils.single_agent_result(scene.agent_doc([_DEFAULT_AGENT]))}")
    for trial_idx in range(trial_num):
        print(f"\n[test] --- Trial {trial_idx + 1}/{trial_num} ---")
        if trial_idx == 0:
            await scene.execute({_DEFAULT_AGENT: _make_code(max_steps)})
            status = await scene.monitor([_DEFAULT_AGENT])
            result = test_utils.single_agent_result(status)["result"]
        else:
            await scene.retry([_DEFAULT_AGENT])
            status = await scene.monitor([_DEFAULT_AGENT])
            result = test_utils.single_agent_result(status)["result"]
        test_utils.print_execution_summary("[test]", result)
    record = test_utils.single_agent_result(scene.record([_DEFAULT_AGENT]))
    test_utils.print_record("[test]", record)
    return record


async def _run_remote(
    config: str, max_steps: int, trial_num: int, config_overrides: list[str] | None = None
) -> dict:
    """Run FrankaAgent episodes through MCP."""
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
            agent_doc = await test_utils.call_tool(session, "agent_doc", {"agents": [_DEFAULT_AGENT]})
            agent_doc = test_utils.single_agent_result(agent_doc)
            print(f"[mcp_test] agent_doc {agent_doc}")
            for trial_idx in range(trial_num):
                print(f"\n[mcp_test] --- Trial {trial_idx + 1}/{trial_num} ---")
                if trial_idx == 0:
                    result = await test_utils.call_tool(
                        session,
                        "execute",
                        {"agent_codes": {_DEFAULT_AGENT: _make_code(max_steps)}},
                    )
                    result = await test_utils.call_tool(
                        session, "monitor", {"agents": [_DEFAULT_AGENT]}
                    )
                    result = test_utils.single_agent_result(result)["result"]
                else:
                    await test_utils.call_tool(session, "retry", {"agents": [_DEFAULT_AGENT]})
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


def run_franka_test(
    config: str | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
    trial_num: int = _DEFAULT_TRIAL_NUM,
    remote: bool = False,
    config_overrides: list[str] | None = None,
) -> dict:
    """Run FrankaAgent episodes in-process or through MCP."""
    if remote:
        if not config:
            raise ValueError("Remote FrankaAgent test requires --config")
        return asyncio.run(_run_remote(config, max_steps, trial_num, config_overrides))
    return asyncio.run(_run_local(config or _DEFAULT_CONFIG, max_steps, trial_num, config_overrides))


def test_local_franka_agent() -> None:
    """Smoke test: run a FrankaAgent episode in-process."""
    result = run_franka_test(config=_DEFAULT_CONFIG)
    assert isinstance(result, dict)
    assert result.get("ok"), f"FrankaAgent test failed: {result}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis FrankaAgent evaluation - local or MCP")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument("--trial-num", type=int, default=_DEFAULT_TRIAL_NUM)
    parser.add_argument("--remote", action="store_true", default=False)
    args, config_overrides = test_utils.parse_args_with_config_overrides(parser)

    result = run_franka_test(
        config=args.config,
        max_steps=args.max_steps,
        trial_num=args.trial_num,
        remote=args.remote,
        config_overrides=config_overrides,
    )
    print("\n[PASS]")
