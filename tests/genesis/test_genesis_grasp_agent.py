"""Test GraspAgent locally or remotely through MCP.

Local task mode:
    python tests/genesis/test_genesis_grasp_agent.py

Local train mode:
    python tests/genesis/test_genesis_grasp_agent.py --train_ep 100

Remote task mode:
    capcmd server --config configs/genesis/genesis_grasp_agent.yaml
    python tests/genesis/test_genesis_grasp_agent.py --remote --config configs/genesis/genesis_grasp_agent.yaml

Remote train mode:
    capcmd server --config configs/genesis/genesis_grasp_agent.yaml
    python tests/genesis/test_genesis_grasp_agent.py --remote --config configs/genesis/genesis_grasp_agent.yaml --train_ep 100
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
_DEFAULT_TASK_NUM = 4
_DEFAULT_CONFIG = "configs/genesis/genesis_grasp_agent.yaml"
_DEFAULT_AGENT = "grasp"


def _make_code(max_steps: int) -> str:
    """Build GraspAgent execution code with values baked in."""
    return f"""\
RESULT = grasp_episode(stage="rl", max_steps={max_steps})
release_grasp()
"""


def _make_train_eval_request(train_ep: int, max_steps: int) -> dict:
    """Build a lightweight GraspAgent train-then-eval request for smoke tests."""
    return {
        "job_options": [
            {"job": "train", "options": {"epoch": train_ep, "record_epoch": 50}},
            {"job": "eval", "options": {"epoch": 10, "stage": "rl", "max_steps": max_steps}},
        ],
        "policy_name": "runner",
    }


def test_grasp_config_defaults_to_rl_stage() -> None:
    """Default grasp stage should be RL to avoid BC-only camera dependencies."""
    from cap_general.frameworks.genesis.agent.genesis_grasp_agent import GenesisGraspAgentConfig

    assert GenesisGraspAgentConfig().stage == "rl"


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
    """Run GraspAgent task episodes or training in-process."""
    scene = _make_local_scene(config, config_overrides)
    scene.reset({_DEFAULT_AGENT: {}})
    scene_doc = scene.agent_doc([_DEFAULT_AGENT])["scene"]
    async_task = scene_doc.get("async_task", True)
    print(f"[test] agent_doc {next(iter(scene_doc['agents'].values()))}")

    if train_ep > 0:
        print("\n[test] --- Train smoke test ---")
        status = await scene.run_pipe({_DEFAULT_AGENT: _make_train_eval_request(train_ep, max_steps)})
        if async_task:
            status = await scene.monitor([_DEFAULT_AGENT])
        result = test_utils.single_agent_result(status)["result"]
        if not result.get("ok", False):
            raise AssertionError(result.get("error") or result)
        test_utils.print_pipeline_summary("[test]", result)
        record = test_utils.single_agent_result(scene.record([_DEFAULT_AGENT]))
        test_utils.print_record("[test]", record)

    for task_idx in range(task_num):
        print(f"\n[test] --- Task {task_idx + 1}/{task_num} ---")
        status = await scene.execute({_DEFAULT_AGENT: _make_code(max_steps)})
        if async_task:
            status = await scene.monitor([_DEFAULT_AGENT])
        result = test_utils.single_agent_result(status)["result"]
        if not result.get("ok", False):
            raise AssertionError(result.get("stderr") or result)
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
    """Run GraspAgent task episodes or training through MCP."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    from cap_general.core.scene import BaseScene

    url = BaseScene.get_server_url(config, overrides=config_overrides)
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await test_utils.call_tool(session, "reset", {"agent_options": {_DEFAULT_AGENT: {}}})
            scene_doc = (await test_utils.call_tool(session, "agent_doc", {"agents": [_DEFAULT_AGENT]}))["scene"]
            async_task = scene_doc.get("async_task", True)

            if train_ep > 0:
                print("\n[mcp_test] --- Train smoke test ---")
                status = await test_utils.call_tool(
                    session,
                    "run_pipe",
                    {"agent_options": {_DEFAULT_AGENT: _make_train_eval_request(train_ep, max_steps)}},
                )
                if async_task:
                    status = await test_utils.call_tool(session, "monitor", {"agents": [_DEFAULT_AGENT]})
                result = test_utils.single_agent_result(status)["result"]
                if not result.get("ok", False):
                    raise AssertionError(result.get("error") or result)
                test_utils.print_pipeline_summary("[mcp_test]", result)
                record = await test_utils.call_tool(session, "record", {"agents": [_DEFAULT_AGENT]})
                record = test_utils.single_agent_result(record)
                test_utils.print_record("[mcp_test]", record)
                return record

            for task_idx in range(task_num):
                print(f"\n[mcp_test] --- Task {task_idx + 1}/{task_num} ---")
                status = await test_utils.call_tool(
                    session,
                    "execute",
                    {"agent_codes": {_DEFAULT_AGENT: _make_code(max_steps)}},
                )
                if async_task:
                    status = await test_utils.call_tool(session, "monitor", {"agents": [_DEFAULT_AGENT]})
                result = test_utils.single_agent_result(status)["result"]
                if not result.get("ok", False):
                    raise AssertionError(result.get("stderr") or result)
                test_utils.print_execution_summary("[mcp_test]", result)
            record = await test_utils.call_tool(session, "record", {"agents": [_DEFAULT_AGENT]})
            record = test_utils.single_agent_result(record)
            test_utils.print_record("[mcp_test]", record)
            return record


def run_grasp_test(
    config: str | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
    task_num: int = _DEFAULT_TASK_NUM,
    remote: bool = False,
    train_ep: int = 0,
    config_overrides: list[str] | None = None,
) -> dict:
    """Run GraspAgent task episodes or training in-process or through MCP."""
    if remote:
        if not config:
            raise ValueError("Remote GraspAgent test requires --config")
        return asyncio.run(_run_remote(config, max_steps, task_num, train_ep, config_overrides))
    return asyncio.run(_run_local(config or _DEFAULT_CONFIG, max_steps, task_num, train_ep, config_overrides))


def test_local_grasp_agent() -> None:
    """Smoke test: run a GraspAgent episode in-process."""
    result = run_grasp_test(config=_DEFAULT_CONFIG)
    assert isinstance(result, dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis GraspAgent evaluation - local or MCP")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument("--task-num", type=int, default=_DEFAULT_TASK_NUM)
    parser.add_argument("--remote", action="store_true", default=False)
    parser.add_argument("--train_ep", type=int, default=0)
    args, config_overrides = test_utils.parse_args_with_config_overrides(parser)

    result = run_grasp_test(
        config=args.config,
        max_steps=args.max_steps,
        task_num=args.task_num,
        remote=args.remote,
        train_ep=args.train_ep,
        config_overrides=config_overrides,
    )
    print("\n[PASS]")
