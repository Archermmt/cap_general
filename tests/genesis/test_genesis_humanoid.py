"""Test GenesisHumanoidControl locally or remotely through MCP.

Local task mode:
    python tests/genesis/test_genesis_humanoid.py

Local train mode:
    python tests/genesis/test_genesis_humanoid.py --train_ep 100

Remote task mode:
    capcmd server --config configs/genesis/genesis_humanoid.yaml
    python tests/genesis/test_genesis_humanoid.py --remote

Remote train mode:
    capcmd server --config configs/genesis/genesis_humanoid.yaml
    python tests/genesis/test_genesis_humanoid.py --remote --train_ep 100
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

_DEFAULT_CONFIG = "configs/genesis/genesis_humanoid.yaml"
_DEFAULT_MAX_STEPS = 100
_DEFAULT_TASK_NUM = 4
_DEFAULT_AGENT = "humanoid"
_TASKS = ("stand", "walk", "run", "hurdle")


def _make_code(task: str, max_steps: int) -> str:
    """Build GenesisHumanoidControl execution code with values baked in."""
    return f"""\
RESULT = locomote(task={task!r}, max_steps={max_steps})
"""


def _make_train_eval_request(train_ep: int, max_steps: int) -> dict:
    """Build a lightweight humanoid train-then-eval request."""
    return {
        "job_options": [
            {"job": "train", "options": {"epoch": train_ep, "record_epoch": 50}},
            {"job": "eval", "options": {"epoch": 1, "stage": "rl", "max_steps": max_steps}},
        ],
        "policy_name": "runner",
    }


def _selected_tasks(task_num: int) -> tuple[str, ...]:
    return _TASKS[: max(min(int(task_num), len(_TASKS)), 0)]


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
    """Exercise humanoid scene reset, pipeline, execution, and recording in-process."""
    scene = _make_local_scene(config, config_overrides)
    reset_result = scene.reset({_DEFAULT_AGENT: {}})
    if not test_utils.single_agent_result(reset_result).get("ok", False):
        raise AssertionError(reset_result)

    scene_doc = scene.control_doc([_DEFAULT_AGENT])["scene"]
    async_task = scene_doc.get("async_task", True)
    control_doc = next(iter(scene_doc["controls"].values()))
    if "locomote" not in control_doc["function_doc"]:
        raise AssertionError("Humanoid scene does not expose locomote")
    print(f"[test] control_doc {control_doc}")

    if train_ep > 0:
        print("\n[test] --- Train smoke test ---")
        status = scene.run_pipe({_DEFAULT_AGENT: _make_train_eval_request(train_ep, max_steps)})
        result = test_utils.single_agent_result(status)["result"]
        if not result.get("ok", False):
            raise AssertionError(result.get("error") or result)
        test_utils.print_pipeline_summary("[test]", result)

    for task_idx, task in enumerate(_selected_tasks(task_num)):
        print(f"\n[test] --- Task {task_idx + 1}/{task_num}: {task} ---")
        status = await scene.execute({_DEFAULT_AGENT: _make_code(task, max_steps)})
        if async_task:
            status = await scene.monitor([_DEFAULT_AGENT])
        result = test_utils.single_agent_result(status)["result"]
        if not result.get("ok", False):
            raise AssertionError(result.get("stderr") or result)
        if result.get("result", {}).get("task") != task:
            raise AssertionError(f"Humanoid scene returned the wrong task: {result}")
        test_utils.print_execution_summary("[test]", result)

    record = test_utils.single_agent_result(scene.record([_DEFAULT_AGENT]))
    if "code" not in record or "info" not in record:
        raise AssertionError(f"Humanoid scene record is incomplete: {record}")
    test_utils.print_record("[test]", record)
    return record


async def _run_remote(
    config: str,
    max_steps: int,
    task_num: int,
    train_ep: int,
    config_overrides: list[str] | None = None,
) -> dict:
    """Exercise the same humanoid scene capabilities through MCP."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    from cap_general.core.scene import BaseScene

    url = BaseScene.get_server_url(config, overrides=config_overrides)
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            reset_result = await test_utils.call_tool(
                session, "reset", {"control_options": {_DEFAULT_AGENT: {}}}
            )
            if not test_utils.single_agent_result(reset_result).get("ok", False):
                raise AssertionError(reset_result)

            scene_doc = (
                await test_utils.call_tool(session, "control_doc", {"controls": [_DEFAULT_AGENT]})
            )["scene"]
            async_task = scene_doc.get("async_task", True)
            control_doc = next(iter(scene_doc["controls"].values()))
            if "locomote" not in control_doc["function_doc"]:
                raise AssertionError("Humanoid MCP scene does not expose locomote")

            if train_ep > 0:
                status = await test_utils.call_tool(
                    session,
                    "run_pipe",
                    {"control_options": {_DEFAULT_AGENT: _make_train_eval_request(train_ep, max_steps)}},
                )
                result = test_utils.single_agent_result(status)["result"]
                if not result.get("ok", False):
                    raise AssertionError(result.get("error") or result)
                test_utils.print_pipeline_summary("[mcp_test]", result)

            for task_idx, task in enumerate(_selected_tasks(task_num)):
                print(f"\n[mcp_test] --- Task {task_idx + 1}/{task_num}: {task} ---")
                status = await test_utils.call_tool(
                    session,
                    "execute",
                    {"control_codes": {_DEFAULT_AGENT: _make_code(task, max_steps)}},
                )
                if async_task:
                    status = await test_utils.call_tool(session, "monitor", {"controls": [_DEFAULT_AGENT]})
                result = test_utils.single_agent_result(status)["result"]
                if not result.get("ok", False):
                    raise AssertionError(result.get("stderr") or result)
                if result.get("result", {}).get("task") != task:
                    raise AssertionError(f"Humanoid MCP scene returned the wrong task: {result}")
                test_utils.print_execution_summary("[mcp_test]", result)

            record = await test_utils.call_tool(session, "record", {"controls": [_DEFAULT_AGENT]})
            record = test_utils.single_agent_result(record)
            if "code" not in record or "info" not in record:
                raise AssertionError(f"Humanoid MCP scene record is incomplete: {record}")
            test_utils.print_record("[mcp_test]", record)
            return record


def run_humanoid_test(
    config: str = _DEFAULT_CONFIG,
    max_steps: int = _DEFAULT_MAX_STEPS,
    task_num: int = _DEFAULT_TASK_NUM,
    remote: bool = False,
    train_ep: int = 0,
    config_overrides: list[str] | None = None,
) -> dict:
    """Run humanoid scene capabilities in-process or through MCP."""
    runner = _run_remote if remote else _run_local
    return asyncio.run(runner(config, max_steps, task_num, train_ep, config_overrides))


def test_local_humanoid_scene() -> None:
    """Smoke test the actual Genesis scene rather than isolated helper interfaces."""
    result = run_humanoid_test(
        max_steps=1,
        task_num=len(_TASKS),
        config_overrides=[
            "show_viewer=false",
            "controls[0].robot.camera_enabled=false",
            "controls[0].robot.max_episode_steps=10",
        ],
    )
    assert isinstance(result, dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis humanoid scene evaluation - local or MCP")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument("--task-num", type=int, default=_DEFAULT_TASK_NUM)
    parser.add_argument("--train_ep", type=int, default=0)
    parser.add_argument("--remote", action="store_true", default=False)
    args, config_overrides = test_utils.parse_args_with_config_overrides(parser)

    run_humanoid_test(
        config=args.config,
        max_steps=args.max_steps,
        task_num=args.task_num,
        remote=args.remote,
        train_ep=args.train_ep,
        config_overrides=config_overrides,
    )
    print("\n[PASS]")
