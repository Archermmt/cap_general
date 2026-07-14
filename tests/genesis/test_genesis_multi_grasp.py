"""Test a Genesis scene with three Grasp controls.

Local mode:
    /Users/archer/anaconda3/envs/simu/bin/python tests/genesis/test_genesis_multi_grasp.py
    /Users/archer/anaconda3/envs/simu/bin/python tests/genesis/test_genesis_multi_grasp.py --parallel

Remote mode:
    capcmd server --config configs/genesis/genesis_multi_grasp.yaml
    /Users/archer/anaconda3/envs/simu/bin/python tests/genesis/test_genesis_multi_grasp.py --remote --config configs/genesis/genesis_multi_grasp.yaml
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
_DEFAULT_CONFIG = "configs/genesis/genesis_multi_grasp.yaml"
_CONTROLS = ("grasp_0", "grasp_1", "grasp_2")


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
    """Run three Genesis GraspControl tasks in-process."""
    scene = _make_local_scene(config, config_overrides)
    scene.reset({control: {} for control in _CONTROLS})
    scene_doc = scene.control_doc(list(_CONTROLS))["scene"]
    async_task = scene_doc.get("async_task", True)
    for mark, control_doc in scene_doc["controls"].items():
        print(f"[test] {mark} control_doc {control_doc}")

    records: dict[str, dict] = {}
    for task_idx in range(task_num):
        task_controls = _CONTROLS if parallel else (_CONTROLS[task_idx % len(_CONTROLS)],)
        print(f"\n[test] --- Task {task_idx + 1}/{task_num}: controls={task_controls} ---")
        statuses = await scene.execute({control: _make_grasp_code(max_steps) for control in task_controls})
        if async_task:
            statuses = await scene.monitor(list(task_controls))
        for control, status in statuses.items():
            test_utils.print_execution_summary(f"[test][{control}]", status["result"])

    all_records = scene.record(list(_CONTROLS))
    for control, record in all_records.items():
        test_utils.print_record(f"[test][{control}]", record)
        records[control] = record
    return records


async def _run_remote(
    config: str,
    max_steps: int,
    task_num: int,
    parallel: bool = False,
    config_overrides: list[str] | None = None,
) -> dict[str, dict]:
    """Run three Genesis GraspControl tasks through MCP."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    from cap_general.core.scene import BaseScene

    url = BaseScene.get_server_url(config, overrides=config_overrides)
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_names = [tool.name for tool in (await session.list_tools()).tools]
            print(f"[mcp_test]({url}) Available tools: {tool_names}")
            await test_utils.call_tool(session, "reset", {"control_options": {control: {} for control in _CONTROLS}})
            scene_doc = (await test_utils.call_tool(session, "control_doc", {"controls": list(_CONTROLS)}))["scene"]
            async_task = scene_doc.get("async_task", True)
            for mark, control_doc in scene_doc["controls"].items():
                print(f"[mcp_test] {mark} control_doc {control_doc}")

            for task_idx in range(task_num):
                task_controls = _CONTROLS if parallel else (_CONTROLS[task_idx % len(_CONTROLS)],)
                print(f"\n[mcp_test] --- Task {task_idx + 1}/{task_num}: controls={task_controls} ---")
                statuses = await test_utils.call_tool(
                    session, "execute",
                    {"control_codes": {control: _make_grasp_code(max_steps) for control in task_controls}},
                )
                if async_task:
                    statuses = await test_utils.call_tool(
                        session, "monitor", {"controls": list(task_controls)},
                    )
                for control, status in statuses.items():
                    test_utils.print_execution_summary(f"[mcp_test][{control}]", status["result"])

            records = await test_utils.call_tool(session, "record", {"controls": list(_CONTROLS)})
            for control, record in records.items():
                test_utils.print_record(f"[mcp_test][{control}]", record)
            return records


def run_multi_controls_test(
    config: str | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
    task_num: int = _DEFAULT_TASK_NUM,
    remote: bool = False,
    parallel: bool = False,
    config_overrides: list[str] | None = None,
) -> dict[str, dict]:
    """Run Grasp controls round-robin, or all together when parallel is enabled."""
    if remote:
        if not config:
            raise ValueError("Remote controls test requires --config")
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


def test_local_multi_controls_scene() -> None:
    """Smoke test: run three Grasp controls in one Genesis scene."""
    result = run_multi_controls_test(config=_DEFAULT_CONFIG)
    assert isinstance(result, dict)
    assert set(result) == set(_CONTROLS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis three-Grasp-control scene evaluation - local or MCP")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=_DEFAULT_MAX_STEPS)
    parser.add_argument("--task-num", type=int, default=_DEFAULT_TASK_NUM)
    parser.add_argument("--remote", action="store_true", default=False)
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Run all controls in every execute batch instead of round-robin execution",
    )
    args, config_overrides = test_utils.parse_args_with_config_overrides(parser)

    run_multi_controls_test(
        config=args.config,
        max_steps=args.max_steps,
        task_num=args.task_num,
        remote=args.remote,
        parallel=args.parallel,
        config_overrides=config_overrides,
    )
    print("\n[PASS]")
