"""Test RobosuiteAgent locally or remotely through MCP.

Local mode:
    python tests/robosuite/test_robosuite_agent.py

Remote mode:
    capcmd server --config configs/robosuite/robosuite_agent.yaml
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

from cap_general.core.utils import test_utils

ORACLE_CODE = """
import numpy as np

_, _, green_ext = get_object_pose("green cube", return_bbox_extent=True)
_, _, red_ext = get_object_pose("red cube", return_bbox_extent=True)

# Sample a grasp pose for the red cube and pick it up
pick_pos, pick_quat = sample_grasp_pose("red cube")
goto_pose(pick_pos, pick_quat, z_approach=0.1)
close_gripper()
# Lift the red cube after grasping
post_pick_pos = pick_pos.copy()
post_pick_pos[2] += 0.2
goto_pose(post_pick_pos, pick_quat)

# Compute placement pose on top of the green cube
green_pos, _, _ = get_object_pose("green cube", return_bbox_extent=False)

place_pos = green_pos.copy()
place_pos[2] = green_pos[2] + green_ext[2]/2 + red_ext[2]/2
# Use down orientation for placement
place_quat = np.array([0.0, 0.0, 1.0, 0.0])

# Approach and place the red cube on the green cube
goto_pose(place_pos, pick_quat, z_approach=0.1)
open_gripper()

# Retract after placing
post_place_pos = place_pos.copy()
post_place_pos[2] += 0.1
goto_pose(post_place_pos, place_quat)
RESULT = {"success": True}
"""

if platform.system() == "Darwin":
    if os.environ.get("MUJOCO_GL") == "egl":
        os.environ["MUJOCO_GL"] = "cgl"
    else:
        os.environ.setdefault("MUJOCO_GL", "cgl")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_DEFAULT_CONFIG = "configs/robosuite/robosuite_agent.yaml"
_DEFAULT_MAX_STEPS = 1500
_DEFAULT_TRIAL_NUM = 1
_DEFAULT_AGENT = "robosuite"


def _run_local(config: str, max_steps: int, trial_num: int) -> dict:
    from cap_general.frameworks.robosuite import RobosuiteAgent  # noqa: F401
    from cap_general.core.scene import BaseScene

    print(f"[test] Loading RobosuiteAgent from: {config}")
    scene = BaseScene.from_yaml(config)
    scene.reset(agent=_DEFAULT_AGENT, options={})
    print(f"[test] agent_doc {scene.agent_doc(agent=_DEFAULT_AGENT)}")
    for trial_idx in range(trial_num):
        print(f"\n[test] --- Trial {trial_idx + 1}/{trial_num} ---")
        if trial_idx == 0:
            result = scene.execute(agent=_DEFAULT_AGENT, code=ORACLE_CODE)
        else:
            result = scene.retry(agent=_DEFAULT_AGENT)
        test_utils.print_execution_summary("[test]", result)
    record = scene.record(agent=_DEFAULT_AGENT, step_idx=-1)
    test_utils.print_record("[test]", record)
    return record


async def _run_remote(config: str, max_steps: int, trial_num: int) -> dict:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    from cap_general.core.scene import BaseScene

    url = BaseScene.get_server_url(config)
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_names = [tool.name for tool in (await session.list_tools()).tools]
            print(f"[mcp_test]({url}) Available tools: {tool_names}")
            await test_utils.call_tool(session, "reset", {"agent": _DEFAULT_AGENT, "options": {}})
            agent_doc = await test_utils.call_tool(session, "agent_doc", {"agent": _DEFAULT_AGENT})
            print(f"[mcp_test] agent_doc {agent_doc}")
            for trial_idx in range(trial_num):
                print(f"\n[mcp_test] --- Trial {trial_idx + 1}/{trial_num} ---")
                if trial_idx == 0:
                    result = await test_utils.call_tool(
                        session,
                        "execute",
                        {"agent": _DEFAULT_AGENT, "code": ORACLE_CODE},
                    )
                else:
                    result = await test_utils.call_tool(session, "retry", {"agent": _DEFAULT_AGENT})
                test_utils.print_execution_summary("[mcp_test]", result)
            record = await test_utils.call_tool(session, "record", {"agent": _DEFAULT_AGENT, "step_idx": -1})
            test_utils.print_record("[mcp_test]", record)
            return record


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
    assert isinstance(result, dict)


def test_mcp_robosuite(config: str = _DEFAULT_CONFIG) -> None:
    """Smoke test: run oracle pick-and-place through MCP and assert no execution error."""
    result = run_robosuite_test(
        config=config,
        max_steps=_DEFAULT_MAX_STEPS,
        trial_num=_DEFAULT_TRIAL_NUM,
        remote=True,
    )
    assert isinstance(result, dict)


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
    print("\n[PASS]")
