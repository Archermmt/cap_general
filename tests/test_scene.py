"""Tests for top-level scene routing."""

from __future__ import annotations

import asyncio
from pathlib import Path

from cap_general.core.agent import BaseAgent
from cap_general.core.robot import BaseRobot
from cap_general.core.scene import BaseScene


@BaseRobot.register()
class SceneDummyRobot(BaseRobot):
    """Small robot for scene routing tests."""

    name = "Scene Dummy Robot"

    @classmethod
    def robot_type(cls) -> str:
        return "scene_dummy_robot"

    def _reset(self, options=None):
        return {"options": options or {}}, {}

    def _step(self, action):
        return {"action": action}, 0.0, False, False, {}

    def get_observation(self, folder):
        return {"folder": str(folder)}


@BaseAgent.register()
class SceneDummyAgent(BaseAgent):
    """Small agent for scene routing tests."""

    name = "Scene Dummy Agent"

    @classmethod
    def agent_type(cls) -> str:
        return "scene_dummy_agent"

    def functions(self):
        return {"echo": self.echo}

    def echo(self, value: str) -> str:
        return value


def _scene_config(agent_name: str = "alpha") -> dict:
    return {
        "type": "scene",
        "server": {"cap_id": "scene_test", "port": 8899},
        "record_dir": "outputs/test_scene",
        "agents": [
            {
                "name": agent_name,
                "alias": ["a"],
                "config": {
                    "type": "scene_dummy_agent",
                    "robot": {"type": "scene_dummy_robot", "reset_time": 0},
                    "policies": {},
                },
            }
        ],
    }


def test_scene_routes_agent_methods_by_name_and_alias():
    scene = BaseScene.from_config(_scene_config())
    agent = scene._get_agent("alpha")

    assert agent._record_dir == Path("outputs/test_scene/alpha").resolve()
    assert agent._logger is scene._logger
    assert scene.reset(agent="alpha", options={"x": 1})["ok"] is True
    assert "echo" in scene.agent_doc(agent="a")["function_doc"]


def test_scene_execute_routes_to_selected_agent():
    scene = BaseScene.from_config(_scene_config())

    async def _run():
        started = await scene.execute(agent="alpha", code='RESULT = {"value": echo("ok")}')
        assert started["running"] is True
        status = await scene.monitor(agent="alpha", wait_finish=True)
        return status["result"]

    result = asyncio.run(_run())

    assert result["ok"] is True
    assert result["result"] == {"value": "ok"}


def test_scene_execute_reports_running_for_busy_agent():
    scene = BaseScene.from_config(_scene_config())

    async def _run():
        first = await scene.execute(
            agent="alpha",
            code='import time\ntime.sleep(0.1)\nRESULT = {"value": "first"}',
        )
        second = await scene.execute(agent="alpha", code='RESULT = {"value": "second"}')
        final = await scene.monitor(agent="alpha", wait_finish=True)
        return first, second, final

    first, second, final = asyncio.run(_run())

    assert first["running"] is True
    assert second["running"] is True
    assert final["running"] is False
    assert final["result"]["result"] == {"value": "first"}


def test_scene_retry_reports_running_for_busy_agent():
    scene = BaseScene.from_config(_scene_config())

    async def _run():
        await scene.execute(agent="alpha", code='RESULT = {"value": "first"}')
        await scene.monitor(agent="alpha", wait_finish=True)
        retry_started = await scene.retry(agent="alpha")
        busy = await scene.retry(agent="alpha")
        final = await scene.monitor(agent="alpha", wait_finish=True)
        return retry_started, busy, final

    retry_started, busy, final = asyncio.run(_run())

    assert retry_started["running"] is True
    assert busy["running"] is True
    assert final["running"] is False
    assert final["method"] == "retry"
    assert final["result"]["result"] == {"value": "first"}


def test_scene_from_yaml_loads_agents(tmp_path: Path):
    config_path = tmp_path / "scene.yaml"
    config_path.write_text(
        """
type: scene
server:
  cap_id: scene_test
  port: 8899
record_dir: outputs/test_scene_yaml
agents:
  - name: alpha
    alias:
      - a
    config:
      type: scene_dummy_agent
      robot:
        type: scene_dummy_robot
        reset_time: 0
      policies: {}
""",
        encoding="utf-8",
    )

    scene = BaseScene.from_yaml(config_path)

    assert scene.server_config.cap_id == "scene_test"
    assert scene._get_agent("a") is scene._get_agent("alpha")
