"""Tests for top-level scene routing."""

from __future__ import annotations

from pathlib import Path

from cap_general.core.agent import BaseAgent
from cap_general.core.env import BaseEnv
from cap_general.core.scene import BaseScene


@BaseEnv.register()
class SceneDummyEnv(BaseEnv):
    """Small env for scene routing tests."""

    name = "Scene Dummy Env"

    @classmethod
    def env_type(cls) -> str:
        return "scene_dummy_env"

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
                    "env": {"type": "scene_dummy_env", "reset_time": 0},
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
    assert agent._owns_logger is False
    assert scene.reset(agent="alpha", options={"x": 1})["ok"] is True
    assert "echo" in scene.agent_doc(agent="a")["function_doc"]


def test_scene_execute_routes_to_selected_agent():
    scene = BaseScene.from_config(_scene_config())

    result = scene.execute(agent="alpha", code='RESULT = {"value": echo("ok")}')

    assert result["ok"] is True
    assert result["result"] == {"value": "ok"}


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
      env:
        type: scene_dummy_env
        reset_time: 0
      policies: {}
""",
        encoding="utf-8",
    )

    scene = BaseScene.from_yaml(config_path)

    assert scene.server_config.cap_id == "scene_test"
    assert scene._get_agent("a") is scene._get_agent("alpha")
