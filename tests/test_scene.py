"""Tests for top-level scene routing."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from cap_general.core.agent import BaseAgent
from cap_general.core.policy import BasePolicy, BasePolicyConfig
from cap_general.core.robot import BaseRobot
from cap_general.core.scene import BaseScene
from cap_general.core.scene.base_scene import AgentInfo
from cap_general.core.utils.config import parse_cli_overrides

_ALPHA_KEY = "a(alpha)"
_BETA_KEY = "b(beta)"


@BaseRobot.register()
class SceneDummyRobot(BaseRobot):
    """Small robot for scene routing tests."""

    name = "Scene Dummy Robot"

    @classmethod
    def robot_type(cls) -> str:
        return "scene_dummy"

    def _reset(self, options=None):
        return {"options": options or {}}, {}

    def _step(self, action):
        return {"action": action}, 0.0, False, False, {}

    def get_observation(self, folder):
        return {"folder": str(folder)}


@dataclass
class SceneDummyPolicyConfig(BasePolicyConfig):
    """Configuration for SceneDummyPolicy."""


@BasePolicy.register()
class SceneDummyPolicy(BasePolicy):
    """Small policy for scene routing tests."""

    name = "Scene Dummy Policy"
    config_cls = SceneDummyPolicyConfig

    @classmethod
    def policy_type(cls) -> str:
        return "scene_dummy"

    def inference(self, *args, **kwargs):
        return None

    def update(self, **kwargs):
        return kwargs


@BaseAgent.register()
class SceneDummyAgent(BaseAgent):
    """Small agent for scene routing tests."""

    name = "Scene Dummy Agent"

    @classmethod
    def agent_type(cls) -> str:
        return "scene_dummy"

    def functions(self):
        return {"echo": self.echo}

    def echo(self, value: str) -> str:
        return value

    def _train(self, policy, epoch, method, options):
        return (
            {
                "policy_name": policy.name,
                "epoch": epoch,
                "method": method,
                "options": options,
            },
            {},
        )


def _scene_config(*agent_names: str, trace_level: str = "all") -> dict:
    agent_names = agent_names or ("alpha",)
    return {
        "type": "base",
        "server": {"cap_id": "scene_test", "port": 8899},
        "record_dir": "outputs/test_scene",
        "agents": [
            {
                "name": agent_name,
                "alias": [agent_name[0]],
                "config": {
                    "type": "scene_dummy",
                    "robot": {"type": "scene_dummy", "reset_time": 0},
                    "policies": {"test": {"type": "scene_dummy"}},
                    "trace_level": trace_level,
                },
            }
            for agent_name in agent_names
        ],
    }


def test_scene_routes_agent_methods_by_name_and_alias():
    scene = BaseScene.from_config(_scene_config())
    agent = scene._get_agent("alpha")

    assert agent._record_dir == Path("outputs/test_scene/alpha").resolve()
    assert agent._logger is scene._logger
    assert scene.reset({"alpha": {"x": 1}})[_ALPHA_KEY]["ok"] is True
    assert "echo" in scene.agent_doc(["a"])[_ALPHA_KEY]["function_doc"]
    assert "folder" in scene.get_obs(["alpha"])[_ALPHA_KEY]


def test_scene_batch_methods_route_multiple_agents():
    scene = BaseScene.from_config(_scene_config("alpha", "beta"))

    resets = scene.reset({"a": {"value": 1}, "beta": {"value": 2}})
    docs = scene.agent_doc(["alpha", "beta"])
    observations = scene.get_obs(["alpha", "beta"])
    history_update = scene.update_history(
        {
            "alpha": {"role": "user", "tool": "reset", "request": {}},
            "b": {"role": "user", "tool": "agent_doc", "request": {}},
        }
    )
    scene.update_history({"alpha": {"role": _ALPHA_KEY, "tool": "reset", "response": {"ok": True}}})
    full_record = scene._get_agent("alpha").record(step_idx=-1)
    for agent_info in scene._agents.values():
        agent_info.agent.record = lambda step_idx: {"step_idx": step_idx}
    records = scene.record(["alpha", "beta"])

    assert set(resets) == {_ALPHA_KEY, _BETA_KEY}
    assert set(docs) == {_ALPHA_KEY, _BETA_KEY}
    assert set(observations) == {_ALPHA_KEY, _BETA_KEY}
    assert scene._get_agent("alpha").mark == _ALPHA_KEY
    assert history_update[_ALPHA_KEY] == {"ok": True, "updated": 1}
    # history is persisted to history.jsonl; record() info no longer embeds it
    assert "executes" in full_record["info"]
    assert scene._get_agent("beta")._history[0]["role"] == "user"
    assert scene._get_agent("beta")._history[0]["tool"] == "agent_doc"
    assert scene._get_agent("beta")._history[0]["request"] == {}
    assert "timestamp" in scene._get_agent("beta")._history[0]
    assert records == {_ALPHA_KEY: {"step_idx": -1}, _BETA_KEY: {"step_idx": -1}}


def test_scene_agent_info_keeps_single_runtime_state_source():
    scene = BaseScene.from_config(_scene_config("alpha", "beta"))

    alpha_info = scene._agents["alpha"]

    assert isinstance(alpha_info, AgentInfo)
    assert alpha_info.agent is scene._get_agent("alpha")
    assert scene.agents["alpha"] is alpha_info.agent
    assert alpha_info.status["agent"] == "alpha"
    assert alpha_info.status["running"] is False
    assert alpha_info.task is None


def test_scene_execute_routes_to_selected_agent():
    scene = BaseScene.from_config(_scene_config())

    async def _run():
        started = await scene.execute({"alpha": 'RESULT = {"value": echo("ok")}'})
        assert started[_ALPHA_KEY]["running"] is True
        status = await scene.monitor(["alpha"])
        return status[_ALPHA_KEY]["result"]

    result = asyncio.run(_run())

    assert result["ok"] is True
    assert result["result"] == {"value": "ok"}


def test_scene_execute_reports_running_for_busy_agent():
    scene = BaseScene.from_config(_scene_config())

    async def _run():
        first = await scene.execute({"alpha": 'import time\ntime.sleep(0.2)\nRESULT = {"value": "first"}'})
        immediate = await scene.monitor(["alpha"], wait_ms=0)
        second = await scene.execute({"alpha": 'RESULT = {"value": "second"}'})
        wait_started = time.perf_counter()
        delayed = await scene.monitor(["alpha"], wait_ms=20)
        wait_duration = time.perf_counter() - wait_started
        final = await scene.monitor(["alpha"])
        return first, immediate, second, delayed, wait_duration, final

    first, immediate, second, delayed, wait_duration, final = asyncio.run(_run())

    assert first[_ALPHA_KEY]["running"] is True
    assert immediate[_ALPHA_KEY]["running"] is True
    assert second[_ALPHA_KEY]["running"] is True
    assert delayed[_ALPHA_KEY]["running"] is True
    assert wait_duration >= 0.018
    assert final[_ALPHA_KEY]["running"] is False
    assert final[_ALPHA_KEY]["result"]["result"] == {"value": "first"}


def test_scene_execute_starts_multiple_agents_together():
    scene = BaseScene.from_config(_scene_config("alpha", "beta"))

    async def _run():
        started = await scene.execute(
            {
                "alpha": 'import time\ntime.sleep(0.1)\nRESULT = {"value": "alpha"}',
                "beta": 'import time\ntime.sleep(0.1)\nRESULT = {"value": "beta"}',
            }
        )
        finished = await scene.monitor(["alpha", "beta"])
        return started, finished

    started, finished = asyncio.run(_run())

    assert set(started) == {_ALPHA_KEY, _BETA_KEY}
    assert all(status["running"] for status in started.values())
    assert finished[_ALPHA_KEY]["result"]["result"] == {"value": "alpha"}
    assert finished[_BETA_KEY]["result"]["result"] == {"value": "beta"}
    assert abs(finished[_ALPHA_KEY]["started_at"] - finished[_BETA_KEY]["started_at"]) < 0.05


def test_scene_retry_reports_running_for_busy_agent():
    scene = BaseScene.from_config(_scene_config())

    async def _run():
        await scene.execute({"alpha": 'RESULT = {"value": "first"}'})
        await scene.monitor(["alpha"])
        retry_started = await scene.retry(["alpha"])
        busy = await scene.retry(["alpha"])
        final = await scene.monitor(["alpha"])
        return retry_started, busy, final

    retry_started, busy, final = asyncio.run(_run())

    assert retry_started[_ALPHA_KEY]["running"] is True
    assert busy[_ALPHA_KEY]["running"] is True
    assert final[_ALPHA_KEY]["running"] is False
    assert final[_ALPHA_KEY]["method"] == "retry"
    assert final[_ALPHA_KEY]["result"]["result"] == {"value": "first"}


def test_scene_auto_trace_records_task_results_only_when_enabled():
    scene = BaseScene.from_config(_scene_config(trace_level="never"))
    agent = scene._get_agent("alpha")

    async def _run():
        await scene.execute({"alpha": 'RESULT = {"value": "disabled"}'})
        await scene.monitor(["alpha"])
        disabled_history = list(agent._history)

        scene.set_trace_level("task")
        await scene.execute({"alpha": 'RESULT = {"value": "task"}'})
        await scene.monitor(["alpha"])
        await scene.retry(["alpha"])
        await scene.monitor(["alpha"])
        await scene.train(
            {
                "alpha": {
                    "policy_name": "test",
                    "epoch": 2,
                    "method": "train",
                    "options": {},
                }
            }
        )
        await scene.monitor(["alpha"])
        task_history = list(agent._history)

        scene.set_trace_level("all")
        await scene.execute({"alpha": 'RESULT = {"value": "enabled"}'})
        await scene.monitor(["alpha"])
        await scene.retry(["alpha"])
        await scene.monitor(["alpha"])
        await scene.train(
            {
                "alpha": {
                    "policy_name": "test",
                    "epoch": 2,
                    "method": "train",
                    "options": {},
                }
            }
        )
        await scene.monitor(["alpha"])
        return disabled_history, task_history, list(agent._history)

    disabled_history, task_history, traced_history = asyncio.run(_run())

    assert disabled_history == []
    task_request_messages = [m for m in task_history if "request" in m]
    assert [m["tool"] for m in task_request_messages] == ["execute", "retry"]
    assert all(m["role"] == "user" for m in task_request_messages)
    assert task_request_messages[0]["request"] == {'code': 'RESULT = {"value": "task"}'}
    assert task_request_messages[1]["request"] == {}
    task_response_messages = [m for m in task_history if "response" in m]
    assert [m["tool"] for m in task_response_messages] == ["execute", "retry"]
    assert all(m["role"] == _ALPHA_KEY for m in task_response_messages)
    assert task_response_messages[0]["response"]["result"] == {"value": "task"}

    request_messages = [m for m in traced_history if "request" in m]
    assert [m["tool"] for m in request_messages] == ["execute", "retry", "execute", "retry", "train"]
    assert all(m["role"] == "user" for m in request_messages)
    assert request_messages[2]["request"] == {'code': 'RESULT = {"value": "enabled"}'}
    assert request_messages[3]["request"] == {}
    assert request_messages[4]["request"] == {
        "policy_name": "test",
        "epoch": 2,
        "method": "train",
        "options": {},
    }
    response_messages = [m for m in traced_history if "response" in m]
    assert [m["tool"] for m in response_messages] == ["execute", "retry", "execute", "retry", "train"]
    assert response_messages[0]["role"] == _ALPHA_KEY
    assert all("mark" not in m for m in traced_history)
    assert response_messages[2]["response"]["result"] == {"value": "enabled"}
    assert response_messages[4]["response"]["result"]["policy_name"] == "test"


def test_scene_from_yaml_loads_agents(tmp_path: Path):
    config_path = tmp_path / "scene.yaml"
    config_path.write_text(
        """
type: base
server:
  cap_id: scene_test
  port: 8899
record_dir: outputs/test_scene_yaml
agents:
  - name: alpha
    alias:
      - a
    config:
      type: scene_dummy
      robot:
        type: scene_dummy
        reset_time: 0
      policies: {}
""",
        encoding="utf-8",
    )

    overrides = parse_cli_overrides(
        [
            "--server.port",
            "9001",
            "--agents[0].alias=[renamed]",
            "--agents[0].config.robot.reset_time",
            "0.01",
        ]
    )
    scene = BaseScene.from_yaml(config_path, overrides=overrides)

    assert scene.server_config.cap_id == "scene_test"
    assert scene.server_config.port == 9001
    assert scene._get_agent("renamed") is scene._get_agent("alpha")
    assert scene._get_agent("alpha")._robot._config.reset_time == 0.01
    assert BaseScene.get_server_url(config_path, overrides=overrides) == "http://127.0.0.1:9001/mcp"

    with pytest.raises(Exception, match="unknown"):
        BaseScene.from_yaml(config_path, overrides=["server.unknown=1"])


def test_parse_cli_overrides_rejects_missing_values():
    assert parse_cli_overrides(["--server.port=9001", "--server.host", "0.0.0.0"]) == [
        "server.port=9001",
        "server.host=0.0.0.0",
    ]
    with pytest.raises(ValueError, match="Missing value"):
        parse_cli_overrides(["--server.port"])
