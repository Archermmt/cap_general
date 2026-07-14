"""Tests for top-level scene routing."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from cap_general.core.control import BaseControl, BaseControlConfig
from cap_general.core.operator import BaseOperator, to_stage_fn
from cap_general.core.policy import BasePolicy, BasePolicyConfig
from cap_general.core.robot import BaseRobot
from cap_general.core.scene import BaseScene
from cap_general.core.scene.base_scene import AgentInfo, ServerConfig
from cap_general.core.pipeline.job.base_job import BaseJob
from cap_general.core.pipeline.job.train_job import TrainJob, TrainJobConfig
from cap_general.core.utils.config import parse_cli_overrides

_ALPHA_KEY = "a(alpha)"
_BETA_KEY = "b(beta)"


@BaseJob.register()
class SceneDummyTrainJob(TrainJob):
    """Minimal train job for scene routing tests."""

    job_group = "train"
    job_type = "scene_dummy"
    config_cls = TrainJobConfig

    def _execute(self, policy, robot, options=None):
        return policy.to_dict(), {}


@BaseRobot.register()
class SceneDummyRobot(BaseRobot):
    """Small robot for scene routing tests."""

    robot_type = "scene_dummy"
    config_cls = BaseRobot.config_cls

    def _reset(self, options=None):
        return {"options": options or {}}, {}

    def _step(self, action):
        return {"action": action}, 0.0, False, False, {}

    def get_observation(self, folder):
        return {"folder": str(folder)}


@dataclass
class SceneDummyPolicyConfig(BasePolicyConfig):
    """Configuration for SceneDummyPolicy."""


@BaseOperator.register()
class SceneDummyOp(BaseOperator):
    """Minimal operator for scene routing tests."""

    op_group = "test"
    op_type = "scene_dummy"

    @to_stage_fn
    def inference(self, inputs):
        return {"output": None}

    @to_stage_fn
    def update(self, inputs):
        return inputs


@BasePolicy.register()
class SceneDummyPolicy(BasePolicy):
    """Small policy for scene routing tests."""

    name = "Scene Dummy Policy"
    config_cls = SceneDummyPolicyConfig

    policy_type = "scene_dummy"

@BaseControl.register()
class SceneDummyAgent(BaseControl):
    """Small agent for scene routing tests."""

    control_type = "scene_dummy"
    config_cls = BaseControlConfig

    def functions(self):
        return {"echo": self.echo}

    def echo(self, value: str) -> str:
        return value


def _scene_config(*agent_names: str, trace_level: str = "all") -> dict:
    agent_names = agent_names or ("alpha",)
    return {
        "type": "base",
        "server": {"cap_id": "scene_test", "port": 8899},
        "record_dir": "outputs/test_scene",
        "trace_level": trace_level,
        "controls": [
            {
                "name": agent_name,
                "alias": agent_name[0],
                "type": "scene_dummy",
                "robot": {"type": "scene_dummy", "reset_time": 0},
                "policies": {
                    "test": {
                        "type": "scene_dummy",
                        "graph": {
                            "name": "scene_dummy",
                            "nodes": [
                                {"name": "model", "node_type": "test::scene_dummy", "config": {}}
                            ],
                        },
                    }
                },
                "pipeline": {
                    "jobs": [{"type": "train::scene_dummy", "config": {"epoch": 1}}]
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
    assert "echo" in scene.control_doc(["a"])["scene"]["controls"][_ALPHA_KEY]["function_doc"]
    assert "folder" in scene.get_obs(["alpha"])[_ALPHA_KEY]


def test_scene_batch_methods_route_multiple_agents():
    scene = BaseScene.from_config(_scene_config("alpha", "beta", trace_level="never"))

    resets = scene.reset({"a": {"value": 1}, "beta": {"value": 2}})
    docs = scene.control_doc(["alpha", "beta"])
    observations = scene.get_obs(["alpha", "beta"])
    history_update = scene.update_history(
        {
            "alpha": {"role": "user", "tool": "reset", "request": {}},
            "b": {"role": "user", "tool": "control_doc", "request": {}},
        }
    )
    scene.update_history({"alpha": {"role": _ALPHA_KEY, "tool": "reset", "response": {"ok": True}}})
    full_record = scene._get_agent("alpha").record(step_idx=-1)
    for agent_info in scene._agents.values():
        agent_info.agent.record = lambda step_idx, clean_frames=False: {
            "step_idx": step_idx,
            "clean_frames": clean_frames,
        }
    records = scene.record(["alpha", "beta"])

    assert set(resets) == {_ALPHA_KEY, _BETA_KEY}
    assert set(docs["scene"]["controls"]) == {_ALPHA_KEY, _BETA_KEY}
    assert set(observations) == {_ALPHA_KEY, _BETA_KEY}
    assert scene._get_agent("alpha").mark == _ALPHA_KEY
    assert history_update[_ALPHA_KEY] == {"ok": True, "updated": 1}
    # History is persisted by the scene; record() info no longer embeds it.
    assert "executes" in full_record["info"]
    assert "agent" not in scene._history[1]
    assert scene._history[1]["control"] == _BETA_KEY
    assert scene._history[1]["role"] == "user"
    assert scene._history[1]["tool"] == "control_doc"
    assert scene._history[1]["request"] == {}
    assert "timestamp" in scene._history[1]
    history_lines = (scene._record_dir / "history.json").read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 3
    assert records == {
        _ALPHA_KEY: {"step_idx": -1, "clean_frames": False},
        _BETA_KEY: {"step_idx": -1, "clean_frames": False},
    }


def test_scene_agent_info_keeps_single_runtime_state_source():
    scene = BaseScene.from_config(_scene_config("alpha", "beta"))

    alpha_info = scene._agents["alpha"]

    assert isinstance(alpha_info, AgentInfo)
    assert alpha_info.agent is scene._get_agent("alpha")
    assert scene.agents["alpha"] is alpha_info.agent
    assert alpha_info.status["agent"] == "alpha"
    assert alpha_info.status["running"] is False
    assert alpha_info.task is None


def test_scene_copies_prefixed_skill_folders_in_fast_mode(tmp_path: Path):
    scene = BaseScene.from_config(_scene_config())
    skill_root = tmp_path / "skills"
    skill_root.mkdir()
    (skill_root / ".keep").touch()
    (skill_root / "cap").mkdir()
    (skill_root / "cap" / "stale.txt").touch()

    result = scene._copy_skills_for_server(
        ServerConfig(
            cap_id="cap",
            skill_folders={"nanobot": str(tmp_path / "nanobot-skills"), "codex": str(skill_root)},
        ),
        client_type="codex",
    )

    assert result == skill_root
    assert (result / "cap" / "stale.txt").is_file()
    assert not (result / "SKILL.md").exists()
    assert (result / "cap_state" / "SKILL.md").is_file()
    assert (result / "cap_reset" / "SKILL.md").is_file()
    assert (result / "cap_pipeline" / "SKILL.md").is_file()
    assert (result / "cap_execute" / "SKILL.md").is_file()
    content = (result / "cap_state" / "SKILL.md").read_text(encoding="utf-8")
    assert "{cap_id}" not in content
    assert "{available_names}" not in content
    execute_content = (result / "cap_execute" / "SKILL.md").read_text(encoding="utf-8")
    pipeline_content = (result / "cap_pipeline" / "SKILL.md").read_text(encoding="utf-8")
    assert "Control Execute Sync" in execute_content
    assert "image-type tool" in execute_content
    assert '"name": "media"' not in execute_content
    assert "Control Pipeline" in pipeline_content
    assert not (result / "cap_execute" / "SKILL.async").exists()
    assert not (result / "cap_execute" / "SKILL.sync").exists()


def test_scene_copies_bundled_skill_root_when_fast_is_disabled(tmp_path: Path):
    config = _scene_config()
    config["task_async"] = True
    scene = BaseScene.from_config(config)
    skill_root = tmp_path / "skills"
    skill_root.mkdir()
    (skill_root / ".keep").touch()

    result = scene._copy_skills_for_server(
        ServerConfig(cap_id="cap", skill_folders={"nanobot": str(skill_root)}, fast=False)
    )

    assert result == skill_root / "cap"
    assert (result / "SKILL.md").is_file()
    assert (result / "cap_state" / "SKILL.md").is_file()
    execute_content = (result / "cap_execute" / "SKILL.md").read_text(encoding="utf-8")
    assert "Control Execute Async" in execute_content
    assert '"name": "media"' in execute_content
    assert "image-type tool" not in execute_content
    assert "Control Pipeline" in (result / "cap_pipeline" / "SKILL.md").read_text(encoding="utf-8")
    assert not (result / "cap_execute" / "SKILL.async").exists()
    assert not (result / "cap_execute" / "SKILL.sync").exists()


def test_scene_trace_splits_batch_results_into_agent_history_entries():
    scene = BaseScene.from_config(_scene_config("alpha", "beta"))

    observations = scene.get_obs(["alpha", "beta"])
    statuses = asyncio.run(scene.monitor(["alpha", "beta"], wait_ms=0))

    assert set(observations) == {_ALPHA_KEY, _BETA_KEY}
    assert set(statuses) == {_ALPHA_KEY, _BETA_KEY}
    assert [message["tool"] for message in scene._history] == [
        "get_obs",
        "get_obs",
        "get_obs",
        "monitor",
        "monitor",
        "monitor",
    ]
    history_lines = [
        json.loads(line)
        for line in (scene._record_dir / "history.json").read_text(encoding="utf-8").splitlines()
    ]
    assert len(history_lines) == 6
    request_lines = [message for message in history_lines if message["role"] == "user"]
    assert [message["tool"] for message in request_lines] == ["get_obs", "monitor"]
    assert all("agent" not in message for message in request_lines)
    assert sum(message["role"] == _ALPHA_KEY for message in history_lines) == 2
    assert sum(message["role"] == _BETA_KEY for message in history_lines) == 2


def test_scene_execute_routes_to_selected_agent():
    config = _scene_config()
    config["async_task"] = True
    scene = BaseScene.from_config(config)

    async def _run():
        started = await scene.execute({"alpha": 'RESULT = {"value": echo("ok")}'})
        assert started[_ALPHA_KEY]["running"] is True
        status = await scene.monitor(["alpha"])
        return status[_ALPHA_KEY]["result"]

    result = asyncio.run(_run())

    assert result["ok"] is True
    assert result["result"] == {"value": "ok"}


def test_scene_dispatch_respects_async_task():
    async_config = _scene_config()
    async_config["async_task"] = True
    async_scene = BaseScene.from_config(async_config)
    sync_config = _scene_config()
    sync_config["async_task"] = False
    sync_scene = BaseScene.from_config(sync_config)

    async def _run():
        current_thread = threading.get_ident()
        worker_thread = await async_scene._dispatch_task(threading.get_ident, {})
        same_thread = await sync_scene._dispatch_task(threading.get_ident, {})
        return current_thread, worker_thread, same_thread

    current_thread, worker_thread, same_thread = asyncio.run(_run())

    assert worker_thread != current_thread
    assert same_thread == current_thread


def test_scene_execute_reports_running_for_busy_agent():
    config = _scene_config()
    config["async_task"] = True
    scene = BaseScene.from_config(config)

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
    config = _scene_config("alpha", "beta")
    config["async_task"] = True
    scene = BaseScene.from_config(config)

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
    timestamp_format = "%Y-%m-%d:%H-%M-%S.%f"
    alpha_started = datetime.strptime(finished[_ALPHA_KEY]["started_at"], timestamp_format)
    beta_started = datetime.strptime(finished[_BETA_KEY]["started_at"], timestamp_format)
    assert abs((alpha_started - beta_started).total_seconds()) < 0.05


def test_scene_retry_reports_running_for_busy_agent():
    config = _scene_config()
    config["async_task"] = True
    scene = BaseScene.from_config(config)

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
    config = _scene_config(trace_level="never")
    config["async_task"] = True
    scene = BaseScene.from_config(config)

    async def _run():
        await scene.execute({"alpha": 'RESULT = {"value": "disabled"}'})
        await scene.monitor(["alpha"])
        scene.get_obs(["alpha"])
        disabled_history = list(scene._history)

        scene.set_trace_level("task")
        await scene.execute({"alpha": 'RESULT = {"value": "task"}'})
        await scene.monitor(["alpha"])
        scene.get_obs(["alpha"])
        await scene.retry(["alpha"])
        await scene.monitor(["alpha"])
        scene.run_pipe(
            {
                "alpha": {
                    "job_options": [{"job": "train", "options": {"epoch": 2}}],
                }
            }
        )
        task_history = list(scene._history)

        scene.set_trace_level("all")
        await scene.execute({"alpha": 'RESULT = {"value": "enabled"}'})
        await scene.monitor(["alpha"])
        scene.get_obs(["alpha"])
        await scene.retry(["alpha"])
        await scene.monitor(["alpha"])
        scene.run_pipe(
            {
                "alpha": {
                    "job_options": [{"job": "train", "options": {"epoch": 2}}],
                }
            }
        )
        return disabled_history, task_history, list(scene._history)

    disabled_history, task_history, traced_history = asyncio.run(_run())

    assert disabled_history == []
    task_request_messages = [m for m in task_history if "request" in m]
    assert [m["tool"] for m in task_request_messages] == [
        "execute",
        "monitor",
        "get_obs",
        "retry",
        "monitor",
    ]
    assert all(m["role"] == "user" for m in task_request_messages)
    assert task_request_messages[0]["request"] == {
        "control_codes": {"alpha": 'RESULT = {"value": "task"}'}
    }
    assert task_request_messages[2]["request"] == {"controls": ["alpha"]}
    assert task_request_messages[3]["request"] == {"controls": ["alpha"]}
    task_response_messages = [m for m in task_history if "response" in m]
    assert [m["tool"] for m in task_response_messages] == [
        "execute",
        "monitor",
        "get_obs",
        "retry",
        "monitor",
    ]
    assert all(m["role"] == _ALPHA_KEY for m in task_response_messages)
    assert task_response_messages[0]["response"]["running"] is True
    assert task_response_messages[1]["response"]["result"]["result"] == {"value": "task"}

    request_messages = [m for m in traced_history if "request" in m]
    assert [m["tool"] for m in request_messages][-6:] == [
        "execute",
        "monitor",
        "get_obs",
        "retry",
        "monitor",
        "run_pipe",
    ]
    assert all(m["role"] == "user" for m in request_messages)
    assert request_messages[-6]["request"] == {
        "control_codes": {"alpha": 'RESULT = {"value": "enabled"}'}
    }
    assert request_messages[-3]["request"] == {"controls": ["alpha"]}
    assert request_messages[-1]["request"] == {
        "control_options": {
            "alpha": {
                "job_options": [{"job": "train", "options": {"epoch": 2}}],
            }
        }
    }
    response_messages = [m for m in traced_history if "response" in m]
    assert [m["tool"] for m in response_messages][-6:] == [
        "execute",
        "monitor",
        "get_obs",
        "retry",
        "monitor",
        "run_pipe",
    ]
    assert response_messages[0]["role"] == _ALPHA_KEY
    assert all("mark" not in m for m in traced_history)
    assert all("agent" not in m for m in request_messages)
    assert all("agent" not in m for m in response_messages)
    assert all("agent" not in m["response"] for m in response_messages)
    assert all(
        ("method" in m["response"]) == (m["tool"] == "monitor")
        for m in response_messages
    )
    assert response_messages[-6]["response"]["running"] is True
    assert response_messages[-5]["response"]["result"]["result"] == {"value": "enabled"}
    assert response_messages[-1]["response"]["running"] is False
    assert response_messages[-1]["response"]["result"]["ok"] is True


def test_scene_debug_visualizes_policy_graphs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rendered: list[tuple[str, str, str, bool]] = []

    class _DummyDigraph:
        def __init__(self, name, node_attr=None, edge_attr=None):
            self.name = name

        def node(self, *_args, **_kwargs):
            return None

        def edge(self, *_args, **_kwargs):
            return None

        def render(self, filename, directory, format, cleanup):
            rendered.append((self.name, filename, directory, cleanup))
            out = Path(directory) / f"{filename}.{format}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("ok", encoding="utf-8")
            return str(out)

    import sys
    import types

    monkeypatch.setitem(sys.modules, "graphviz", types.SimpleNamespace(Digraph=_DummyDigraph))

    scene = BaseScene.from_config(
        {
            "type": "base",
            "record_dir": str(tmp_path / "scene"),
            "debug": True,
            "server": {"cap_id": "scene_test", "port": 8899},
            "controls": [
                {
                    "name": "alpha",
                    "alias": "a",
                    "type": "scene_dummy",
                    "robot": {"type": "scene_dummy", "reset_time": 0},
                    "policies": {
                        "test": {
                            "type": "scene_dummy",
                            "graph": {
                                "name": "scene_dummy",
                                "nodes": [
                                    {"name": "model", "node_type": "test::scene_dummy", "config": {}}
                                ],
                            },
                        }
                    },
                }
            ],
        }
    )

    assert scene._get_agent("alpha")._config.debug is True
    assert rendered
    assert (tmp_path / "scene" / "alpha" / "visualize").exists()


def test_scene_debug_visualize_falls_back_to_dot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _ExecutableNotFound(Exception):
        pass

    class _DummyDigraph:
        def __init__(self, name, node_attr=None, edge_attr=None):
            self.name = name
            self.source = "digraph G { a -> b }"

        def node(self, *_args, **_kwargs):
            return None

        def edge(self, *_args, **_kwargs):
            return None

        def render(self, filename, directory, format, cleanup):
            raise _ExecutableNotFound("dot missing")

    _ExecutableNotFound.__name__ = "ExecutableNotFound"

    import sys
    import types

    monkeypatch.setitem(sys.modules, "graphviz", types.SimpleNamespace(Digraph=_DummyDigraph))

    scene = BaseScene.from_config(
        {
            "type": "base",
            "record_dir": str(tmp_path / "scene"),
            "debug": True,
            "server": {"cap_id": "scene_test", "port": 8899},
            "controls": [
                {
                    "name": "alpha",
                    "alias": "a",
                    "type": "scene_dummy",
                    "robot": {"type": "scene_dummy", "reset_time": 0},
                    "policies": {
                        "test": {
                            "type": "scene_dummy",
                            "graph": {
                                "name": "scene_dummy",
                                "nodes": [
                                    {"name": "model", "node_type": "test::scene_dummy", "config": {}}
                                ],
                            },
                        }
                    },
                }
            ],
        }
    )

    dot_path = tmp_path / "scene" / "alpha" / "visualize" / "test.dot"
    assert scene._get_agent("alpha")._config.debug is True
    assert dot_path.exists()
    assert dot_path.read_text(encoding="utf-8") == "digraph G { a -> b }"


def test_scene_from_yaml_loads_agents(tmp_path: Path):
    config_path = tmp_path / "scene.yaml"
    config_path.write_text(
        """
type: base
server:
  cap_id: scene_test
  port: 8899
record_dir: outputs/test_scene_yaml
debug: false
controls:
  - name: alpha
    alias: a
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
            "--controls[0].alias=renamed",
            "--controls[0].robot.reset_time",
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
