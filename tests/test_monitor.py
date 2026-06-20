"""Tests for monitor aggregation helpers."""

from pathlib import Path

import numpy as np

from cap_general.core.env import BaseEnv, BaseEnvConfig
from cap_general.core.monitor import BaseMonitor, BaseMonitorConfig, get_monitor_manager


class DummyMonitorEnv:
    """Small env-like object for monitor tests."""

    def __init__(self, image_name: str):
        self.image_name = image_name

    def get_observation(self, folder: str | Path) -> dict:
        image_path = Path(folder) / self.image_name
        return {
            "images": {"camera": image_path},
            "main_image": image_path,
            "state": self.image_name,
        }


class ImageMonitor(BaseMonitor):
    """Monitor with one raw image observation."""

    def _get_monitor_obs(self) -> dict:
        return {"monitor_camera": np.zeros((4, 4, 3), dtype=np.uint8)}


@BaseEnv.register()
class MonitorManagedEnv(BaseEnv):
    """Small env for monitor config tests."""

    name = "Monitor Managed Env"

    @classmethod
    def env_type(cls) -> str:
        return "monitor_managed_env"

    def _reset(self, options=None):
        return {}, {}

    def _step(self, action):
        return {}, 0.0, False, False, {}


def test_base_monitor_get_obs_aggregates_env_images(tmp_path):
    monitor = BaseMonitor(config=BaseMonitorConfig(name="test"))
    monitor.bind_env(DummyMonitorEnv("a.png"))
    monitor.bind_env(DummyMonitorEnv("b.png"))

    obs = monitor.get_observation(tmp_path)

    assert obs["monitor"] == "test"
    assert set(obs["envs"]) == {"env_0", "env_1"}
    assert obs["images"]["env_0.camera"] == tmp_path / "env_0" / "a.png"
    assert obs["images"]["env_1.camera"] == tmp_path / "env_1" / "b.png"
    assert obs["main_image"] == tmp_path / "env_0" / "a.png"


def test_base_monitor_bind_env_keeps_unique_env_list():
    monitor = BaseMonitor(config=BaseMonitorConfig(name="test"))
    env = DummyMonitorEnv("image.png")

    monitor.bind_env(env)
    monitor.bind_env(env)

    assert monitor.envs == [env]


def test_base_monitor_get_monitor_obs_interface_saves_images(tmp_path):
    monitor = ImageMonitor(config=BaseMonitorConfig(name="image_monitor"))

    obs = monitor.get_observation(tmp_path)

    assert obs["images"]["monitor_camera"] == str(tmp_path / "monitor_camera.png")
    assert obs["main_image"] == str(tmp_path / "monitor_camera.png")


def test_monitor_manager_singleton_creates_monitor(tmp_path):
    manager = get_monitor_manager()
    manager.reset()

    monitor = manager.create_monitor("global")
    monitor.bind_env(DummyMonitorEnv("image.png"))
    obs = manager.get_observation(tmp_path, monitor_name="global")

    assert obs["monitor"] == "global"
    assert obs["main_image"] == tmp_path / "env_0" / "image.png"


def test_monitor_registered_base_from_config():
    monitor = BaseMonitor.from_config({"type": "base_monitor", "name": "configured"})

    assert isinstance(monitor, BaseMonitor)
    assert monitor._name == "configured"


def test_monitor_register_decorator_for_subclasses():
    @BaseMonitor.register()
    class CustomMonitor(BaseMonitor):
        name = "Custom Monitor"

        @classmethod
        def monitor_type(cls) -> str:
            return "custom_monitor"

    assert BaseMonitor.get_registered_class("custom_monitor") is CustomMonitor


def test_env_monitor_config_disabled_by_default():
    manager = get_monitor_manager()
    manager.reset()

    env = MonitorManagedEnv(config=BaseEnvConfig())

    assert env._monitor is None
    assert manager.monitors == {}


def test_env_monitor_config_disabled_when_enable_false():
    manager = get_monitor_manager()
    manager.reset()

    env = MonitorManagedEnv(
        config=BaseEnvConfig(
            monitor={
                "enable": False,
                "name": "disabled_monitor",
                "config": {"type": "base_monitor"},
            }
        )
    )

    assert env._monitor is None
    assert manager.get_monitor("disabled_monitor") is None


def test_env_monitor_config_enabled_creates_monitor_and_registers_env():
    manager = get_monitor_manager()
    manager.reset()

    env = MonitorManagedEnv(
        config=BaseEnvConfig(
            monitor={
                "enable": True,
                "name": "env_monitor",
                "config": {"type": "base_monitor"},
            }
        )
    )

    assert env._monitor is manager.get_monitor("env_monitor")
    assert env._monitor._name == "env_monitor"
    assert env._monitor.envs == [env]


def test_env_monitor_config_from_registered_env_config():
    manager = get_monitor_manager()
    manager.reset()

    env = BaseEnv.from_config(
        {
            "type": "monitor_managed_env",
            "monitor": {
                "enable": True,
                "name": "registered_env_monitor",
                "config": {"type": "base_monitor"},
            },
        }
    )

    assert env._monitor is manager.get_monitor("registered_env_monitor")
    assert env._monitor.envs == [env]
