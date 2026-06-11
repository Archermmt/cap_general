"""Tests for Gymnasium-style BaseEnv behavior."""

from typing import Any, SupportsFloat

from cap_general.core.env import BaseEnv, BaseEnvConfig


@BaseEnv.register()
class DummyEnv(BaseEnv):
    """Small concrete environment for base interface tests."""

    name = "Dummy Env"

    @classmethod
    def env_type(cls) -> str:
        return "dummy_env"

    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.get_observation(), {"seed": seed, "options": options or {}}

    def _step(
        self,
        action: Any,
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        return self.get_observation(), 0.0, False, False, {"action": action}

    def get_observation(self) -> dict[str, Any]:
        return {"step": self.step_cnt}


def test_env_base_reset_returns_gymnasium_tuple():
    env = DummyEnv(config=BaseEnvConfig())

    obs, info = env.reset(seed=123, options={"difficulty": "easy"})

    assert obs == {"step": 0}
    assert info == {"seed": 123, "options": {"difficulty": "easy"}}


def test_env_base_step_returns_gymnasium_tuple_and_tracks_step_count():
    env = DummyEnv(config=BaseEnvConfig())
    env.reset()

    obs, reward, terminated, truncated, info = env.step({"move": 1})

    assert obs == {"step": 1}
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info == {"action": {"move": 1}}
    assert env.step_cnt == 1


def test_env_base_registry():
    assert BaseEnv.env_type() == "base_env"
    assert BaseEnv.get_registered_class("dummy_env") is DummyEnv
