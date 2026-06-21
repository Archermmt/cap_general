"""Tests for BaseRobot behavior."""

from typing import Any, SupportsFloat

from cap_general.core.robot import BaseRobot, BaseRobotConfig


@BaseRobot.register()
class DummyRobot(BaseRobot):
    """Small concrete robot for base interface tests."""

    name = "Dummy Robot"

    @classmethod
    def robot_type(cls) -> str:
        return "dummy_robot"

    def _reset(
        self,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"step": self.step_cnt}, {"seed": self._seed, "options": options or {}}

    def _step(
        self,
        action: Any,
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        return {"step": self.step_cnt}, 0.0, False, False, {"action": action}

    def get_observation(self, folder) -> dict[str, Any]:
        return {"step": self.step_cnt}


class RewardDummyRobot(DummyRobot):
    """Dummy robot whose step reward placeholder differs from computed reward."""

    def _step(
        self,
        action: Any,
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        return {"step": self.step_cnt}, 999.0, False, False, {"action": action}

    def compute_reward(self) -> SupportsFloat:
        return 3.5


def test_robot_base_reset_returns_gymnasium_tuple():
    robot = DummyRobot(config=BaseRobotConfig(seed=123))

    obs, info = robot.reset(options={"difficulty": "easy"})

    assert obs == {"step": 0}
    assert info == {"seed": 123, "options": {"difficulty": "easy"}}


def test_robot_base_step_returns_gymnasium_tuple_and_tracks_step_count():
    robot = DummyRobot(config=BaseRobotConfig())
    robot.reset()

    obs, reward, terminated, truncated, info = robot.step({"move": 1})

    assert obs == {"step": 1}
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info == {"action": {"move": 1}}
    assert robot.step_cnt == 1


def test_robot_base_step_uses_compute_reward():
    robot = RewardDummyRobot(config=BaseRobotConfig())
    robot.reset()

    _, reward, _, _, _ = robot.step({"move": 1})

    assert reward == 3.5


def test_robot_base_registry():
    assert BaseRobot.robot_type() == "base_robot"
    assert BaseRobot.get_registered_class("dummy_robot") is DummyRobot
