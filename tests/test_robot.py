"""Tests for BaseRobot behavior."""

import logging
from typing import Any, SupportsFloat

from cap_general.core.robot import BaseRobot, BaseRobotConfig

LOGGER = logging.getLogger(__name__)


@BaseRobot.register()
class DummyRobot(BaseRobot):
    """Small concrete robot for base interface tests."""

    robot_type = "dummy"
    config_cls = BaseRobot.config_cls

    def _reset(
        self,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.training:
            return {"mode": "train", "options": options or {}}
        return {"step": self.step_cnt}, {"seed": self._config.seed, "options": options or {}}

    def _step(
        self,
        action: Any,
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        if self.training:
            return {"mode": "train"}, 1.0, False, {"action": action}
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
    robot = DummyRobot(config=BaseRobotConfig(seed=123), logger=LOGGER)

    obs, info = robot.reset(options={"difficulty": "easy"})

    assert obs == {"step": 0}
    assert info == {"seed": 123, "options": {"difficulty": "easy"}}


def test_robot_base_step_returns_gymnasium_tuple_and_tracks_step_count():
    robot = DummyRobot(config=BaseRobotConfig(), logger=LOGGER)
    robot.reset()

    obs, reward, terminated, truncated, info = robot.step({"move": 1})

    assert obs == {"step": 1}
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info == {"action": {"move": 1}}
    assert robot.step_cnt == 1


def test_robot_base_step_uses_compute_reward():
    robot = RewardDummyRobot(config=BaseRobotConfig(), logger=LOGGER)
    robot.reset()

    _, reward, _, _, _ = robot.step({"move": 1})

    assert reward == 3.5


def test_robot_train_and_eval_switch_reset_and_step_semantics():
    robot = DummyRobot(config=BaseRobotConfig(), logger=LOGGER)

    assert robot.training is False
    assert robot.train() is robot
    assert robot.training is True
    assert robot.reset(options={"seed": 1}) == {"mode": "train", "options": {"seed": 1}}
    assert robot.step({"move": 1}) == ({"mode": "train"}, 1.0, False, {"action": {"move": 1}})
    assert robot.step_cnt == 0

    assert robot.eval() is robot
    assert robot.training is False
    obs, _, _, _, _ = robot.step({"move": 2})
    assert obs == {"step": 1}
    assert robot.step_cnt == 1


def test_grasp_robot_train_restores_training_episode_length_and_resets():
    from cap_general.frameworks.genesis.robot.genesis_grasp_robot import GenesisGraspRobot, GenesisGraspRobotConfig

    robot = GenesisGraspRobot(config=GenesisGraspRobotConfig(), logger=LOGGER)
    # Simulate a fully-built genesis state without actual Genesis deps
    robot.ctrl_dt = 0.01
    robot._env_cfg = {"episode_length_s": 10_000.0}
    robot.max_episode_length = 1_000_000
    robot._train_episode_length_s = 3.0
    robot._eval_episode_length_s = 10_000.0

    reset_calls = []
    obs_calls = []

    def fake_reset():
        reset_calls.append(1)
        return {"policy": "reset"}

    def fake_get_obs():
        obs_calls.append(1)
        return {"policy": "eval"}

    robot._reset = fake_reset
    robot._get_observations = fake_get_obs

    robot.train()

    assert robot.training is True
    assert robot._env_cfg["episode_length_s"] == 3.0
    assert robot.max_episode_length == 300
    assert len(reset_calls) == 1
    assert robot._last_policy_obs == {"policy": "reset"}

    robot.eval()

    assert robot.training is False
    assert robot._env_cfg["episode_length_s"] == 10_000.0
    assert robot.max_episode_length == 1_000_000
    assert robot.policy_obs == {"policy": "eval"}


def test_robot_base_registry():
    assert BaseRobot.robot_type == "base"
    assert BaseRobot.get_registered_class("dummy") is DummyRobot
