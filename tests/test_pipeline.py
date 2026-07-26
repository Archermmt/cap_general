"""Tests for pipeline jobs."""

import logging
from types import SimpleNamespace

from cap_general.frameworks.genesis.pipeline.job.genesis_eval_job import (
    GenesisEvalJob,
    GenesisEvalJobConfig,
)


class _EvalPolicy:
    def __init__(self):
        self.training = True
        self.inputs = []

    def eval(self):
        self.training = False
        return self

    def run(self, stage, inputs):
        self.inputs.append((stage, inputs))
        return SimpleNamespace(success=True, output=0.0)


class _EvalRobot:
    def __init__(self):
        self.training = False
        self.step_index = 0
        self.robot = SimpleNamespace(ee_pose=_FloatValue())

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def reset(self):
        self.step_index = 0
        return {"obs": 0}, {}

    @property
    def policy_obs(self):
        return {"policy": self.step_index}

    def step(self, action):
        self.step_index += 1
        return {"obs": self.step_index}, float(self.step_index), self.step_index == 3, False, {}

    def get_stereo_rgb_images(self, normalize=True):
        assert normalize is True
        return _FloatValue()


class _FloatValue:
    def float(self):
        return self


def test_genesis_eval_job_averages_final_episode_rewards():
    policy = _EvalPolicy()
    robot = _EvalRobot()
    job = GenesisEvalJob(
        GenesisEvalJobConfig(epoch=2, stage="rl", max_steps=5),
        logger=logging.getLogger(__name__),
    )

    policy_config, report = job.execute(policy, robot)

    assert policy_config is None
    assert report == {
        "average_reward": 3.0,
        "rewards": [3.0, 3.0],
        "epoch": 2,
        "stage": "rl",
    }
    assert all(
        stage == "inference" and "policy" in inputs["obs"]
        for stage, inputs in policy.inputs
    )
    assert policy.training is False
    assert robot.training is False


def test_genesis_eval_job_uses_bc_policy_inputs():
    policy = _EvalPolicy()
    robot = _EvalRobot()
    job = GenesisEvalJob(
        GenesisEvalJobConfig(epoch=1, stage="bc", max_steps=1),
        logger=logging.getLogger(__name__),
    )

    _, report = job.execute(policy, robot)

    assert report["average_reward"] == 1.0
    _, inputs = policy.inputs[0]
    assert inputs["env"] is robot
    assert isinstance(inputs["rgb_obs"], _FloatValue)
    assert isinstance(inputs["ee_pose"], _FloatValue)
