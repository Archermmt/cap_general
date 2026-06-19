"""Genesis task agents."""

from cap_general.frameworks.genesis.agent.franka_agent import FrankaAgent, FrankaAgentConfig
from cap_general.frameworks.genesis.agent.go2_agent import Go2Agent, Go2AgentConfig
from cap_general.frameworks.genesis.agent.grasp_agent import GraspAgent, GraspAgentConfig

__all__ = [
    "FrankaAgent",
    "FrankaAgentConfig",
    "Go2Agent",
    "Go2AgentConfig",
    "GraspAgent",
    "GraspAgentConfig",
]
