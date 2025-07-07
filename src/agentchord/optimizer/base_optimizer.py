from typing import Dict

from ..agent_system import BaseAgent


class BaseOptimizer:
    """
    Base class for optimizers.
    """

    def __init__(self, agents: Dict[str, BaseAgent]):
        """
        Initialize the optimizer with the agent system.
        :param agent_system: The agents to be optimized.
        """
        self.agents = agents
        self.trajectories = list()
        self.optimization_history = list()

    def step(self):
        """
        Perform a single optimization step.
        This method should be overridden by subclasses to implement specific optimization logic.
        """
        raise NotImplementedError("The step method must be implemented by subclasses.")