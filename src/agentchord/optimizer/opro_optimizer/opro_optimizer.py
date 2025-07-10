from typing import Dict

from ...agent_system import BaseAgent, Input
from ...model import ModelConfig, ModelFactory
from ..base_optimizer import BaseOptimizer
from .meta_prompt import META_PROMPT_INPUT, META_PROMPT_INSTRUCTION


class OPROOptimizer(BaseOptimizer):
    """
    Optimizer for OPRO (Optimization by PROmpting).
    This optimizer is designed to work with the OPRO framework, which focuses on optimizing agent interactions.
    """

    def __init__(self, agents: Dict[str, BaseAgent], model_config: ModelConfig):
        """
        Initialize the OPRO optimizer with the agent system.
        :param agents: The agents to be optimized.
        """
        super().__init__(agents)
        self.model_config = model_config
        self.model = ModelFactory(self.model_config).create_model()
        self.optimization_info = Input().get_optimization_info()

    def step(self):
        self.optimization_info = Input().get_optimization_info()
        for trajectory in self.optimization_info:
            trajectory_str = " -> ".join([f"{output[0]}: {output[1]}" for output in trajectory])
            # Todo: Implement the optimization logic using the model.