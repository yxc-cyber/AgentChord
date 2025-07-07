from typing import List, Self, Tuple, Union

from ..agent_system import BaseAgent
from ..utils import SingletonMeta


class Input(metaclass=SingletonMeta):
    """
    Input class represents an input in the AgentChord system.
    """

    def __init__(self):
        """
        Initialize the Input agent with input data.
        :param input_data: The input data to be represented by this agent.
        """
        self.optimization_info = list()
    
    def append_optimization_info(self, info: List[List[Tuple[Union[str, BaseAgent, Self], str]]]) -> List[List[Tuple[Union[str, BaseAgent, Self], str]]]:
        """
        Append optimization information to the agent's optimization info list.
        """
        self.optimization_info.extend(info)
        return self.optimization_info

    def get_optimization_info(self) -> List[List[Tuple[Union[str, BaseAgent, Self], str]]]:
        """
        Get the optimization information of the agent.
        """
        return self.optimization_info
    
    def __repr__(self):
        return "Input"