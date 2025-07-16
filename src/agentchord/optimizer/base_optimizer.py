from copy import deepcopy
from typing import Dict, Optional, Union

from ..agent_system import BaseAgent, Input
from ..utils import Logger, WandBConfig


class BaseOptimizer:
    """
    Base class for optimizers.
    """

    def __init__(
            self,
            agents: Dict[str, BaseAgent],
            name: str = "BaseOptimizer",
            log_name: str = "",
            wandb_config: Optional[WandBConfig] = None
        ):
        """
        Initialize the optimizer with the agent system.
        :param agent_system: The agents to be optimized.
        """
        self.name = name
        self.log_name = log_name
        self.agents = agents
        self.trajectories = list()
        self.optimization_history = list()
        self.update_attributes()
        self.logger = Logger(self.name, self.log_name)
        self.wandb_config = wandb_config
        if wandb_config:
            self.init_wandb(wandb_config)

    def update_attributes(self):
        """
        Update the attributes of the optimizer.
        This method is called to ensure that the optimizer has the latest information about the agents and their configurations.
        """
        self.optimization_info = Input().get_optimization_info()
        self.prompts = {agent_name: agent.get_prompt() for agent_name, agent in self.agents.items()}

    def update_optimization_history(self, performance: str):
        current_prompts = deepcopy(self.prompts)
        self.optimization_history.append({
            "prompts": current_prompts,
            "performance": performance
        })

    def update_prompts(self, new_prompts: Dict[str, str]):
        """
        Update the prompts of the agents.
        :param new_prompts: A dictionary mapping agent names to their new prompts.
        """
        for agent_name, new_prompt in new_prompts.items():
            if agent_name in self.agents:
                self.agents[agent_name].set_prompt(new_prompt)
                self.prompts[agent_name] = new_prompt

    def step(self):
        """
        Perform a single optimization step.
        This method should be overridden by subclasses to implement specific optimization logic.
        """
        raise NotImplementedError("The step method must be implemented by subclasses.")
    
    def set_debug_level(self, debug: bool):
        """
        Set the debug level for the logger.
        :param debug: If True, set the logger to debug level; otherwise, set it to info level.
        """
        self.logger.set_debug_level(debug)

    def set_log_redirection(self, file_name: str):
        """
        Set the log redirection to a specific file.
        :param file_name: The name of the file to redirect logs to.
        """
        self.logger.set_log_redirection(file_name)

    def init_wandb(self, wandb_config: WandBConfig):
        """
        Initialize Weights & Biases for logging.
        This method should be called to set up Weights & Biases if it is being used for logging.
        """
        import wandb
        wandb.login()
        wandb.init(**wandb_config.to_dict())
        self.logger.debug(f"Weights & Biases initialized: {wandb_config}")

    def report_to_wandb(self, info: Dict[str, Union[int, float, str]]):
        """
        Report information to Weights & Biases.
        :param info: A dictionary containing information to log.
        """
        import wandb
        wandb.log(info)
        self.logger.debug(f"Reported to Weights & Biases: {info}")

    def finish_wandb(self):
        """
        Finish the Weights & Biases run.
        This method should be called at the end of the optimization process to finalize the logging.
        """
        import wandb
        wandb.finish()
        self.logger.debug("Weights & Biases run finished.")