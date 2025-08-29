import json
import re
from copy import deepcopy
from typing import Dict, Optional, Union

from ..agent_system import BaseAgent, Input
from ..utils import DONT_CHANGE_FOOTER, DONT_CHANGE_HEADER, Logger, WandBConfig


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
                dont_change_blocks = []
                pattern = re.compile(f"{DONT_CHANGE_HEADER}(.*?){DONT_CHANGE_FOOTER}", re.DOTALL)
                matches = pattern.findall(self.prompts[agent_name])
                for match in matches:
                    dont_change_blocks.append(match)
                # Replace DONT_CHANGE blocks in new_prompt with extracted ones
                new_blocks = pattern.findall(new_prompt)
                new_prompt_parts = []
                last_end = 0
                for i, match in enumerate(new_blocks):
                    start = new_prompt.find(DONT_CHANGE_HEADER, last_end)
                    end = new_prompt.find(DONT_CHANGE_FOOTER, start) + len(DONT_CHANGE_FOOTER)
                    if i < len(dont_change_blocks):
                        # Replace with extracted block
                        block = f"{DONT_CHANGE_HEADER}{dont_change_blocks[i]}{DONT_CHANGE_FOOTER}"
                    else:
                        # Skip exceeding blocks in new_prompt
                        block = ""
                    new_prompt_parts.append(new_prompt[last_end:start])
                    new_prompt_parts.append(block)
                    last_end = end
                new_prompt_parts.append(new_prompt[last_end:])
                new_prompt = "".join(new_prompt_parts)
                # Append any remaining extracted blocks at the end
                if len(dont_change_blocks) > len(new_blocks):
                    for i in range(len(new_blocks), len(dont_change_blocks)):
                        new_prompt += f"{DONT_CHANGE_HEADER}{dont_change_blocks[i]}{DONT_CHANGE_FOOTER}"
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

    def save_optimizer_state(self, file_path: str):
        """
        Save the state of the optimizer to a file.
        :param file_path: The path to the file where the state will be saved.
        """
        with open(file_path, 'w') as f:
            json.dump(self.optimization_history, f, indent=2)
        self.logger.debug(f"Optimizer state saved to {file_path}")

    def load_optimizer_state(self, file_path: str):
        """
        Load the state of the optimizer from a file.
        :param file_path: The path to the file from which the state will be loaded.
        """
        with open(file_path, 'r') as f:
            self.optimization_history = json.load(f)
        self.logger.debug(f"Optimizer state loaded from {file_path}")
        self.update_attributes()