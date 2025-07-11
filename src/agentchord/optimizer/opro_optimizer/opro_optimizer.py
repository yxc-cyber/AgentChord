import json
import re
from typing import Dict, Optional, Union

from ...agent_system import BaseAgent, Input
from ...model import ModelConfig, ModelFactory
from ..base_optimizer import BaseOptimizer
from .meta_prompt import META_PROMPT_INPUT, META_PROMPT_INSTRUCTION


class OPROOptimizer(BaseOptimizer):
    """
    Optimizer for OPRO (Optimization by PROmpting).
    This optimizer is designed to work with the OPRO framework, which focuses on optimizing agent interactions.
    """

    def __init__(
            self,
            agents: Dict[str, BaseAgent],
            model_config: ModelConfig,
            name: str = "OPROOptimizer",
            log_name: str = ""
        ):
        """
        Initialize the OPRO optimizer with the agent system.
        :param agents: The agents to be optimized.
        """
        super().__init__(agents=agents, name=name, log_name=log_name)
        self.model_config = model_config
        self.model = ModelFactory(self.model_config).create_model()

    def step(self, performance: str) -> None:
        """ 
        Perform a single optimization step.
        This method will use the model to optimize the prompts of the agents based on the optimization history and trajectories.
        """
        self.update_attributes()
        self.update_optimization_history(performance)

        inference_trajectories = [" -> ".join([f"{output[0]}: {output[1]}" for output in trajectory]) for trajectory in self.optimization_info]
        instrcution_str = META_PROMPT_INSTRUCTION
        input_str = META_PROMPT_INPUT.format(
            agent_structure=json.dumps(self.prompts, indent=2),
            optimization_history=json.dumps(self.optimization_history, indent=2),
            inference_trajectories=json.dumps(inference_trajectories, indent=2)
        )

        messages = [
            {"role": "system", "content": instrcution_str},
            {"role": "user", "content": input_str}
        ]
        self.logger.debug(f"Optimization message: {messages}")
        output_message = self.model.completion(
            messages=messages,
        ).choices[0].message
        self.logger.debug(f"Optimization result: {output_message.content}")
        optimization_output = self.parse_optimization_output(output_message.content)
        self.update_prompts(optimization_output)
        Input().optimization_info_initialization()

    def parse_optimization_output(self, output_message: str) -> Dict[str, str]:
        """
        Parse the output message from the model to extract the new prompts for the agents.
        :param output_message: The output message from the model.
        :return: A dictionary mapping agent names to their new prompts.
        """
        parsed_output = self.parse_json_string(output_message)
        if isinstance(parsed_output, dict):
            if "agent_prompts" in parsed_output and self.check_optimization_output(parsed_output["agent_prompts"]):
                return parsed_output["agent_prompts"]
        elif isinstance(parsed_output, list):
            for item in parsed_output:
                if isinstance(item, dict) and "agent_prompts" in item and self.check_optimization_output(item["agent_prompts"]):
                    return item["agent_prompts"]
        else:
            raise ValueError("The output JSON is neither a dictionary nor a list of dictionaries.")
        return dict()
        
    def parse_json_string(self, json_string: str) -> Optional[Union[dict, list]]:
        """
        Parse a JSON string and return the corresponding Python object.
        """
        def sanitize_invalid_escapes(s):
            # Replace \' with '
            s = re.sub(r"\\'", "'", s)
            # Replace all other invalid backslash escapes (except valid ones)
            s = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
            return s
        json_string = sanitize_invalid_escapes(json_string)
        def extract_from_markdown(json_string: str) -> str:
            """
            Extract JSON content from markdown format.
            """
            json_format = re.compile(r"```(json)?(.*)```", re.DOTALL)
            match = json_format.match(json_string)
            if match:
                return match.group(2).strip()
            return json_string
        json_string = extract_from_markdown(json_string)

        objects = list()
        start_idx = 0
        while start_idx < len(json_string):
            while start_idx < len(json_string) and json_string[start_idx] != '{' and json_string[start_idx] != '[':
                start_idx += 1
            if start_idx >= len(json_string):
                break
            try:
                obj, end_idx = json.JSONDecoder().raw_decode(json_string, idx=start_idx)
                objects.append(obj)
                start_idx = end_idx
            except json.JSONDecodeError:
                # If we can't decode, it might be due to multiple JSON objects or invalid format
                break
        if len(objects) == 1:
            return objects[0]
        elif len(objects) > 1:
            return objects
        else:
            return None
        
    def check_optimization_output(self, output: Dict[str, str]) -> bool:
        """
        Check if the optimization output is valid.
        :param output: The output from the model.
        :return: True if the output is valid, False otherwise.
        """
        if not isinstance(output, dict):
            return False
        for agent_name, prompt in output.items():
            if not isinstance(agent_name, str) or not isinstance(prompt, str):
                return False
            if not agent_name in self.agents:
                return False
        return True