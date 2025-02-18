import json
from typing import Any, List

import litellm
from litellm import Message

from ..environment import BaseEnvironment
from ..metadata import BaseMetaData
from ..utils import ModelConfig
from .base_agent_system import BaseAgentSystem


class BaseAgent(BaseAgentSystem):
    def __init__(self, system_name: str, environment: BaseEnvironment, prompt: str, model_config: ModelConfig, log_name: str = ""):
        super().__init__(system_name=system_name, environment=environment, log_name=log_name)
        self.subsystems = None
        self.on_start_actions = None
        self.on_completion_actions = None
        self.subsystem_sequence = None
        self.prompt = prompt
        self.model_config = model_config
        self.messages = list()
        self.messages_initialization()

    def execution_loop(self, meta_data: BaseMetaData) -> BaseMetaData:
        meta_data = self.execution(meta_data)
        return meta_data
    
    def completion(self, messages: List[dict]) -> Message:
        if self.model_config.client_model:
            output_message = litellm.completion(
                messages=messages,
                tools=self.tool_descriptions,
                model=self.model_config.client_model
            ).choices[0].message
        else:
            # Todo: adapt local models
            raise NotImplementedError
        return output_message
    
    def execution(self, meta_data: BaseMetaData) -> BaseMetaData:
        self.logger.debug(f"Receiving {meta_data}")
        input_content = meta_data.input
        self.messages.append({"role": "user", "content": input_content})
        self.logger.debug(f"Message history: {self.messages}")
        output_message = self.completion(self.messages)
        self.logger.debug(f"New message: {output_message.json()}")
        self.messages.append(output_message.json())
        if output_message.tool_calls:
            tool_call_id = output_message.tool_calls[0].id
            tool_name = output_message.tool_calls[0].function.name
            tool_arguments = json.loads(output_message.tool_calls[0].function.arguments)
            tool_result = self.environment.apply_tool(tool_name, tool_arguments)
            self.messages.append({
                "role":"tool",
                "tool_call_id":tool_call_id,
                "name": tool_name, 
                "content":tool_result
            })
            meta_data.output = None
            meta_data.tool = {"tool_name": tool_name, "tool_arguments": tool_arguments, "tool_result": tool_result}
        else:
            meta_data.output = output_message.content
            meta_data.tool = None
        self.logger.debug(f"Returning {meta_data}")
        return meta_data
    
    def on_initialization(self) -> BaseMetaData:
        self.logger.debug("On initialization")
        self.messages_initialization()
        meta_data = self.environment.get_initial_setup()
        self.logger.debug("Initialization done")
        return meta_data
    
    def on_finalization(self, meta_data: BaseMetaData) -> Any:
        self.logger.debug("On finalization")
        final_data = meta_data.output
        self.logger.debug("Finalization done")
        return final_data

    def messages_initialization(self):
        self.messages = [{"role": "system", "content": self.prompt}]

    def get_pipeline_description_list(self) -> list:
        return list()
    
    def get_pipeline_description(self) -> str:
        return f"Agent: {self.system_name} ({self.__class__})"
    
    def set_debug_level(self, debug: bool):
        self.logger.set_debug_level(debug)

    def set_log_redirection(self, file_name: str):
        self.logger.set_log_redirection(file_name)