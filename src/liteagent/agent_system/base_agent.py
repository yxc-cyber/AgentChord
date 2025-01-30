import json
from typing import Any

import litellm

from ..environment import BaseEnvironment
from ..metadata import BaseMetaData
from ..utils import Logger, ModelConfig
from .base_agent_system import BaseAgentSystem


class BaseAgent(BaseAgentSystem):
    def __init__(self, system_name: str, environment: BaseEnvironment, prompt: str, model_config: ModelConfig):
        super().__init__(system_name=system_name, environment=environment)
        self.subsystems = None
        self.on_start_actions = None
        self.on_completion_actions = None
        self.subsystem_sequence = None
        self.prompt = prompt
        self.client_config = model_config.client_config
        self.messages = list()
        self.logger = Logger(self.system_name)

    def completion_loop(self, meta_data: BaseMetaData) -> BaseMetaData:
        meta_data = self.completion(meta_data)
        return meta_data
    
    def completion(self, meta_data: BaseMetaData) -> BaseMetaData:
        self.logger.debug(f"Receiving {meta_data}")
        input_content = meta_data.input
        self.messages.append({"role": "user", "content": input_content})
        self.logger.debug(f"Message history: {self.messages}")
        output_message = litellm.completion(
            messages=self.messages,
            tools=self.tool_descriptions,
            **self.client_config
        ).choices[0].message
        self.logger.debug(f"New message: {output_message.json()}")
        self.messages.append(output_message)
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

    def get_pipeline_description_list(self, indent_level: int) -> list:
        return list()
    
    def get_pipeline_description(self) -> str:
        return f"Agent: {self.system_name} ({self.__class__})"