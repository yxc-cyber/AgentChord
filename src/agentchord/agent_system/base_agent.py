import json
from typing import Any, List

from litellm import Message

from ..environment import BaseEnvironment
from ..gbc_object import GBC, GBCBase
from ..metadata import BaseMetaData
from ..model import ModelConfig, ModelFactory
from ..utils import (
    EMPTY_PLACEHOLDER,
    INPUT_FOOTER,
    INPUT_HEADER,
    INPUT_SEPARATOR,
    INPUT_WITH_NOTE,
    NOTE_NO_ACTION,
    OUTPUT_NOTE_INFO,
    TOOL_FOOTER,
    TOOL_HEADER,
    TOOL_INFO,
)
from .base_agent_system import BaseAgentSystem


class BaseAgent(BaseAgentSystem):
    def __init__(self, system_name: str, environment: BaseEnvironment, prompt: str, model_config: ModelConfig, maximum_loops: int = 5, log_name: str = ""):
        self.prompt = prompt
        self.model_config = model_config
        self.model = ModelFactory(self.model_config).create_model()
        self.messages = list()
        super().__init__(system_name=system_name, environment=environment, maximum_loops=maximum_loops, log_name=log_name)
        self.messages_initialization()
        self.logger.debug(f"Available tools: {self.tool_descriptions}")

    def execution_loop(self, meta_data: BaseMetaData, loop: bool = False) -> BaseMetaData:
        meta_data = self.execution(meta_data)
        return meta_data
    
    def completion(self, messages: List[dict]) -> Message:
        output_message = self.model.completion(
            messages=messages,
            tools=self.tool_descriptions
        ).choices[0].message
        return output_message
    
    def execution(self, meta_data: BaseMetaData) -> BaseMetaData:
        self.logger.debug(f"Receiving {meta_data}")
        input_content = meta_data.input
        previous_note = meta_data.note
        previous_tool_usage = meta_data.tool
        meta_data.input = ""
        meta_data.note = ""
        # meta_data.tool = list()  # Keep the previous tool usage
        meta_data.output = ""

        # Prepare the input content and previous note
        assert (isinstance(input_content, str) and isinstance(previous_note, str)) \
            or (isinstance(input_content, list) and isinstance(previous_note, list) and len(input_content)==len(previous_note)), \
            "Input and note must be either both strings or both lists of strings."
        if isinstance(input_content, str):
            content = INPUT_WITH_NOTE.format(
                input = input_content or EMPTY_PLACEHOLDER,
                note = previous_note or EMPTY_PLACEHOLDER
            )
            content = f"{INPUT_HEADER}{content}{INPUT_FOOTER}"
            if isinstance(input_content, GBCBase):
                content = GBC(
                    content,
                    connections=input_content.get_connections(),
                    weights=input_content.get_weights(),
                    subject=self.system_name
                )
            else:
                content = GBC(
                    content,
                    connections=input_content,
                    weights=1.0,
                    subject=self.system_name
                )
            self.messages.append({
                "role": "user",
                "content": content
            })
        elif isinstance(input_content, list):
            content_list = [INPUT_WITH_NOTE.format(
                input = input_content or EMPTY_PLACEHOLDER,
                note = previous_note or EMPTY_PLACEHOLDER
            ) for input_content, previous_note in zip(input_content, previous_note)]
            content = f"{INPUT_HEADER}{INPUT_SEPARATOR.join(content_list)}{INPUT_FOOTER}"
            connections = list()
            weights = list()
            for single_input_content in input_content:
                if isinstance(single_input_content, GBCBase):
                    connections.extend(single_input_content.get_connections())
                    weights.extend(single_input_content.get_weights())
                else:
                    connections.append(single_input_content)
                    weights.append(1.0)
            content = GBC(
                content,
                connections=connections,
                weights=weights,
                subject=self.system_name
            )
            self.messages.append({
                "role": "user",
                "content": content
            })
        else:
            raise ValueError("Input and note must be either both strings or both lists of strings.")
        
        # Begin the execution loop
        terminate = False
        loop_counter = 0
        connection_pool = content.get_connections()
        while not terminate and loop_counter < self.maximum_loops:
            self.logger.debug(f"Message history: {self.messages}")
            output_message = self.completion(self.messages)
            self.logger.debug(f"New message: {output_message.json()}")
            self.messages.append(output_message.json())
            if output_message.tool_calls:
                for tool_call in output_message.tool_calls:
                    tool_call_id = tool_call.id
                    tool_name = tool_call.function.name
                    tool_arguments = json.loads(tool_call.function.arguments)
                    if tool_name == self.inner_environment.TERMINATE:
                        tool_result = self.inner_environment.apply_tool(tool_name, tool_arguments)
                        tool_result_dict = json.loads(tool_result)
                        output_note_info = OUTPUT_NOTE_INFO.format(
                            output = tool_result_dict["output"] or EMPTY_PLACEHOLDER,
                            note = tool_result_dict["note"] or EMPTY_PLACEHOLDER
                        )
                        output_note_info = GBC(
                            output_note_info,
                            connections=output_message.gbc_tool_calls.get_connections(),
                            weights=output_message.gbc_tool_calls.get_weights(),
                            subject=self.system_name
                        )
                        meta_data.output = GBC(
                            tool_result_dict["output"], 
                            connections=output_note_info,
                            weights=1.0,
                            subject=self.system_name
                        )
                        meta_data.note = GBC(
                            tool_result_dict["note"],
                            connections=output_note_info,
                            weights=1.0,
                            sdubject=self.system_name
                        )
                        terminate = True
                    else:
                        tool_result = self.environment.apply_tool(tool_name, tool_arguments)
                        meta_data.tool.append({"tool_name": tool_name, "tool_arguments": tool_arguments, "tool_result": tool_result})
                    
                    tool_info = TOOL_INFO.format(tool_result = tool_result or EMPTY_PLACEHOLDER)
                    tool_info = GBC(
                        tool_info,
                        connections=output_message.gbc_tool_calls.get_connections(),
                        weights=output_message.gbc_tool_calls.get_weights(),
                        subject=self.system_name
                    )
                    connection_pool.append(tool_info)
                    tool_result = f"{TOOL_HEADER}{tool_result}{TOOL_FOOTER}"
                    tool_result = GBC(
                        tool_result,
                        connections=connection_pool,
                        weights=[1.0] * len(connection_pool),
                        subject=self.system_name
                    )
                    self.messages.append({
                        "role":"tool",
                        "tool_call_id":tool_call_id,
                        "name": tool_name,
                        "content":tool_result
                    })
            else:
                output_note_info = OUTPUT_NOTE_INFO.format(
                    output = output_message.content or EMPTY_PLACEHOLDER,
                    note = EMPTY_PLACEHOLDER
                )
                output_note_info = GBC(
                    output_note_info,
                    connections=output_message.gbc_content.get_connections(),
                    weights=output_message.gbc_content.get_weights(),
                    subject=self.system_name
                )
                meta_data.output = GBC(
                    "",
                    connections=output_note_info,
                    weights=1.0,
                    subject=self.system_name
                )
                meta_data.note = GBC(
                    NOTE_NO_ACTION.format(output = output_message.content or EMPTY_PLACEHOLDER),
                    connections=output_note_info,
                    weights=1.0,
                    subject=self.system_name
                )
                terminate = True
            loop_counter += 1
        self.logger.debug(f"Returning {meta_data}")
        return meta_data
    
    def set_environment(self, environment):
        self.messages_initialization()
        super().set_environment(environment)
    
    def on_initialization(self) -> BaseMetaData:
        self.logger.debug("On initialization")
        self.messages_initialization()
        meta_data = self.environment.get_initial_metadata()
        self.logger.debug("Initialization done")
        return meta_data
    
    def on_finalization(self, meta_data: BaseMetaData) -> Any:
        self.logger.debug("On finalization")
        final_data = meta_data.output
        self.logger.debug("Finalization done")
        return final_data

    def messages_initialization(self):
        self.messages = [{"role": "system", "content": self.prompt}]

    def _get_pipeline_description_list(self) -> list:
        return list()
    
    def get_pipeline_description(self) -> str:
        return f"Agent: {self.system_name} ({self.__class__})"
    
    def set_debug_level(self, debug: bool):
        self.logger.set_debug_level(debug)

    def set_log_redirection(self, file_name: str):
        self.logger.set_log_redirection(file_name)