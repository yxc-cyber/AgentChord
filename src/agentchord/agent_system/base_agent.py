import json
from typing import Any, Dict, List, Optional, Self, Tuple, Union

from litellm import Message

from ..environment import BaseEnvironment
from ..metadata import BaseMetaData
from ..model import ModelConfig, ModelFactory
from ..utils import (
    EMPTY_PLACEHOLDER,
    INPUT,
    INPUT_FOOTER,
    INPUT_HEADER,
    INPUT_SEPARATOR,
    OUTPUT_NO_ACTION,
    OUTPUT_TOO_MANY_ACTIONS,
    OUTPUT_TOO_MUCH_THINKING,
    PROMPT_TEMPLATE,
    TOOL_FOOTER,
    TOOL_HEADER,
)
from .base_agent_system import BaseAgentSystem


class BaseAgent(BaseAgentSystem):
    def __init__(
            self,
            system_name: str,
            environment: BaseEnvironment,
            prompt: str,
            model_config: ModelConfig,
            tools: Optional[List[str]] = None,
            maximum_loops: int = 5,
            log_name: str = "",
        ):
        self.prompt = prompt
        self.model_config = model_config
        self.model = ModelFactory(self.model_config).create_model()
        self.messages = list()
        super().__init__(system_name=system_name, environment=environment, maximum_loops=maximum_loops, log_name=log_name, tools=tools)
        self.messages_initialization()
        self.optimization_info = list()
        self.logger.debug(f"Available tools: {self.tool_descriptions}")

    def execution_loop(self, meta_data: BaseMetaData, loop: bool = False) -> BaseMetaData:
        meta_data = self.execution(meta_data)
        return meta_data
    
    def completion(self, messages: List[dict], tools: Optional[List[str]] = None) -> Message:
        if tools is not None:
            tool_descriptions = list()
            for tool in tools:
                if tool in self.tool_description_dict:
                    tool_descriptions.append(self.tool_description_dict[tool])
                else:
                    raise ValueError(f"Tool {tool} not found in tool descriptions.")
            output_message = self.model.completion(
                messages=messages,
                tools=tool_descriptions
            ).choices[0].message
        else:
            output_message = self.model.completion(
                messages=messages,
                tools=self.tool_descriptions
            ).choices[0].message
        return output_message
    
    def execution(self, meta_data: BaseMetaData) -> BaseMetaData:
        self.logger.debug(f"Receiving {meta_data}")
        input_content = meta_data.input
        previous_tool_usage = meta_data.tool
        meta_data.input = ""
        # meta_data.tool = list()  # Keep the previous tool usage
        meta_data.output = ""

        # Prepare the input content
        if isinstance(input_content, str):
            content = INPUT.format(
                content = input_content or EMPTY_PLACEHOLDER
            )
            content = f"{INPUT_HEADER}{content}{INPUT_FOOTER}"
            self.messages.append({
                "role": "user",
                "content": content
            })
        elif isinstance(input_content, list):
            content_list = [INPUT.format(
                content = input_content or EMPTY_PLACEHOLDER
            ) for input_content in input_content]
            content = f"{INPUT_HEADER}{INPUT_SEPARATOR.join(content_list)}{INPUT_FOOTER}"
            self.messages.append({
                "role": "user",
                "content": content
            })
        else:
            raise ValueError("Input must be either strings or lists of strings.")
        
        # Begin the execution loop
        terminate = False
        loop_counter = 0
        while not terminate and loop_counter < self.maximum_loops:
            self.logger.debug(f"Message history: {self.messages}")
            if loop_counter == self.maximum_loops - 1:
                output_message = self.completion(self.messages, [self.inner_environment.TERMINATE])
            else:
                output_message = self.completion(self.messages)
            if output_message.tool_calls:
                output_message.tool_calls = output_message.tool_calls[:1]  # Limit to the first tool call for simplicity
            self.logger.debug(f"New message: {output_message.json()}")
            self.messages.append(output_message.json())
            if output_message.tool_calls:
                for tool_call in output_message.tool_calls:
                    tool_call_id = tool_call.id
                    tool_name = tool_call.function.name.replace(INPUT_HEADER, "").replace(INPUT_FOOTER, "").replace(INPUT_SEPARATOR, "").replace(TOOL_HEADER, "").replace(TOOL_FOOTER, "")
                    try:
                        tool_arguments = json.loads(tool_call.function.arguments.replace(INPUT_HEADER, "").replace(INPUT_FOOTER, "").replace(INPUT_SEPARATOR, "").replace(TOOL_HEADER, "").replace(TOOL_FOOTER, ""))
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to decode tool arguments: {tool_call.function.arguments}")
                        tool_arguments = dict()
                    if tool_name == self.inner_environment.TERMINATE:
                        tool_result = self.inner_environment.apply_tool(tool_name, tool_arguments)
                        self.logger.debug(f"Tool result: {tool_result}")
                        tool_result_dict = json.loads(tool_result)
                        if "output" not in tool_result_dict:
                            assert "message" in tool_result_dict
                            tool_result_dict["output"] = tool_result_dict["message"]
                        meta_data.output = tool_result_dict["output"]
                        terminate = True
                    else:
                        if tool_name in self.tools:
                            tool_result = self.environment.apply_tool(tool_name, tool_arguments)
                        else:
                            tool_result = json.dumps({
                                "message": f"Tool '{tool_name}' is not available. Available tools: {", ".join(self.tools)}."
                            })
                        self.logger.debug(f"Tool result: {tool_result}")
                        meta_data.tool.append({"tool_name": tool_name, "tool_arguments": tool_arguments, "tool_result": tool_result})
                    
                    tool_result = f"{TOOL_HEADER}{tool_result}{TOOL_FOOTER}"
                    self.messages.append({
                        "role":"tool",
                        "tool_call_id":tool_call_id,
                        "name": tool_name,
                        "content":tool_result
                    })
            else:
                if output_message.content.startswith("<think>") and not "</think>" in output_message.content:
                    self.logger.warning("Output content starts with <think> but does not end with </think>. Possible incomplete thinking.")
                    meta_data.output = OUTPUT_TOO_MUCH_THINKING
                else:
                    meta_data.output = OUTPUT_NO_ACTION.format(content = output_message.content or EMPTY_PLACEHOLDER)
                terminate = True
            loop_counter += 1
        if loop_counter >= self.maximum_loops and meta_data.output == "":
            self.logger.warning(f"Maximum loops reached ({self.maximum_loops}) without termination. Returning empty output.")
            meta_data.output = OUTPUT_TOO_MANY_ACTIONS
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
        self.messages = [{"role": "system", "content": PROMPT_TEMPLATE.format(prompt=self.prompt)}]

    def messages_simplification(
            self,
            remove_roles: Optional[List[str]] = None,
            remove_tools: Optional[List[str]] = None,
            keep_last_n: Optional[int] = None
        ) -> List[dict]:
        simplified_messages = self.messages
        if remove_roles is not None:
            simplified_messages = [message for message in simplified_messages if message["role"] not in remove_roles]
        if remove_tools is not None:
            simplified_messages = [
                message
                for message in simplified_messages
                if not (
                    (message["role"] == "tool" and message.get("name") in remove_tools)
                    or (
                        message["role"] == "assistant"
                        and any(
                            tool_call.get("function", {}).get("name") in remove_tools
                            for tool_call in message.get("tool_calls", []) or []
                        )
                    )
                )
            ]
        if keep_last_n is not None and simplified_messages:
            prompt_message = simplified_messages[0]
            remaining_messages = simplified_messages[1:]
            if keep_last_n <= 0:
                simplified_messages = [prompt_message]
            elif len(remaining_messages) > keep_last_n:
                simplified_messages = [prompt_message] + remaining_messages[-keep_last_n:]
            else:
                simplified_messages = [prompt_message] + remaining_messages
        self.messages = simplified_messages
        return simplified_messages

    def _get_pipeline_description_list(self) -> list:
        return list()
    
    def get_pipeline_description(self) -> str:
        return f"Agent: {self.system_name} ({self.__class__})"
    
    def set_debug_level(self, debug: bool):
        self.logger.set_debug_level(debug)

    def set_log_redirection(self, file_name: str):
        self.logger.set_log_redirection(file_name)

    def get_agents(self) -> Dict[str, Self]:
        return {self.system_name: self}
    
    def get_prompt(self) -> str:
        """
        Get the prompt of the agent.
        """
        return self.prompt
    
    def get_tools(self) -> List[dict]:
        """
        Get the tool descriptions of the agent.
        """
        return [self.tool_description_dict[tool] for tool in self.tools if tool in self.tool_description_dict]
    
    def set_prompt(self, prompt: str):
        """
        Set the prompt of the agent.
        :param prompt: The new prompt for the agent.
        """
        self.prompt = prompt
        self.messages_initialization()

    def append_optimization_info(self, info: List[List[Tuple[Union[str, Self], str]]]) -> List[List[Tuple[Union[str, Self], str]]]:
        """
        Append optimization information to the agent's optimization info list.
        """
        self.optimization_info.extend(info)
        return self.optimization_info

    def get_optimization_info(self) -> List[List[Tuple[Union[str, Self], str]]]:
        """
        Get the optimization information of the agent.
        """
        return self.optimization_info