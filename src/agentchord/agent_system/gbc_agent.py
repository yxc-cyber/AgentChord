import json
from copy import copy
from typing import List, Optional

from ..environment import BaseEnvironment
from ..gbc_object import GBC, GBCBase
from ..metadata import BaseMetaData
from ..model import ModelConfig
from ..utils import (
    EMPTY_PLACEHOLDER,
    INPUT,
    INPUT_FOOTER,
    INPUT_HEADER,
    INPUT_SEPARATOR,
    OUTPUT_INFO,
    OUTPUT_NO_ACTION,
    OUTPUT_TOO_MANY_ACTIONS,
    OUTPUT_TOO_MUCH_THINKING,
    TOOL_FOOTER,
    TOOL_HEADER,
    TOOL_INFO,
    TOOL_RESULT_INFO,
)
from .base_agent import BaseAgent


class GBCAgent(BaseAgent):
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
        super().__init__(
            system_name=system_name,
            environment=environment,
            prompt=prompt,
            model_config=model_config,
            tools=tools,
            maximum_loops=maximum_loops,
            log_name=log_name,
        )
    
    def execution(self, meta_data: BaseMetaData) -> BaseMetaData:
        self.logger.debug(f"Receiving {meta_data}")
        input_content = meta_data.input
        previous_tool_usage = meta_data.tool
        meta_data.input = ""
        meta_data.output = ""

        # Prepare the input content
        if isinstance(input_content, str):
            content = INPUT.format(
                content = input_content or EMPTY_PLACEHOLDER,
            )
            content = f"{INPUT_HEADER}{content}{INPUT_FOOTER}"
            if isinstance(input_content, GBCBase):
                content = GBC(
                    content,
                    connections=input_content.get_connections(),
                    weights=input_content.get_weights(),
                    subject=self
                )
            else:
                content = GBC(
                    content,
                    connections=input_content,
                    weights=1.0,
                    subject=self
                )
            self.messages.append({
                "role": "user",
                "content": content
            })
        elif isinstance(input_content, list):
            content_list = [INPUT.format(
                content = input_content or EMPTY_PLACEHOLDER
            ) for input_content in input_content]
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
                subject=self
            )
            self.messages.append({
                "role": "user",
                "content": content
            })
        else:
            raise ValueError("Input must be either strings or lists of strings.")
        
        # Begin the execution loop
        terminate = False
        loop_counter = 0
        connection_pool = copy(content.get_connections())
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
                temp_connection_pool = list()
                for tool_call in output_message.tool_calls:
                    tool_call_id = tool_call.id
                    tool_name = tool_call.function.name
                    try:
                        tool_arguments = json.loads(tool_call.function.arguments)
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
                        output_info = OUTPUT_INFO.format(
                            output = tool_result_dict["output"] or EMPTY_PLACEHOLDER
                        )
                        output_info = GBC(
                            output_info,
                            connections=output_message.gbc_tool_calls.get_connections(),
                            weights=output_message.gbc_tool_calls.get_weights(),
                            subject=self
                        )
                        meta_data.output = GBC(
                            str(tool_result_dict["output"]), 
                            connections=output_info,
                            weights=1.0,
                            subject=self
                        )
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
                    
                    tool_info = TOOL_INFO.format(
                        tool_name = tool_name or EMPTY_PLACEHOLDER,
                        tool_parameters = json.dumps(tool_arguments, indent=2) or EMPTY_PLACEHOLDER
                    )
                    tool_info = GBC(
                        tool_info,
                        connections=output_message.gbc_tool_calls.get_connections(),
                        weights=output_message.gbc_tool_calls.get_weights(),
                        subject=self
                    )

                    tool_result_info = TOOL_RESULT_INFO.format(tool_result = tool_result or EMPTY_PLACEHOLDER)
                    tool_result_info = GBC(
                        tool_result_info,
                        connections=tool_info,
                        weights=1.0,
                        subject=self
                    )
                    temp_connection_pool.append(tool_result_info)
                    tool_result = f"{TOOL_HEADER}{tool_result}{TOOL_FOOTER}"
                    tool_result = GBC(
                        tool_result,
                        connections=copy(connection_pool),
                        weights=[1.0] * len(connection_pool),
                        subject=self
                    )
                    self.messages.append({
                        "role":"tool",
                        "tool_call_id":tool_call_id,
                        "name": tool_name,
                        "content":tool_result
                    })
                connection_pool.extend(temp_connection_pool)
                self.messages[-1]["content"].bind_connections(
                    connections=copy(connection_pool),
                    weights=[1.0] * len(connection_pool)
                )
            else:
                if output_message.content.startswith("<think>") and not output_message.content.endswith("</think>"):
                    self.logger.warning("Output content starts with <think> but does not end with </think>. Possible incomplete thinking.")
                    output_info = OUTPUT_TOO_MUCH_THINKING
                else:
                    output_info = OUTPUT_INFO.format(
                        output = OUTPUT_NO_ACTION.format(content = output_message.content or EMPTY_PLACEHOLDER)
                    )
                output_info = GBC(
                    output_info,
                    connections=output_message.gbc_content.get_connections(),
                    weights=output_message.gbc_content.get_weights(),
                    subject=self
                )
                meta_data.output = GBC(
                    OUTPUT_NO_ACTION.format(content = output_message.content or EMPTY_PLACEHOLDER),
                    connections=output_info,
                    weights=1.0,
                    subject=self
                )
                terminate = True
            loop_counter += 1
        if loop_counter >= self.maximum_loops and not (isinstance(meta_data.output, GBCBase)):
            self.logger.warning(f"Maximum loops reached ({self.maximum_loops}) without termination. Returning empty output.")
            output_info = OUTPUT_INFO.format(
                output = OUTPUT_TOO_MANY_ACTIONS
            )
            output_info = GBC(
                output_info,
                connections=self.messages[-1]["content"].get_connections(),
                weights=self.messages[-1]["content"].get_weights(),
                subject=self
            )
            meta_data.output = GBC(
                OUTPUT_TOO_MANY_ACTIONS,
                connections=output_info,
                weights=1.0,
                subject=self
            )
        self.logger.debug(f"Returning {meta_data}")
        return meta_data