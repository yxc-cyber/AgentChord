from .base_agent import BaseAgent
from ..environment import BaseEnvironment
from ..model import ModelConfig
from ..metadata import BaseMetaData

from ..utils import USER_PROMPT_TEMPLATE
from ..utils import (
    EMPTY_PLACEHOLDER,
    INPUT,
    INPUT_FOOTER,
    INPUT_HEADER,
    INPUT_SEPARATOR,
    OUTPUT_TOO_MANY_ACTIONS,
    OUTPUT_TOO_MUCH_THINKING,
    TOOL_FOOTER,
    TOOL_HEADER,
)

from typing import Optional, List
import json

class BaseUser(BaseAgent):
    def __init__(self,
            system_name: str,
            prompt: str,
            model_config: ModelConfig,
            tools: Optional[List[str]] = None,
            maximum_loops: int = 5,
            log_name: str = "",):
         super().__init__(
            system_name=system_name,
            environment=BaseEnvironment(),
            prompt=prompt,
            model_config=model_config,
            tools=tools,
            maximum_loops=maximum_loops,
            log_name=log_name,
         )

    def messages_initialization(self):
        self.messages = [{"role": "system", "content": USER_PROMPT_TEMPLATE.format(prompt=self.prompt)}]

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
                    meta_data.output = output_message.content or EMPTY_PLACEHOLDER
                terminate = True
            loop_counter += 1
        if loop_counter >= self.maximum_loops and meta_data.output == "":
            self.logger.warning(f"Maximum loops reached ({self.maximum_loops}) without termination. Returning empty output.")
            meta_data.output = OUTPUT_TOO_MANY_ACTIONS
        self.logger.debug(f"Returning {meta_data}")
        return meta_data