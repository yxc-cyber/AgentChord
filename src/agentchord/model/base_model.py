from typing import List, Optional, Union

from litellm import CustomStreamWrapper, Message, ModelResponse, completion

from .model_config import ModelConfig


class BaseModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def completion(
            self,
            messages: List[Union[dict, Message]],
            tools: Optional[list] = None,
            tool_choice: Optional[str] = None
        ) -> Union[ModelResponse, CustomStreamWrapper]:
        if self.config.enable_thinking is not None and self.config.enable_thinking == False:
            messages_no_think = list()
            for message in messages:
                if isinstance(message, dict):
                    if message.get("role") == "system":
                        content = message.get("content", "")
                        content_no_think = content + " /no_think"
                        message_no_think = {"role": "system", "content": content_no_think}
                        messages_no_think.append(message_no_think)
                    else:
                        messages_no_think.append(message)
                elif isinstance(message, Message):
                    if message.role == "system":
                        content = message.content
                        content_no_think = content + " /no_think"
                        message_no_think = Message(role="system", content=content_no_think)
                        messages_no_think.append(message_no_think)
                    else:
                        messages_no_think.append(message)
            return completion(messages=messages_no_think, tools=tools, tool_choice=tool_choice, **(self.config.get_client_configuration()))
        return completion(messages=messages, tools=tools, tool_choice=tool_choice, **(self.config.get_client_configuration()))