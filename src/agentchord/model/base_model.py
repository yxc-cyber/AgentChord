from typing import List, Optional, Union

from litellm import CustomStreamWrapper, Message, ModelResponse, completion

from ..utils import ModelConfig


class BaseModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def completion(self, messages: List[Union[dict, Message]], tools: Optional[list] = None, tool_choice: Optional[str] = None) -> Union[ModelResponse, CustomStreamWrapper]:
        return completion(messages=messages, tools=tools, tool_choice=tool_choice, **(self.config.get_client_configuration()))