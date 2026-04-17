import uuid
from typing import List, Optional, Union

from litellm import (
    ChatCompletionMessageToolCall,
    Choices,
    CustomStreamWrapper,
    Message,
    ModelResponse,
)

from .base_model import BaseModel
from .model_config import ModelConfig


class DummyModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
    def completion(
        self,
        messages: List[Union[dict, Message]],
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        dummy_weights: bool = False,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        print(f"Messages: \n{messages}")
        content = input("Content: \n")
        if not content:
            tool_choice = input("Tool choice: \n")
            tool_arguments = input("Tool arguments: \n")
            function = {"function": {"name": tool_choice, "arguments": tool_arguments}}
            return ModelResponse(
                choices=[
                    Choices(
                        finish_reason="stop",
                        index=0,
                        message=Message(
                            content=None,
                            tool_calls=[ChatCompletionMessageToolCall(id=str(uuid.uuid4()), type="function", function=function)],
                        )
                    )
                ]
            )
        else:
            return ModelResponse(
                choices=[
                    Choices(
                        finish_reason="stop",
                        index=0,
                        message=Message(
                            content=content,
                            tool_calls=[]
                        )
                    )
                ]
            )