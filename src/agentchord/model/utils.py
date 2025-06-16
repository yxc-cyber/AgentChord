from typing import Literal, Optional, Union

from litellm import Message

from ..gbc_object import GBCList, GBCStr

# Gradient strategy constants
FINEGRAINED = 0
SUM_SQUARES = 1
GRADIENT_STRATEGIES = {
    "finegrained": FINEGRAINED,
    "sum_squares": SUM_SQUARES,
}

# Connection strategy constants
MEAN_PRODUCT_INPUT = 0
MAX_PRODUCT_INPUT = 1
MEAN_L1_NORM = 2
MAX_L1_NORM = 3
CONNECTION_STRATEGIES = {
    "mean_product_input": MEAN_PRODUCT_INPUT,
    "max_product_input": MAX_PRODUCT_INPUT,
    "mean_l1_norm": MEAN_L1_NORM,
    "max_l1_norm": MAX_L1_NORM,
}

# Custom message class for handling GBC objects
class Message(Message):
    """
    Custom Message class that extends litellm.Message to handle GBC objects.
    It stores the GBC content in a separate attribute `gbc_content` for compatibility with litellm while storing the string representation in `content`.
    """
    """
        content: Optional[str] = None,
        role: Literal["assistant"] = "assistant",
        function_call=None,
        tool_calls: Optional[list] = None,
    """
    def __init__(
            self,
            content: Optional[Union[str, GBCStr]] = None,
            role: Literal["assistant"] = "assistant",
            function_call=None,
            tool_calls: Optional[list] = None,
            **kwargs,
        ):
        super().__init__(
            content=str(content) if isinstance(content, GBCStr) else content,
            role=role,
            function_call=function_call,
            tool_calls=list(tool_calls) if isinstance(tool_calls, GBCList) else tool_calls,
            **kwargs
        )
        self._gbc_content = content if isinstance(content, GBCStr) else None
        self._gbc_tool_calls = tool_calls if isinstance(tool_calls, GBCList) else None

    @property
    def gbc_content(self) -> Optional[GBCStr]:
        """
        Returns the GBC content if it exists, otherwise returns None.
        """
        return self._gbc_content
    
    @gbc_content.setter
    def gbc_content(self, value: Optional[GBCStr]) -> None:
        """
        Sets the GBC content.
        """
        if isinstance(value, GBCStr):
            self.content = str(value)
            self._gbc_content = value
        else:
            self._gbc_content = None

    @property
    def gbc_tool_calls(self) -> Optional[GBCList]:
        """
        Returns the GBC tool calls if they exist, otherwise returns None.
        """
        return self._gbc_tool_calls
    
    @gbc_tool_calls.setter
    def gbc_tool_calls(self, value: Optional[GBCList]) -> None:
        """
        Sets the GBC tool calls.
        """
        if isinstance(value, GBCList):
            self.tool_calls = list(value)
            self._gbc_tool_calls = value
        else:
            self._gbc_tool_calls = None


# Json parsing utilities
def parse_json_string(json_string: str) -> Optional[Union[dict, list]]:
    """
    Parse a JSON string and return the corresponding Python object.
    """
    import json
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        import re
        json_format = re.compile(r"```(json)?(.*)```", re.DOTALL)
        match = json_format.match(json_string)
        if match:
            json_content = match.group(2).strip()
            return json.loads(json_content)
        else:
            return None